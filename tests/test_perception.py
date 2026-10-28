"""洞察层验收：LLM 揣摩意图喂给指标——含蓄取消可识别、情绪度量更准、
失败回退启发式、决策权仍在闸门。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import make_tmpdir, rmtree, run_tests

from src.feedback import analyze_with_perception
from src.llm_client import MockLLMClient
from src.main import Agent
from src.perception import extract_json, perceive
from src.config import PerceptionConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FakeAnalysisClient(MockLLMClient):
    """mock 渲染 + 可编程的 render_raw：离线模拟真实洞察返回。"""
    name = "fake_analysis"

    def __init__(self, raw_json_or_text):
        super().__init__()
        self.raw = raw_json_or_text
        self.calls = []

    def render_raw(self, text, seed, *, temperature=0.3, max_tokens=250):
        self.calls.append(text)
        return self.raw


def _agent(tmp, client) -> Agent:
    return Agent(config_dir=os.path.join(ROOT, "config"),
                 db_path=os.path.join(tmp, "agent.db"), llm_client=client)


def test_extract_json_robust():
    assert extract_json('{"sentiment": 0.5}') == {"sentiment": 0.5}
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('分析如下：\n{"a": 1}\n以上。') == {"a": 1}
    assert extract_json("完全不合法") is None
    assert extract_json("") is None
    assert extract_json('[1,2]') == [1, 2]  # 数组是合法输出（judge 等场景）


def test_perceive_parses_and_clamps():
    ok = FakeAnalysisClient(json.dumps({
        "sentiment": 5.0, "emotion": "超级无敌开心兴奋到飞起", "intent_type": "闲聊",
        "topics": ["团建", "天气"], "urgency": 0.8, "cancel_topics": [], "done_topics": []},
        ensure_ascii=False))
    perc = perceive(ok, "周末团建定了", 1, PerceptionConfig())
    assert perc is not None
    assert perc["sentiment"] == 1.0  # 越界被夹回
    assert perc["urgency"] == 0.8 and perc["intent_type"] == "闲聊"
    assert perc["topics"] == ["团建", "天气"] and len(perc["emotion"]) <= 8
    # 垃圾输出 -> None（主链路回退词表）
    bad = FakeAnalysisClient("今天天气不错，不适合分析")
    assert perceive(bad, "随便", 1, PerceptionConfig()) is None
    # mock 客户端 -> 直接 None
    assert perceive(MockLLMClient(), "x", 1, PerceptionConfig()) is None
    # 配置关闭 -> None
    assert perceive(ok, "x", 1, PerceptionConfig(enabled=False)) is None


def test_subtle_cancellation_recognized():
    """含蓄取消（无任何取消关键词）只有感知层能识别。"""
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp, MockLLMClient())  # 先用 mock 立意图
        agent.on_user_message("我打算下周开始学吉他", now=1000.0)
        assert len(agent.memory.open_intents("xiaoman")) == 1
        # 词表路径识别不了"那件事以后再说吧"
        from src.main import Agent as _A
        agent._update_intent_lifecycle("吉他那件事以后再说吧", now=1100.0)
        assert len(agent.memory.open_intents("xiaoman")) == 1, "词表不应误关"
        # 感知路径：模型判定 cancel_topics=["吉他"]
        perc = {"sentiment": -0.1, "emotion": "平静", "intent_type": "取消计划",
                "topics": ["吉他"], "urgency": 0.2,
                "cancel_topics": ["吉他"], "done_topics": [], "text_head": "吉他那件事"}
        agent._update_intent_lifecycle("吉他那件事以后再说吧", now=1200.0, perc=perc)
        assert agent.memory.open_intents("xiaoman") == [], "感知应关闭含蓄取消的意图"
    finally:
        rmtree(tmp)


def test_perception_feeds_metrics_and_audit():
    tmp = make_tmpdir()
    try:
        raw = json.dumps({"sentiment": 0.8, "emotion": "感激", "intent_type": "分享",
                          "topics": ["升职"], "urgency": 0.4,
                          "cancel_topics": [], "done_topics": []}, ensure_ascii=False)
        client = FakeAnalysisClient(raw)
        agent = _agent(tmp, client)
        r = agent.on_user_message("跟你说个事，我升职啦", now=1000.0)
        # 感知进了审计与返回
        assert r["meta"]["perception"]["intent_type"] == "分享"
        row = agent.conn.execute(
            "SELECT meta FROM messages WHERE sender='user' ORDER BY id DESC LIMIT 1").fetchone()
        assert json.loads(row["meta"])["perception"]["emotion"] == "感激"
        # 情绪更准：词表本会给 0.0，感知给 0.8 -> 关系提升 0.05*0.8=0.04（限幅内）
        before = agent.engine.new_row(0)["relation_state"]["affinity"]
        after = agent.get_row(1000.0)["relation_state"]["affinity"]
        assert abs(after - 0.54) < 1e-9, after
        # 分享类消息显著度提高
        mem = agent.memory.retrieve("xiaoman", "升职", 5, 0.3)
        assert any(m["salience"] == 0.55 for m in mem)
        # valence 被用户情绪拨动（+0.8*0.05=0.04）
        assert agent.get_row(1000.0)["state_vector"]["valence"] > 0.35
    finally:
        rmtree(tmp)


def test_analyze_with_perception_merge():
    base = analyze_with_perception("别提这个了，烦", None)
    assert base["rejected"] is True
    perc_only = analyze_with_perception(
        "这个话题就到这吧",
        {"sentiment": -0.2, "intent_type": "拒绝"})
    assert perc_only["rejected"] is True and perc_only["sentiment"] == -0.6, \
        "感知判定拒绝时情绪至少 -0.6"
    q = analyze_with_perception("帮我看看这个方案",
                                {"sentiment": 0.1, "intent_type": "提问"})
    assert q["is_question"] is True


def test_decision_power_stays_in_gate():
    """感知只改输入：闸门对'是否主动'的判定不受感知直接控制。"""
    tmp = make_tmpdir()
    try:
        raw = json.dumps({"sentiment": 0.9, "emotion": "兴奋", "intent_type": "分享",
                          "topics": ["x"], "urgency": 1.0,
                          "cancel_topics": [], "done_topics": []})
        client = FakeAnalysisClient(raw)
        agent = _agent(tmp, client)
        agent.on_user_message("超开心的一件事，不细说了", now=1000.0)
        # 无意图/无候选：即使感知 urgency=1.0 也不得凭空产生 L2 主动
        result = agent.evaluate_proactive(now=1000.0 + 3600)
        assert result.get("sent") is None, "主动仍必须有来源，感知不能凭空造主动"
    finally:
        rmtree(tmp)


def test_context_enters_analysis():
    """上下文必须进入思考：分析调用里带最近对话与未完成意图。"""
    from src.perception import build_task
    task = build_task("那件事还是算了", {
        "dialogue": ["你：我打算把阳台改成小花园", "我：听起来不错"],
        "open_intents": ["我打算把阳台改成小花园"]})
    assert "【最近对话】" in task and "阳台改成小花园" in task
    assert "【对方提过、还没做完的事】" in task
    assert "【用户最新消息】那件事还是算了" in task
    # 无上下文时不生成空节（分节标记不出现）
    bare = build_task("你好", None)
    assert "【最近对话】" not in bare and "【对方提过" not in bare and "【用户最新消息】你好" in bare

    # 上一轮动作回执进上下文：感知层才能把"再试试"映射回具体动作重发
    retry = build_task("再试试", {"recent_hand":
        'image.draw({"prompt": "米老鼠"}) -> 失败(被上游内容安全拦截了)'})
    assert "【上一轮动作】" in retry and "image.draw" in retry
    assert "重新发起" in retry, "header 必须带重试指引（且限定明确重做请求）"
    bare2 = build_task("你好", None)
    # header 的字段定义里会提到【上一轮动作】，但具体回执内容不该出现
    assert "image.draw" not in bare2

    tmp = make_tmpdir()
    try:
        client = FakeAnalysisClient("{}")  # 解析失败也无妨，只验证注入
        agent = _agent(tmp, client)
        agent.on_user_message("我打算把阳台改成小花园，周末开工", now=1000.0)
        agent.on_user_message("随便聊聊今天", now=1100.0)
        # 第二条消息的分析调用应包含第一条产生的未完成意图
        assert any("还没做完" in c and "阳台" in c for c in client.calls), \
            "分析提示必须带上未完成意图上下文"
    finally:
        rmtree(tmp)


def test_pronoun_cancellation_resolved_by_context():
    """纯代词取消（"那件事还是算了"）：感知用具体话题词闭环意图。"""
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp, MockLLMClient())
        agent.on_user_message("我打算把书房收拾成工作室，周末开工", now=1000.0)
        assert len(agent.memory.open_intents("xiaoman")) == 1
        perc = {"sentiment": -0.2, "emotion": "疲惫", "intent_type": "取消计划",
                "topics": ["书房工作室"], "urgency": 0.2,
                "cancel_topics": ["书房工作室"], "done_topics": [], "text_head": "那件事"}
        agent._update_intent_lifecycle("那件事还是先算了，最近太忙", now=1200.0, perc=perc)
        assert agent.memory.open_intents("xiaoman") == [], "代词取消应经感知还原后关闭意图"
    finally:
        rmtree(tmp)


def test_deferral_keeps_intent_open():
    """推迟≠取消：缓一缓/等发工资 → 意图保留（之后仍可主动问起），只记备注。"""
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp, MockLLMClient())
        agent.on_user_message("我决定把客厅的旧沙发换掉", now=1000.0)
        assert len(agent.memory.open_intents("xiaoman")) == 1
        perc = {"sentiment": -0.2, "emotion": "平静", "intent_type": "推迟计划",
                "topics": ["换沙发"], "urgency": 0.2,
                "cancel_topics": [], "defer_topics": ["客厅换旧沙发"],
                "done_topics": [], "text_head": "这件事先缓一缓"}
        agent._update_intent_lifecycle("这件事先缓一缓吧，等发工资再说", now=1200.0, perc=perc)
        intents = agent.memory.open_intents("xiaoman")
        assert len(intents) == 1, "推迟不得关闭意图"
        notes = [m for m in agent.memory.retrieve("xiaoman", "沙发", 5, 0.3)
                 if "已推迟" in m["content"]]
        assert notes, "应留下「已推迟」备注"
    finally:
        rmtree(tmp)


def test_reply_plan_parsed_and_clamped():
    """拟人思考路径：perception 输出 reply_plan（说什么/怎么说/情绪），
    解析并裁剪；缺字段/非 dict 不炸。"""
    ok = FakeAnalysisClient('{"sentiment": 0.3, "intent_type": "分享", "emotion": "兴奋",'
                       ' "reply_plan": {"what": "接他说的显卡涨价，问最后买了没",'
                       ' "how": "先惊讶再问，两短句", "mood": "惊讶"}}')
    perc = perceive(ok, "显卡终于到货了", 1, PerceptionConfig())
    assert perc["reply_plan"]["what"].startswith("接他")
    assert perc["reply_plan"]["mood"] == "惊讶"
    bad = FakeAnalysisClient('{"sentiment": 0.0, "intent_type": "闲聊", "reply_plan": "随便说说"}')
    perc2 = perceive(bad, "x", 1, PerceptionConfig())
    assert perc2["reply_plan"] is None, "非 dict 的 plan 不进链路"


if __name__ == "__main__":
    sys.exit(run_tests(globals(), "test_perception"))
