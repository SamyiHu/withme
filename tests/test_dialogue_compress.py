"""对话滚动摘要验收：窗口外消息不裸丢——攒到阈值后台折叠进摘要，
回复 prompt 带「更早对话摘要」；失败不推进水位（消息不丢）。

背景（2026-10-27）：此前 prompt 只带最近 16 条原文，更早的直接丢，
模型表现为"失忆 + 复读"（火锅梗连提多次）。"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import make_tmpdir, repo_config_dir, rmtree, run_tests

from src.config import ConfigError, config_from_dict, load_config
from src.main import Agent
from src.storage import get_setting, set_setting

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROLE = "xiaoman"


class FakeClient:
    """render 记录 prompt；render_raw 吐固定摘要（可注入失败/超长场景）。"""

    name = "fake"

    def __init__(self, raw_out="摘要：你们约了周末火锅，还聊了晨跑计划"):
        self.last_usage = {"prompt_tokens": 10, "completion_tokens": 5,
                           "estimated": True}
        self.prompts = []
        self.raw_calls = []
        self._raw_out = raw_out

    def render(self, prompt, seed):
        self.prompts.append(prompt)
        return "嗯哼，在呢"

    def render_raw(self, text, seed, *, temperature=0.3, max_tokens=250):
        self.raw_calls.append(text)
        return self._raw_out


def _agent(tmp, client) -> Agent:
    return Agent(config_dir=os.path.join(ROOT, "config"),
                 db_path=os.path.join(tmp, "agent.db"), llm_client=client)


def _insert_messages(agent, n, start_id_after=0):
    for i in range(1, n + 1):
        agent.conn.execute(
            "INSERT INTO messages(role_id, sender, content, meta, created_at)"
            " VALUES(?,?,?,?,?)",
            (ROLE, "user" if i % 2 else "agent", f"消息{i}", "{}",
             1000.0 + i))
    agent.conn.commit()


def test_compress_triggers_and_stores():
    tmp = make_tmpdir()
    try:
        client = FakeClient()
        agent = _agent(tmp, client)
        _insert_messages(agent, 40)           # 未压缩积压 40 > 触发 30
        agent.maybe_compress_dialogue()
        st = get_setting(agent.conn, f"conv_summary:{ROLE}")
        assert st, "应写入滚动摘要"
        assert st["upto_id"] == 24            # 40 - 16 近窗 = 折叠到第 24 条
        assert "火锅" in st["summary"]
        # 提炼调用看到的是近窗之外的批次：含第 1/24 条、不含第 25 条
        text = [t for t in client.raw_calls if "滚动摘要" in t][-1]
        assert "消息1" in text and "消息24" in text and "消息25" not in text
    finally:
        rmtree(tmp)


def test_group_summary_disabled_prevents_self_poisoning():
    """群聊不生成滚动摘要：多人混在同一 messages 表，摘要会把群友的话和
    自己的气话写成事实，再喂回下一轮（实测摘要留下“阿满明确我谁也不归谁管”）。"""
    tmp = make_tmpdir()
    try:
        client = FakeClient()
        agent = _agent(tmp, client)
        agent.is_group = True
        for i in range(1, 41):
            sender = "user" if i % 2 else "agent"
            content = ("[群友 胖达] 阿满你本来就不归谁管" if sender == "user"
                       else "我谁也不归谁管")
            agent.conn.execute(
                "INSERT INTO messages(role_id, sender, content, meta, created_at)"
                " VALUES(?,?,?,?,?)", (ROLE, sender, content, "{}", 1000.0 + i))
        agent.conn.commit()
        agent.maybe_compress_dialogue()
        assert get_setting(agent.conn, f"conv_summary:{ROLE}") is None, "群聊不得生成摘要"
        assert not client.raw_calls, "群聊摘要不应调用 LLM"
    finally:
        rmtree(tmp)


def test_compress_not_below_trigger():
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp, FakeClient())
        _insert_messages(agent, 20)           # 窗口外积压 4 < 14，不压
        agent.maybe_compress_dialogue()
        assert get_setting(agent.conn, f"conv_summary:{ROLE}") is None
    finally:
        rmtree(tmp)


def test_compress_failure_keeps_watermark():
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp, FakeClient(raw_out=""))   # 提炼失败（mock 同款行为）
        _insert_messages(agent, 40)
        agent.maybe_compress_dialogue()
        assert get_setting(agent.conn, f"conv_summary:{ROLE}") is None, \
            "失败不得推进水位（否则那批消息永久丢失）"
    finally:
        rmtree(tmp)


def test_summary_hard_capped():
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp, FakeClient(raw_out="长" * 2000))
        _insert_messages(agent, 40)
        agent.maybe_compress_dialogue()
        st = get_setting(agent.conn, f"conv_summary:{ROLE}")
        assert len(st["summary"]) == agent.config.dialogue.summary_max_chars
    finally:
        rmtree(tmp)


def test_reply_prompt_carries_early_summary():
    tmp = make_tmpdir()
    try:
        client = FakeClient()
        agent = _agent(tmp, client)
        set_setting(agent.conn, f"conv_summary:{ROLE}",
                    {"summary": "更早：你们聊过火锅", "upto_id": 5,
                     "updated_at": time.time()})
        out = agent.on_user_message("在吗", now=2000.0)
        assert out["reply"]
        respond = [p for p in client.prompts
                   if p.get("本次任务", {}).get("意图") == "respond"]
        assert respond, "应有 respond 渲染"
        prompt = respond[-1]
        assert prompt.get("更早对话摘要") == "更早：你们聊过火锅"
        keys = list(prompt.keys())
        assert keys.index("更早对话摘要") < keys.index("最近对话"), \
            "摘要应排在近窗原文之前（稳定前缀）"
    finally:
        rmtree(tmp)


def test_no_summary_key_without_summary():
    tmp = make_tmpdir()
    try:
        client = FakeClient()
        agent = _agent(tmp, client)
        agent.on_user_message("在吗", now=1000.0)
        respond = [p for p in client.prompts
                   if p.get("本次任务", {}).get("意图") == "respond"]
        assert respond and "更早对话摘要" not in respond[-1]
    finally:
        rmtree(tmp)


def test_config_dialogue_section_and_validation():
    cfg = load_config(os.path.join(repo_config_dir(), "config.example.yaml"))
    assert cfg.dialogue.window >= 6
    assert cfg.dialogue.compress_trigger > cfg.dialogue.window
    for bad in ({"window": 2}, {"compress_trigger": 10}, {"summary_max_chars": 5}):
        try:
            config_from_dict({"dialogue": bad})
            raise AssertionError(f"应拒绝非法 dialogue 配置 {bad}")
        except ConfigError:
            continue


def test_old_dialogue_recall():
    """旧对话回捞（billion-context 的 decompress/search 移植）：窗口外的原句
    按关键词捞回 prompt——摘要负责"还记得"，这里负责"记得准"。"""
    tmp = make_tmpdir()
    try:
        client = FakeClient()
        agent = _agent(tmp, client)
        for i, content in enumerate(
                ["上次说好周末去吃火锅", "火锅记得订位", "水群", "水群",
                 "水群", "水群", "水群", "水群", "水群", "水群",
                 "水群", "水群", "水群", "水群", "水群", "水群",
                 "水群", "水群"]):   # 前 2 条在 16 窗口之外
            agent.conn.execute(
                "INSERT INTO messages(role_id, sender, content, meta, created_at)"
                " VALUES(?,?,?,?,?)",
                (ROLE, "user" if i % 2 == 0 else "agent", content, "{}",
                 1000.0 + i))
        agent.conn.commit()
        out = agent.on_user_message("上次说的火锅是啥时候来着", now=5000.0)
        assert out["reply"]
        respond = [p for p in client.prompts
                   if p.get("本次任务", {}).get("意图") == "respond"][-1]
        hits = respond.get("旧对话片段") or []
        assert hits and any("火锅" in h for h in hits), "窗口外的火锅原句要捞回来"
        assert all("水群" not in h for h in hits), "无关键词命中的不进"
        # 无关新话题：不注入该节
        client.prompts.clear()
        agent.on_user_message("今天天气咋样", now=6000.0)
        respond = [p for p in client.prompts
                   if p.get("本次任务", {}).get("意图") == "respond"][-1]
        assert "旧对话片段" not in respond
    finally:
        rmtree(tmp)


def test_chime_in_decision():
    """Agent.chime_in：JSON 协议（join=false/异常输出 -> 潜水不落库；
    join=true -> 只发 text，决策理由绝不进正文）。旧自由文本协议也兼容。"""
    tmp = make_tmpdir()
    try:
        cases = [
            ('{"join": false, "worth": 3}', False),                # JSON 潜水
            ('{"join": true, "worth": 8, "text": "哈哈这梗绝了"}', True),   # 高分接话
            ('{"join": true, "worth": 4, "text": "附和一句"}', False),      # 想接但分不够：双门拦下
            ('{"join": true, "text": "没给分"}', False),           # 缺分=可以但不值得
            ('我觉得该接：{"join": true, "worth": 9, "text": "绷不住乐了"}', True),
            ("否", False),                                        # 旧协议潜水
            ("", False),
            ("哈哈这梗绝了，我也遇过", True),                       # 旧协议接话
        ]
        for i, (raw_out, should_reply) in enumerate(cases):
            client = FakeClient(raw_out=raw_out)
            agent = Agent(config_dir=os.path.join(ROOT, "config"),
                          db_path=os.path.join(tmp, f"case{i}.db"),
                          llm_client=client)
            out = agent.chime_in(["[AW]：今天服务器炸了", "[阿澄]：乐，又是周五"],
                                 now=1000.0)
            assert bool(out) == should_reply, f"raw={raw_out!r} -> {out!r}"
            if should_reply:
                assert "join" not in out and "我觉得" not in out, "正文不得带决策痕迹"
            n = agent.conn.execute(
                "SELECT COUNT(*) c FROM messages WHERE meta LIKE '%chime%'"
            ).fetchone()["c"]
            assert n == (1 if should_reply else 0)
    finally:
        rmtree(tmp)


def test_chime_in_prompt_knows_group_name():
    """插嘴提示词要告诉模型"我叫什么"：群昵称/别名被点名时必须接。

    2026-10-27 实测：群里说「机器人都不理我」/「阿满这波…」，模型不知道
    那是自己，按"别人的私事"潜水。"""
    tmp = make_tmpdir()
    try:
        client = FakeClient(raw_out='{"join": true, "worth": 9, "text": "在呢"}')
        agent = _agent(tmp, client)
        out = agent.chime_in(
            ["[深情的胖达]：阿满这波是照着剧本念的",
             "[阿澄]：你什么情况？"],
            now=1000.0, self_names=["阿满", "小满", "机器人"],
            directed=True, owner_name="阿澄")
        assert out == "在呢"
        # 审核 Agent 会在插嘴之后再发一次 render_raw（独立审稿调用），
        # 所以按内容找插嘴提示词，不能假设它是最后一次调用。
        prompt = next(c for c in client.raw_calls if "值不值得你插话" in c)
        assert "阿满" in prompt and "小满" in prompt and "机器人" in prompt
        assert "这条消息在直接点你" in prompt
        assert "阿澄" in prompt and "创造者" in prompt
        assert "信任和依赖" in prompt and "信任不等于盲从" in prompt
        assert "不是客服" in prompt and "代骂服务" in prompt
        # 只是被提到（别人让第三方骂自己）时，不能自我服务化
        client2 = FakeClient(raw_out='{"join": false, "worth": 3}')
        agent2 = _agent(tmp, client2)
        agent2.chime_in(["[群友]：帮我骂 阿满", "[群友]：行"], now=1001.0,
                        self_names=["阿满"], directed=False, about_me=True,
                        owner_name="阿澄")
        assert "不是直接叫你" in next(c for c in client2.raw_calls
                                     if "值不值得你插话" in c)
    finally:
        rmtree(tmp)


if __name__ == "__main__":
    sys.exit(run_tests(globals(), name="test_dialogue_compress"))
