"""双 Agent 架构验收：审核 Agent（前额叶）出门过审 + 能力清单注入 + 失效回忆曲线。

用户架构：互动 Agent（杏仁核，负责表达）+ 审核 Agent（不说话，只做意图猜测/
内容审核/工具调用/披露/风格建议）。输入侧器官已收拢（perception/style_ctrl/
state_style），本批新增输出侧过审与能力曲线。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import make_tmpdir, rmtree, run_tests

from src import reviewer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROLE = "xiaoman"


def test_denial_detection():
    ok_cases = ["那我看看", "[语音] 来喽", "我搜了下，TI2026 落地上海", "嗯嗯在听"]
    deny_cases = [
        "语音条我是真发不了，AI 只能打字",
        "我做不到发语音，只能打字",
        "看不了图，我瞎的",
        "我没能力联网搜索",
        "不好意思，搜索这种事我发不了",
        # 2026-10-29 鱼群真实病例：拟人化编造借口假装没能力
        "语音键坏了，嘴替功能只剩打字版，将就一下？",
        "我语音功能失灵了，你听不到我的声音",
    ]
    ok_cases += ["中午吃火锅", "你猜怎么着", "6"]
    tic_cases = ["诶，我搜了一下没搜到", "好家伙这都行", "行吧那算了。",  # 起手词/句号
                 "今天天气不错。适合出门",                                # 句中句号
                 "行" + chr(10) * 2 + "那就一句一行来"]                    # 短两段式
    from src.reviewer import mechanical_fix
    joined = mechanical_fix("行" + chr(10) * 2 + "那就一句一行来",
                            ["tic_twoline"])
    assert chr(10) not in joined and "，" in joined, joined
    for c in tic_cases:
        assert reviewer.review_output(c)["issues"], f"病灶漏检：{c}"
        assert reviewer.review_output(c)["correction"]
    for c in ok_cases:
        assert reviewer.review_output(c)["ok"], f"误判：{c}"
    for c in deny_cases:
        assert not reviewer.review_output(c)["ok"], f"漏判：{c}"
        assert reviewer.review_output(c)["correction"]


def test_emoji_spam_is_caught_and_stripped():
    """表情刷屏/尴尬笑：规则命中 + 机械清理（一条最多留一个，禁用的直接删）。"""
    bad = ["这也太离谱了😂😂", "行吧😅", "真的假的😂", "好家伙🤣🤣🤣"]
    for b in bad:
        r = reviewer.review_output(b)
        assert "emoji_spam" in r["issues"], b
        fixed = reviewer.mechanical_fix(b, ["emoji_spam"])
        assert fixed.count("😂") == 0 and fixed.count("😅") == 0 and fixed.count("🤣") == 0, fixed
    # 单个正常表情不算毛病
    assert reviewer.review_output("收到了👍")["ok"]
    assert "emoji_spam" not in reviewer.review_output("嗯嗯 我在")["issues"]


def test_capability_block_in_prompt():
    from src.main import Agent

    class Client:
        name = "fake"
        last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "estimated": True}

        def __init__(self):
            self.prompts = []

        def render(self, prompt, seed):
            self.prompts.append(prompt)
            return "嗯"

        def render_raw(self, text, seed, *, temperature=0.3, max_tokens=250):
            return ""

    tmp = make_tmpdir()
    try:
        client = Client()
        agent = Agent(config_dir=os.path.join(ROOT, "config"),
                      db_path=os.path.join(tmp, "a.db"), llm_client=client)
        agent.on_user_message("在吗", now=1000.0)
        p = client.prompts[-1]
        caps = " ".join(p.get("能力清单") or [])
        assert "语音" in caps and "搜索" in caps and "看图" in caps, caps
        assert "能力提醒" not in p, "干净状态不该有回忆注入"
    finally:
        rmtree(tmp)


def test_denial_triggers_rewrite_and_recall_curve():
    from src.main import Agent
    """嘴硬否认 -> 打回重写成合规版；cap_fail 抬升（回忆注入常驻）；
    干净回合让曲线回落撤除。"""

    class StubbornClient:
        name = "fake"
        last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "estimated": True}

        def __init__(self):
            self.prompts = []
            self.renders = 0

        def render(self, prompt, seed):
            self.prompts.append(prompt)
            self.renders += 1
            # 第一次（respond）嘴硬，之后（打回重写）改口
            return ("语音条我是真发不了，只能打字" if self.renders == 1
                    else "[语音] 行吧，那我露一手")

        def render_raw(self, text, seed, *, temperature=0.3, max_tokens=250):
            return ""

    tmp = make_tmpdir()
    try:
        client = StubbornClient()
        agent = Agent(config_dir=os.path.join(ROOT, "config"),
                      db_path=os.path.join(tmp, "a.db"), llm_client=client)
        out = agent.on_user_message("用语音回答我", now=1000.0)
        assert "露一手" in out["reply"] and "发不了" not in out["reply"], \
            "否认版必须被审核打回换成合规版"
        assert agent.get_row(1000.0)["cap_fail"] >= 1, "cap_fail 必须抬升"
        assert any("能力提醒" in p for p in client.prompts), "回忆注入必须出现"
        # 干净回合：曲线回落
        client.renders = 99   # 后续 render 都走"露一手"版（合法）
        client.prompts.clear()
        agent.on_user_message("聊点别的", now=2000.0)
        assert agent.get_row(2000.0)["cap_fail"] == 0, "干净回合要回落"
    finally:
        rmtree(tmp)


def test_substantive_reply_exempt_from_toolong():
    """干货豁免（真实病例 2026-10-29，AI技术冲了群）：362 字技术分析含
    GitHub 链接与列点，被 too_long + 18 字兜底腰斩成 17 字残句——含链接/
    代码/命令/列点的回复是正经内容，不按闲聊长度截。闲聊水话照抓。"""
    cases = [
        "续上哈\n\n先说 python-vsdx 这个库：\n- 优点是纯 python，pip 装完就能用\n"
        "链接 https://github.com/dave-howard/python-vsdx",
        "来了来了\n\nhttps://github.com/dave-howard/vsdx",
        "装法：```pip install vsdx```",
        "三步走：\n- 先解压\n- 改 XML\n- 打回去",
    ]
    for c in cases:
        assert "too_long" not in reviewer.rule_issues(c), c[:30]
    assert "too_long" in reviewer.rule_issues(
        "哈哈哈哈今天天气不错我们出去玩好不好呀，真的是太棒了呀，"
        "要不我们下午就去公园溜达一圈顺便买点吃的回来吧")


def test_product_model_list_exempt_from_toolong():
    """型号清单也是干货（真实病例 2026-11-01，空气净化器那轮）：群里问
    "推荐几款吧"，回复里的"小米4 Pro、米家全效、352 X86C"不被当成闲聊，
    否则 chime 路径没有重写回调，36 字兜底会把清单切成"…型号：小米4"。"""
    cases = [
        "链接我就不发外链了，直接京东搜型号：小米4 Pro、米家全效、352 X86C，"
        "三室一厅客厅用这几款里挑个CADR 600上下的就行",
        "京东搜：352 X86C 或者米家Ultra，CADR都在600上下，预算低点就小米5S双机组合",
        "客厅用CADR 600的，卧室200-300就够，米家4lite滤芯便宜，买之前把耗材价算进去",
    ]
    for c in cases:
        assert len(c) > 36, c
        assert reviewer.is_substantive(c), c
        assert "too_long" not in reviewer.rule_issues(c), c
    # 闲聊水话照抓：没有链接/型号/并列清单的长句仍然是 too_long
    plain = ("哈哈哈哈今天天气不错我们出去玩好不好呀，真的是太棒了呀，"
             "要不我们下午就去公园溜达一圈顺便买点吃的回来吧")
    assert not reviewer.is_substantive(plain)
    assert "too_long" in reviewer.rule_issues(plain)


def test_needs_detail_context_exempts_toolong():
    """长度决策在感知层：intent_type 提问/求助/分享/提出计划 -> needs_detail
    -> 审核不再按闲聊长度抓。闲聊/情绪安抚照抓。"""
    from src.perception import needs_detail
    assert needs_detail({"intent_type": "提问", "urgency": 0.4}) is True
    assert needs_detail({"intent_type": "求助建议"}) is True
    # 分享/提出计划 = 日常分享琐事，必须短说（2026-10-29：@ 回复被
    # "该详细"指令带成两段式讲课，和插嘴短句反差明显）
    assert needs_detail({"intent_type": "分享"}) is False
    assert needs_detail({"intent_type": "提出计划"}) is False
    assert needs_detail({"intent_type": "闲聊", "urgency": 0.3}) is False
    assert needs_detail({"intent_type": "情绪安抚"}) is False
    assert needs_detail({"intent_type": "闲聊", "urgency": 0.8,
                         "topics": ["作业"]}) is True   # 高紧急兜底
    assert needs_detail(None) is False
    # 正文明确要内容：讲故事/继续/一口气讲完——意图分类常归到闲谈，
    # 只按 intent_type 判会漏，随后被 36 字兜底砍成残句（2026-10-30 实测）
    assert needs_detail({"intent_type": "提出计划",
                         "text_head": "编一个平行世界的故事吧，历史的终归不是现代的"}) is True
    assert needs_detail({"intent_type": "闲聊",
                         "text_head": "你讲故事讲完嘛一口气，我不能一句一句看嘞"}) is True
    assert needs_detail({"intent_type": "闲聊", "text_head": "然后呢"}) is True
    # 选品/推荐请求 = 要内容（2026-11-01 真实病例："那推荐几款吧，给出京东连接"
    # 被当闲聊，回复被 36 字兜底切成"…型号：小米4"）
    assert needs_detail({"text_head": "那推荐几款吧，给出京东连接"}) is True
    assert needs_detail({"text_head": "CADR 600的那个呢"}) is False
    assert needs_detail({"intent_type": "分享", "text_head": "今天吃了顿火锅"}) is False
    # 审核：超 36 字的纯文字长回复——闲聊抓、该详细放
    long_plain = ("这个问题我想了挺久的，说实话答案没那么简单，得从好几个角度"
                  "慢慢说才行，你先别急，我捋一捋再跟你讲清楚")
    assert len(long_plain) > 36
    assert "too_long" in reviewer.rule_issues(long_plain, {})
    assert "too_long" not in reviewer.rule_issues(
        long_plain, {"needs_detail": True})
    # 真实病例（2026-10-29）：36 字内的俏皮两段话不该被砍——旧 18 字阈值
    # 把"是奖品太差看不上我"的包袱砍掉，只剩前半句像在闹情绪
    banter = "怎么，我连抽的资格都没有？是奖品太差看不上我"
    assert "too_long" not in reviewer.rule_issues(banter, {})
    # 重写要求与 36 字兜底对齐（旧文案"压到 12 字"会把故事压成一句话）
    assert "36" in reviewer._FIX_TEXT["too_long"]


def test_llm_review_only_on_hard_risks():
    """审核 LLM 只对硬病灶触发（能力否认/AI 自曝/客服腔）；风格小毛病走机械
    修复。留空 llm_purposes 不得回退成"respond 每条都审"——旧行为让每条回复
    多花 2.4~7.7s 等审稿员。显式白名单仍每条都审。"""
    from src.reviewer import ReviewAgent

    class Cfg:
        mode = "llm"
        max_llm_per_day = 0
        llm_purposes = []
        risk_triggers = True
        temperature = 0.1
        max_tokens = 260
        timeout_seconds = 20

    class Client:
        def __init__(self):
            self.calls = 0

        def render_raw(self, text, seed, *, temperature=0.3, max_tokens=250,
                       timeout=None):
            self.calls += 1
            return '{"pass": true, "issues": [], "fix": ""}'

    c1 = Client()
    a = ReviewAgent(Cfg(), c1)
    r = a.review("今天天气不错。适合出门", {})          # 纯风格病灶（句号）
    assert r["llm_used"] is False and c1.calls == 0
    r2 = a.review("我发不了语音，只能打字", {})          # 硬病灶
    assert r2["llm_used"] is True and c1.calls == 1

    class Cfg2(Cfg):
        llm_purposes = ["respond"]

    c2 = Client()
    a2 = ReviewAgent(Cfg2(), c2)
    r3 = a2.review("好的", {})                          # 白名单：每条都审
    assert r3["llm_used"] is True and c2.calls == 1


if __name__ == "__main__":
    sys.exit(run_tests(globals(), name="test_reviewer"))
