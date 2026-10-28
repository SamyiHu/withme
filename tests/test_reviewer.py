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


if __name__ == "__main__":
    sys.exit(run_tests(globals(), name="test_reviewer"))
