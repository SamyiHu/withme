"""风格曲线控制器验收：反馈实时修正 + 向基线回归保稳定（用户设计）。

唠叨治理：负反馈抬 concise -> 可机检 LEN 硬约束（state_style 链强制执行），
积极互动放松，每条向基线回归 10%——修正快但漂不走。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import make_tmpdir, rmtree, run_tests

from src import style_ctrl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROLE = "xiaoman"


def test_feedback_moves_curve():
    b = {"concise": 0.55}
    # 明确嫌长：大幅收紧
    b = style_ctrl.update(b, "你话太多了，说重点", 80)
    assert b["concise"] >= 0.7, b
    # 单字敷衍：隐性负反馈
    b2 = style_ctrl.update({"concise": 0.55}, "哦", 80)
    assert b2["concise"] > 0.6, b2
    # 积极长回复：放松
    b3 = style_ctrl.update({"concise": 0.7}, "哈哈我跟你讲这事儿可长了，" * 3, 30)
    assert b3["concise"] < 0.7, b3


def test_curve_regresses_to_baseline():
    """稳定器：无新反馈时逐步回到基线附近，不会卡死在极端。"""
    b = {"concise": 0.95}
    for _ in range(30):
        b = style_ctrl.update(b, "正常的消息", 50)
    assert abs(b["concise"] - 0.5) < 0.06, b


def test_constraints_ladder():
    assert style_ctrl.constraints(0.85) == ["LEN<=16", "EXCL<=0"]
    assert style_ctrl.constraints(0.70) == ["LEN<=24"]
    assert style_ctrl.constraints(0.58) == ["LEN<=36"]
    assert style_ctrl.constraints(0.40) == []


def test_agent_applies_curve_end_to_end():
    """端到端：被嫌唠叨后，下一轮 prompt 带上 LEN 硬约束（style_audit 可审计）。"""
    import json
    from src.main import Agent

    class Client:
        name = "fake"
        last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "estimated": True}

        def __init__(self):
            self.prompts = []

        def render(self, prompt, seed):
            self.prompts.append(prompt)
            return "嗯，在"

        def render_raw(self, text, seed, *, temperature=0.3, max_tokens=250):
            return ""

    tmp = make_tmpdir()
    try:
        client = Client()
        agent = Agent(config_dir=os.path.join(ROOT, "config"),
                      db_path=os.path.join(tmp, "a.db"), llm_client=client)
        agent.on_user_message("随便聊聊", now=1000.0)
        agent.on_user_message("你太啰嗦了，少说点", now=1100.0)
        meta = json.loads(agent.conn.execute(
            "SELECT meta FROM messages WHERE sender='agent' ORDER BY id DESC LIMIT 1"
        ).fetchone()["meta"])
        style = meta.get("style") or {}
        assert any(str(c).startswith("LEN<=") for c in style.get("constraints", [])), \
            f"嫌唠叨后必须有 LEN 硬约束：{style}"
        assert style.get("style_bias", {}).get("concise", 0) > 0.6
    finally:
        rmtree(tmp)


if __name__ == "__main__":
    sys.exit(run_tests(globals(), name="test_style_ctrl"))
