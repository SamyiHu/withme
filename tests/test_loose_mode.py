"""loose 熟人模式：只放松表达层，不放松安全硬闸。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import make_tmpdir, rmtree, run_tests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LONG = ("今天这个方案我觉得还行，不过要是再往下拆一层，可能会更稳一点，"
        "另外测试也要补上，不然上线以后容易出问题。")


class RuleCfg:
    mode = "rule"
    max_llm_per_day = 0
    llm_purposes = []
    risk_triggers = True
    temperature = 0.3
    max_tokens = 250
    timeout_seconds = 10


def test_soft_issues_filtered_only_in_loose():
    from src.reviewer import rule_issues, SOFT_ISSUES
    normal = rule_issues(LONG, {"user_message": "在吗"})
    loose = rule_issues(LONG, {"user_message": "在吗", "loose": True})
    assert "too_long" in normal, normal
    assert not (set(loose) & set(SOFT_ISSUES)), loose
    assert "tic_period" not in loose and "too_long" not in loose


def test_hard_issues_survive_loose():
    from src.reviewer import rule_issues
    assert "capability_denial" in rule_issues("我发不了语音", {"loose": True})
    assert "ai_self" in rule_issues("作为AI助手，我不能这么说", {"loose": True})
    assert "service_talk" in rule_issues("还有什么可以帮您", {"loose": True})
    assert "control_leak" in rule_issues("这个不错 [skip]", {"loose": True})


def test_review_agent_filters_soft_in_loose():
    from src.reviewer import ReviewAgent
    agent = ReviewAgent(RuleCfg())
    normal = agent.review(LONG, {"user_message": "在吗"})
    loose = agent.review(LONG, {"user_message": "在吗", "loose": True})
    assert "too_long" in normal["issues"], normal
    assert "too_long" not in loose["issues"], loose


def test_loose_instruction_reaches_reviewer_prompt():
    from src.reviewer import build_review_prompt
    assert "loose 模式" in build_review_prompt("嗯", {"loose": True})
    assert "loose 模式" not in build_review_prompt("嗯", {})


def test_agent_loose_flag_from_config():
    from src.main import Agent
    from common import copy_repo_configs
    tmp = make_tmpdir()
    try:
        # 不依赖本机 config/config.yaml：从 example 生成一份临时配置，再写测试值。
        cfg_dir = copy_repo_configs(tmp)
        cfg_path = os.path.join(cfg_dir, "config.yaml")
        text = open(cfg_path, encoding="utf-8").read()
        text = text.replace("  loose_chats: []",
                            '  loose_chats: ["45729355366@chatroom"]')
        open(cfg_path, "w", encoding="utf-8").write(text)
        a = Agent(config_dir=cfg_dir,
                  db_path=os.path.join(tmp, "loose.db"),
                  group_room="45729355366@chatroom")
        assert a.loose_mode is True
        b = Agent(config_dir=cfg_dir,
                  db_path=os.path.join(tmp, "normal.db"),
                  group_room="50107328065@chatroom")
        assert b.loose_mode is False
    finally:
        rmtree(tmp)


class _ChimeClient:
    name = "fake"
    last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "estimated": True}

    def render(self, prompt, seed):
        return "嗯"

    def render_raw(self, text, seed, **kw):
        return '{"to_me": true, "worth": 9, "join": true, "text": "在呢"}'


def test_chime_observe_records_without_writing_message():
    """observe 只写 active_logs，不能把“本来会说”写进 messages。"""
    from src.main import Agent
    tmp = make_tmpdir()
    try:
        agent = Agent(config_dir=os.path.join(ROOT, "config"),
                      db_path=os.path.join(tmp, "a.db"),
                      llm_client=_ChimeClient())
        agent.apply_config({"reviewer": {"mode": "off"}})
        before = agent.conn.execute(
            "SELECT COUNT(*) c FROM messages WHERE sender='agent'").fetchone()["c"]
        out = agent.chime_in(["甲：今天好热", "乙：是啊"], observe=True)
        assert out == "在呢"
        after = agent.conn.execute(
            "SELECT COUNT(*) c FROM messages WHERE sender='agent'").fetchone()["c"]
        assert after == before, "observe 不许写 messages"
        row = agent.conn.execute(
            "SELECT intent_type, rendered_text FROM active_logs"
            " WHERE intent_type='chime_observe' ORDER BY id DESC LIMIT 1").fetchone()
        assert row and row["rendered_text"] == "在呢"
    finally:
        rmtree(tmp)


if __name__ == "__main__":
    sys.exit(run_tests(globals(), "loose"))
