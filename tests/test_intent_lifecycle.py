"""意图闭环验收：用户取消/完成后，未完成意图必须关闭，
不再进入 Prompt 的"未完成意图"，避免已放弃的计划被反复提起。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import make_tmpdir, rmtree, run_tests

from src.main import Agent

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _agent(tmp) -> Agent:
    return Agent(config_dir=os.path.join(ROOT, "config"),
                 db_path=os.path.join(tmp, "agent.db"))


def test_cancel_closes_intent():
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp)
        agent.on_user_message("我打算周末去学冲浪", now=1000.0)
        intents = agent.memory.open_intents("xiaoman")
        assert len(intents) == 1
        # 用户取消（否定词命中意图内容）
        agent.on_user_message("不冲了吧，台风要来了", now=2000.0)
        assert agent.memory.open_intents("xiaoman") == [], "取消后意图必须关闭"
        # 留下事件记忆说明取消原因，且不再出现在未完成意图里
        events = [m for m in agent.memory.retrieve("xiaoman", "冲浪", 5, 0.3)
                  if "已取消" in m["content"]]
        assert events, "应留下「已取消」事件记忆"
    finally:
        rmtree(tmp)


def test_done_closes_intent():
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp)
        agent.on_user_message("我准备考驾照，暑假报名", now=1000.0)
        assert len(agent.memory.open_intents("xiaoman")) == 1
        agent.on_user_message("驾照搞定了，科四满分", now=3000.0)
        assert agent.memory.open_intents("xiaoman") == []
    finally:
        rmtree(tmp)


def test_unrelated_cancel_keeps_intent():
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp)
        agent.on_user_message("我打算周末去学冲浪", now=1000.0)
        # 有取消词但与意图无任何重叠 -> 不应误关
        agent.on_user_message("游戏氪金计划算了，不搞了", now=2000.0)
        intents = agent.memory.open_intents("xiaoman")
        assert len(intents) == 1, "无关取消不得误关意图"
    finally:
        rmtree(tmp)


def test_recent_dialogue_in_prompt():
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp)
        agent.on_user_message("今天上班好累", now=1000.0)
        result = agent.on_user_message("那肯定的，你在干嘛", now=1100.0)
        # 模型侧可见的对话尾巴应包含上一轮双方发言（此处以 mock prompt 重建验证结构）
        dialogue = agent.recent_messages(4, before_id=999999)
        assert any("上班好累" in m["content"] for m in dialogue)
        base_row = agent.engine.new_row(0)
        builder_prompt = agent.builder.build(
            intent="respond", source=None,
            state_vector=base_row["state_vector"],
            relation_state=base_row["relation_state"],
            memories=[], open_intents=[], permission_level="L2",
            last_active_effect="尚无记录", recent_user_message="那肯定的，你在干嘛",
            recent_dialogue=dialogue)
        assert "最近对话" in builder_prompt
        tail = builder_prompt["最近对话"]
        assert len(tail) <= 16 and all(len(line) <= 82 for line in tail)
        assert any(line.startswith("你：") for line in tail)
        assert any(line.startswith("我：") for line in tail)
        # 无尾巴时该节为空列表（键恒在，schema 稳定利于前缀缓存）
        p2 = agent.builder.build(
            intent="respond", source=None,
            state_vector=base_row["state_vector"],
            relation_state=base_row["relation_state"],
            memories=[], open_intents=[], permission_level="L2",
            last_active_effect="尚无记录")
        assert p2["最近对话"] == []
    finally:
        rmtree(tmp)


if __name__ == "__main__":
    sys.exit(run_tests(globals(), "test_intent_lifecycle"))
