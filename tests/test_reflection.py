"""反思层（成长核心）验收：洞察沉淀、有界人格漂移、规则成长封顶、
阈值触发、漂移后收敛仍成立、reset 回角色卡、失败兜底。全部离线。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import make_tmpdir, rmtree, run_tests

from src.llm_client import MockLLMClient
from src.main import Agent
from src.storage import get_setting

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FakeReflectClient(MockLLMClient):
    name = "fake_reflection"

    def __init__(self, outputs):
        super().__init__()
        self.outputs = list(outputs)
        self.calls = []

    def render_raw(self, text, seed, *, temperature=0.3, max_tokens=250):
        self.calls.append(text)
        return self.outputs.pop(0) if self.outputs else "不是JSON"


def _agent(tmp, outputs=None) -> Agent:
    return Agent(config_dir=os.path.join(ROOT, "config"),
                 db_path=os.path.join(tmp, "agent.db"),
                 llm_client=FakeReflectClient(outputs or []))


def _out(insights=(), new_rule="", shift=None):
    return json.dumps({"insights": list(insights), "new_rule": new_rule,
                       "trait_shift": shift or {}}, ensure_ascii=False)


def test_reflection_applies_and_persists():
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp, [_out(insights=["他最近在赶项目，压力大"],
                                  new_rule="项目忙的时候少提健身目标",
                                  shift={"care": -0.05})])
        result = agent.maybe_reflect(force=True)
        assert result["ran"], result
        a = result["applied"]
        assert a["insights"] and a["new_rule"] and a["trait_shift"] == {"care": -0.05}
        # 洞察成为可检索的 reflection 记忆
        assert any(m["type"] == "reflection" and "赶项目" in m["content"]
                   for m in agent.memory.retrieve("xiaoman", "项目 压力", 5, 0.3))
        # 规则进入聊天习惯
        assert "项目忙的时候少提健身目标" in agent.persona.style["chat_rules"]
        # 漂移生效：care 中心 0.7 -> 0.65
        centers = {e.state_name: e.center for e in agent.persona.attractor}
        assert abs(centers["care"] - 0.65) < 1e-9
        # 审计行
        assert agent.conn.execute("SELECT COUNT(*) c FROM reflections").fetchone()["c"] == 1
        # 持久化：同库新 Agent 恢复漂移
        agent2 = _agent(tmp)
        centers2 = {e.state_name: e.center for e in agent2.persona.attractor}
        assert abs(centers2["care"] - 0.65) < 1e-9
    finally:
        rmtree(tmp)


def test_drift_step_and_total_caps():
    tmp = make_tmpdir()
    try:
        # 每次提议 +0.10：单步封顶 0.05；累计封顶 0.15
        agent = _agent(tmp, [_out(shift={"care": 0.10})] * 5)
        for _ in range(5):
            agent.maybe_reflect(force=True)
        drift = get_setting(agent.conn, "attractor_drift")
        assert abs(drift["care"] - 0.15) < 1e-9, drift
        centers = {e.state_name: e.center for e in agent.persona.attractor}
        assert abs(centers["care"] - (0.7 + 0.15)) < 1e-9
    finally:
        rmtree(tmp)


def test_rule_growth_capped():
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp, [_out(new_rule="规则A"), _out(new_rule="规则B"),
                             _out(new_rule="规则C")])
        for _ in range(3):
            agent.maybe_reflect(force=True)
        rules = agent.persona.style["chat_rules"]
        assert "规则A" in rules and "规则B" in rules and "规则C" not in rules
        assert len(get_setting(agent.conn, "growth_rules")) == 2
    finally:
        rmtree(tmp)


def test_threshold_gates():
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp)
        # 消息不足 -> 不反思
        agent.on_user_message("今天随便聊一句", now=1000.0)
        assert agent.maybe_reflect(now=2000.0) is None
        # 够 6 条消息但间隔不足 -> 不反思
        for i in range(6):
            agent.on_user_message(f"凑数消息第{i}轮，聊聊今天", now=2000.0 + i * 60)
        assert agent.maybe_reflect(now=2100.0) is None
    finally:
        rmtree(tmp)


def test_convergence_holds_after_drift():
    from src.state_dims import DIMS, clamp_vector
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp, [_out(shift={"care": -0.15, "expression": 0.15})])
        agent.maybe_reflect(force=True)
        engine, attractor = agent.engine, agent.engine.attractor
        row = engine.new_row(1000.0)
        for name in DIMS:
            row["state_vector"][name] = attractor.entries[name].center + 0.4
        row["state_vector"] = clamp_vector(row["state_vector"])
        row = engine.lazy_evolve(row, 1000.0 + 6 * 3600)
        assert attractor.distance(row["state_vector"]) <= attractor.reference_tolerance, \
            "漂移后回拉目标移动，但收敛性必须保持"
    finally:
        rmtree(tmp)


def test_reset_growth():
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp, [_out(new_rule="临时规则", shift={"valence": -0.1})])
        agent.maybe_reflect(force=True)
        assert agent.persona.style["chat_rules"][-1] == "临时规则"
        agent.apply_settings({"growth": {"reset": True}})
        assert "临时规则" not in agent.persona.style["chat_rules"]
        centers = {e.state_name: e.center for e in agent.persona.attractor}
        assert abs(centers["valence"] - agent._base_centers["valence"]) < 1e-9
    finally:
        rmtree(tmp)


def test_failure_fallback():
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp, ["模型抽风了，这绝对不是JSON"])
        result = agent.maybe_reflect(force=True)
        assert result["ran"] is False and result["reason"] == "parse_failed"
        assert get_setting(agent.conn, "attractor_drift", {}) in (None, {},)
        assert agent.conn.execute("SELECT COUNT(*) c FROM reflections").fetchone()["c"] == 1
    finally:
        rmtree(tmp)


if __name__ == "__main__":
    sys.exit(run_tests(globals(), "test_reflection"))
