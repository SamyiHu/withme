"""反思层（成长核心）验收：洞察沉淀、有界人格漂移、规则成长封顶、
阈值触发、漂移后收敛仍成立、reset 回角色卡、失败兜底。全部离线。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import make_tmpdir, rmtree, run_tests

from src.llm_client import MockLLMClient
from src.main import Agent
from src.state_dims import DIMS
from src.storage import get_setting, set_setting

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
        assert a["insights"] and a["new_rule"]
        assert a["growth"]["n"] > 0
        # 洞察成为可检索的 reflection 记忆
        assert any(m["type"] == "reflection" and "赶项目" in m["content"]
                   for m in agent.memory.retrieve("xiaoman", "项目 压力", 5, 0.3))
        # 规则只进 growth_rules（单一来源），不再混写聊天习惯
        assert get_setting(agent.conn, "growth_rules") == ["项目忙的时候少提健身目标"]
        assert "项目忙的时候少提健身目标" not in agent.persona.style.get("chat_rules", []), \
            "成长规则不得同时写进 chat_rules（否则提示词里会出现两次）"
        # 外环生效：中心 = 角色卡 + p，且 p 在包络内
        state = get_setting(agent.conn, "growth_state")
        assert state and state["n"] > 0
        centers = {e.state_name: e.center for e in agent.persona.attractor}
        for dim in DIMS:
            assert abs(centers[dim] - (agent._base_centers[dim] + state["p"][dim])) < 1e-9
            assert abs(state["p"][dim]) <= agent.config.reflection.growth_envelope + 1e-9
        # 审计行
        assert agent.conn.execute("SELECT COUNT(*) c FROM reflections").fetchone()["c"] == 1
        # 持久化：同库新 Agent 恢复漂移
        agent2 = _agent(tmp)
        centers2 = {e.state_name: e.center for e in agent2.persona.attractor}
        for dim in DIMS:
            assert abs(centers2[dim] - (agent._base_centers[dim] + state["p"][dim])) < 1e-9
    finally:
        rmtree(tmp)


def test_growth_envelope_caps_centers():
    tmp = make_tmpdir()
    try:
        # 反复提议 +0.10：旧实现累计封顶 0.15；新外环按包络封顶，且步长饱和
        agent = _agent(tmp, [_out(shift={"care": 0.10})] * 40)
        for _ in range(40):
            agent.maybe_reflect(force=True)
        state = get_setting(agent.conn, "growth_state")
        env = agent.config.reflection.growth_envelope
        assert all(abs(v) <= env + 1e-9 for v in state["p"].values()), state["p"]
        centers = {e.state_name: e.center for e in agent.persona.attractor}
        assert centers["care"] <= agent._base_centers["care"] + env + 1e-9
        assert centers["care"] >= agent._base_centers["care"] - env - 1e-9
    finally:
        rmtree(tmp)


def test_rule_growth_fifo():
    """规则满了不丢弃新规则：淘汰最旧的一条（旧实现到 2 条后静默丢弃，
    规则线就此冻结——2026-10-27 起连续 4 次反思的新规则全部没生效）。"""
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp, [_out(new_rule="规则A"), _out(new_rule="规则B"),
                             _out(new_rule="规则C"), _out(new_rule="规则D")])
        results = [agent.maybe_reflect(force=True) for _ in range(4)]
        growth = get_setting(agent.conn, "growth_rules")
        assert growth == ["规则B", "规则C", "规则D"], growth
        assert results[-1]["applied"].get("new_rule_replaced") == "规则A"
        assert "规则A" not in agent.persona.style.get("chat_rules", [])
    finally:
        rmtree(tmp)


def test_max_extra_rules_configurable():
    """规则上限可在控制台/接口调整，非法值被拒。"""
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp)
        agent.apply_config({"reflection": {"max_extra_rules": 5}})
        assert agent.config.reflection.max_extra_rules == 5
        # 同一处 num() 曾把 payload 当 config 传，审核数字参数也没生效；一并守住
        agent.apply_config({"reviewer": {"max_tokens": 120}})
        assert agent.config.reviewer.max_tokens == 120
        try:
            agent.apply_config({"reflection": {"max_extra_rules": 0}})
            raise AssertionError("应拒绝 0 条")
        except ValueError:
            pass
    finally:
        rmtree(tmp)


def test_growth_rules_single_section_in_prompt():
    """成长规则只出现在【成长规则】段：写入侧封顶后，提示词不再二次截断。"""
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp)
        rules = ["规则1", "规则2", "规则3", "规则4"]
        bp = agent.builder.build(
            intent="respond", source=None,
            state_vector=dict(agent.persona.baseline_state),
            relation_state={"affinity": 0.5, "trust": 0.5, "familiarity": 0.3,
                            "boundaries": [], "recent_active_effect": "尚无记录"},
            memories=[], open_intents=[], permission_level="L1",
            last_active_effect="尚无记录", growth_rules=rules, now=1000.0)
        assert bp["成长规则"] == rules
        assert not any(r in bp["表达风格"]["聊天习惯"] for r in rules), \
            "成长规则不得同时混进聊天习惯（会重复占提示词）"
    finally:
        rmtree(tmp)


def test_legacy_growth_rules_stripped_from_style():
    """老数据里规则同时存在 growth_rules 和 persona_style.chat_rules：
    启动时剥掉后者，避免提示词重复、也让手工删除能生效。"""
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp)
        set_setting(agent.conn, "growth_rules", ["老规则"])
        set_setting(agent.conn, "persona_style", {"chat_rules": ["老规则", "手写规则"]})
        agent2 = _agent(tmp)
        assert agent2.persona.style.get("chat_rules") == ["手写规则"]
        assert get_setting(agent2.conn, "persona_style")["chat_rules"] == ["手写规则"]
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
        assert get_setting(agent.conn, "growth_rules") == ["临时规则"]
        agent.apply_settings({"growth": {"reset": True}})
        assert not get_setting(agent.conn, "growth_rules")
        assert "临时规则" not in agent.persona.style.get("chat_rules", [])
        state = get_setting(agent.conn, "growth_state")
        assert state["p"] == {d: 0.0 for d in DIMS}
        assert state["n"] == 0
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
