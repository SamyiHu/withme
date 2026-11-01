"""成长外环验收（固定数据集，离线确定性）。

覆盖 docs/growth.md 的验收项：收敛、饱和、目标跟随、包络、核心不变量、
reset 回滚，以及反思接线后中心确实等于「角色卡 + p」。
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import make_tmpdir, rmtree, run_tests

from src import growth
from src.config import ReflectionConfig
from src.llm_client import MockLLMClient
from src.main import Agent
from src.state_dims import DIMS, RANGES
from src.storage import get_setting

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROLE = "xiaoman"

BASE = {"energy": 0.6, "social_need": 0.5, "curiosity": 0.7, "care": 0.7,
        "expression": 0.55, "arousal": 0.4, "valence": 0.35}


def _ev(affinity=0.9, trust=0.8, familiarity=0.7, sentiment=0.5,
        total=4, rejected=0, depth=0.6, n=6):
    return {
        "active_stats": {"total": total, "rejected": rejected,
                         "avg_sentiment": sentiment},
        "relation": {"affinity": affinity, "trust": trust,
                     "familiarity": familiarity},
        "depth": depth,
        "new_user_msgs": n,
    }


def _run(steps, evidence, state=None, cfg=None):
    state = state or {"p": {}, "target": {}, "n": 0}
    cfg = cfg or ReflectionConfig()
    last_delta = 0.0
    for _ in range(steps):
        before = dict(state["p"])
        state = growth.advance(state, evidence, BASE, cfg)
        last_delta = max(abs(state["p"][d] - before.get(d, 0.0)) for d in DIMS)
    return state, last_delta


def test_converges_to_target_on_fixed_dataset():
    """同一批证据反复喂：步长单调变小、最终停在目标附近。"""
    state = {"p": {}, "target": {}, "n": 0}
    cfg = ReflectionConfig()
    deltas = []
    for _ in range(200):
        before = dict(state["p"])
        state = growth.advance(state, _ev(), BASE, cfg)
        deltas.append(max(abs(state["p"][d] - before.get(d, 0.0)) for d in DIMS))
    assert deltas[0] > deltas[5] > deltas[20] > deltas[50], "步长应逐步衰减"
    assert deltas[-1] < 1e-6, f"应已收敛，最后一步 delta={deltas[-1]:.2e}"
    gap = max(abs(state["p"][d] - state["target"][d]) for d in DIMS)
    assert gap < 1e-5, f"p 应贴近目标 P*，gap={gap:.2e}"
    assert all(abs(v) <= cfg.growth_envelope + 1e-9 for v in state["p"].values())


def test_saturation_stops_moving():
    """收敛后继续喂同样的数据：不再移动（有尽头）。"""
    state, _ = _run(200, _ev())
    _, delta = _run(60, _ev(), state)
    assert delta < 1e-6, f"饱和后不应再移动，delta={delta:.2e}"


def test_target_follows_user_change():
    """用户从热络变冷淡：目标翻转，p 在包络内重新收敛到负向。"""
    state, _ = _run(200, _ev(affinity=0.9, trust=0.9, familiarity=0.8,
                             sentiment=0.6, depth=0.8))
    assert state["target"]["valence"] > 0.05
    assert state["p"]["valence"] > 0.02
    cold = _ev(affinity=0.2, trust=0.2, familiarity=0.2,
               sentiment=-0.6, total=8, rejected=5, depth=0.2)
    state, delta = _run(400, cold, state)
    assert state["target"]["valence"] < -0.05
    assert state["p"]["valence"] < -0.02
    assert delta < 1e-6
    assert all(abs(v) <= 0.25 + 1e-9 for v in state["p"].values())


def test_envelope_and_state_range():
    """极端证据也不能把中心推出包络或状态取值范围。"""
    hot = _ev(affinity=1.0, trust=1.0, familiarity=1.0,
              sentiment=1.0, total=1, rejected=0, depth=1.0, n=100)
    cold = _ev(affinity=0.0, trust=0.0, familiarity=0.0,
               sentiment=-1.0, total=4, rejected=4, depth=0.0, n=100)
    for ev in (hot, cold):
        state, _ = _run(300, ev)
        for dim in DIMS:
            lo, hi = RANGES[dim]
            assert abs(state["p"][dim]) <= 0.25 + 1e-9, dim
            assert lo <= BASE[dim] + state["p"][dim] <= hi, dim


def test_profile_text_is_bounded_summary():
    state = {"p": {d: 0.0 for d in DIMS}, "target": {}, "n": 100}
    state["p"].update({"social_need": 0.08, "care": 0.05, "valence": -0.03})
    text = growth.profile_text(state, BASE, ReflectionConfig())
    assert "长期相处" in text and "社交需求" in text
    assert len(text) <= 220
    assert "{" not in text, "画像不应泄漏原始 JSON"
    # 还没长成（证据不足）不注入提示词
    assert growth.profile_text({**state, "n": 5}, BASE, ReflectionConfig()) == ""


def test_prompt_includes_growth_profile_in_stable_prefix():
    tmp = make_tmpdir()
    try:
        agent = _new_agent(tmp)
        bp = agent.builder.build(
            intent="respond", source=None,
            state_vector=dict(agent.persona.baseline_state),
            relation_state={"affinity": 0.5, "trust": 0.5, "familiarity": 0.3,
                            "boundaries": [], "recent_active_effect": "尚无记录"},
            memories=[], open_intents=[], permission_level="L1",
            last_active_effect="尚无记录", growth_profile="长期相处下来，我更想有人陪。",
            now=1000.0)
        keys = list(bp.keys())
        assert bp["长期成长"].startswith("长期相处")
        assert keys.index("长期成长") < keys.index("最近对话"), "长期成长应放稳定前缀"
    finally:
        rmtree(tmp)


class FakeClient(MockLLMClient):
    name = "fake_growth"

    def __init__(self, raw_out):
        super().__init__()
        self.raw_out = raw_out

    def render_raw(self, text, seed, *, temperature=0.3, max_tokens=250):
        return self.raw_out


def _new_agent(tmp, raw_out=None):
    client = FakeClient(raw_out or json.dumps(
        {"insights": ["他最近在赶项目"], "new_rule": "", "trait_shift": {}} ,
        ensure_ascii=False))
    return Agent(config_dir=os.path.join(ROOT, "config"),
                 db_path=os.path.join(tmp, "agent.db"), llm_client=client)


def _insert_messages(agent, n):
    for i in range(1, n + 1):
        agent.conn.execute(
            "INSERT INTO messages(role_id, sender, content, meta, created_at)"
            " VALUES(?,?,?,?,?)",
            (ROLE, "user" if i % 2 else "agent",
             f"今天聊了聊项目{i}，有点累也有点开心", "{}", 1000.0 + i))
    agent.conn.commit()


def test_reflection_wires_growth_state_and_event():
    tmp = make_tmpdir()
    try:
        agent = _new_agent(tmp)
        _insert_messages(agent, agent.config.dialogue.compress_trigger + 10)
        invariants = list(agent.persona.core_invariants)
        result = agent.maybe_reflect(force=True)
        assert result["ran"], result
        state = get_setting(agent.conn, "growth_state")
        assert state and state["n"] > 0
        centers = {e.state_name: e.center for e in agent.persona.attractor}
        for dim in DIMS:
            assert abs(centers[dim] - (agent._base_centers[dim] + state["p"][dim])) < 1e-9
            assert abs(state["p"][dim]) <= agent.config.reflection.growth_envelope + 1e-9
        # 核心不变量/边界不参与成长
        assert agent.persona.core_invariants == invariants
        # 账本留痕
        rows = agent.conn.execute(
            "SELECT kind, state_after FROM growth_events ORDER BY id DESC LIMIT 1").fetchall()
        assert rows and rows[0]["kind"] == "reflection"
        assert json.loads(rows[0]["state_after"])["n"] == state["n"]
    finally:
        rmtree(tmp)


def test_reset_growth_returns_to_factory():
    tmp = make_tmpdir()
    try:
        agent = _new_agent(tmp)
        _insert_messages(agent, agent.config.dialogue.compress_trigger + 10)
        agent.maybe_reflect(force=True)
        agent.apply_settings({"growth": {"reset": True}})
        state = get_setting(agent.conn, "growth_state")
        assert state["p"] == {d: 0.0 for d in DIMS}
        assert state["target"] == {d: 0.0 for d in DIMS}
        assert state["n"] == 0
        centers = {e.state_name: e.center for e in agent.persona.attractor}
        for dim in DIMS:
            assert abs(centers[dim] - agent._base_centers[dim]) < 1e-9
        kinds = [r["kind"] for r in agent.conn.execute(
            "SELECT kind FROM growth_events ORDER BY id").fetchall()]
        assert "reset" in kinds, "回滚也要留审计"
    finally:
        rmtree(tmp)


if __name__ == "__main__":
    sys.exit(run_tests(globals(), "test_growth"))
