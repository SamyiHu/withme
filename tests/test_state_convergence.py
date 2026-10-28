"""P1 验收：无事件长时间运行状态不爆炸、距离收敛回 tolerance；
epsilon_base 调大时偏离增加但仍被回拉约束。"""
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import copy_repo_configs, make_tmpdir, rmtree, run_tests

from src.config import load_config, load_persona
from src.state_dims import DIMS, clamp_vector
from src.state_engine import StateEngine

T0 = 1790000000.0


def _engine(tmp, **overrides):
    cfg = load_config(os.path.join(copy_repo_configs(tmp), "config.yaml"))
    persona = load_persona(os.path.join(tmp, "config", "persona.yaml"))
    for k, v in overrides.items():
        if k == "epsilon_base":
            cfg.chaos.epsilon_base = v
        elif k == "alpha":
            cfg.convergence.alpha = v
    return StateEngine(cfg, persona)


def _distance_track(engine, row, hours, sample_minutes=30):
    """演化 hours 小时，沿途按 sample_minutes 采样吸引子距离。"""
    dists, t = [], row["last_tick"]
    end = t + hours * 3600
    while t < end:
        nxt = min(t + sample_minutes * 60, end)
        row = engine.lazy_evolve(row, nxt)
        t = nxt
        dists.append(engine.attractor.distance(row["state_vector"]))
    return row, dists


def test_converge_from_far_offset():
    tmp = make_tmpdir()
    try:
        engine = _engine(tmp)
        row = engine.new_row(T0)
        # 强行把每个维度推离吸引子中心 +0.45（valence 会触到上界）
        for name in DIMS:
            row["state_vector"][name] = engine.attractor.entries[name].center + 0.45
        row["state_vector"] = clamp_vector(row["state_vector"])
        start = engine.attractor.distance(row["state_vector"])
        assert start > engine.attractor.reference_tolerance
        row, dists = _distance_track(engine, row, hours=6)
        tol = engine.attractor.reference_tolerance
        assert dists[-1] <= tol, f"6 小时后未收敛：{dists[-1]:.3f} > tol {tol:.3f}"
        # 回拉显著：第一个采样点就已大幅靠近吸引子，且不再回到初始偏离量级
        assert dists[0] < 0.5 * start
        assert max(dists) < 0.5 * start
        # 逐维也大多回到容差内
        within = engine.attractor.per_dim_within_tolerance(row["state_vector"])
        assert sum(within.values()) >= 6, within
    finally:
        rmtree(tmp)


def test_long_run_stable_within_tolerance():
    tmp = make_tmpdir()
    try:
        engine = _engine(tmp)
        row, dists = _distance_track(engine, engine.new_row(T0), hours=48)
        tol = engine.attractor.reference_tolerance
        assert max(dists) < 2.2 * tol, f"长期运行出现越界漂移：max={max(dists):.3f}"
        assert statistics.median(dists) <= tol, \
            f"距离中位数 {statistics.median(dists):.3f} 超过 tolerance {tol:.3f}"
        # 后 24h 与前 24g 分布相近 => 平稳，不是随机游走
        assert statistics.median(dists[24 * 2:]) <= 1.3 * statistics.median(dists[: 24 * 2]) + 1e-9
    finally:
        rmtree(tmp)


def test_larger_epsilon_more_deviation_but_bounded():
    tmp = make_tmpdir()
    try:
        base_engine = _engine(tmp, epsilon_base=0.05)
        big_engine = _engine(tmp, epsilon_base=0.30)
        _, d_small = _distance_track(base_engine, base_engine.new_row(T0), hours=24)
        _, d_big = _distance_track(big_engine, big_engine.new_row(T0), hours=24)
        assert statistics.median(d_big) > statistics.median(d_small), \
            "epsilon 调大偏离应当增加"
        tol = big_engine.attractor.reference_tolerance
        assert max(d_big) < 3.0 * tol + 0.2, "大噪声下仍应有界（回拉 + 维度 clamp）"
        half = len(d_big) // 2
        assert statistics.median(d_big[half:]) <= 1.5 * statistics.median(d_big[:half]) + 1e-9, \
            "大噪声下不应持续发散"
    finally:
        rmtree(tmp)


def test_attractor_helpers():
    tmp = make_tmpdir()
    try:
        engine = _engine(tmp)
        sv = dict(engine.persona.baseline_state)
        assert engine.attractor.distance(sv) < 1e-9
        assert abs(engine.attractor.consistency(sv) - 1.0) < 1e-9
        sv["energy"] = 0.0
        assert 0.0 < engine.attractor.consistency(sv) < 1.0
        pull = engine.attractor.pull(sv, alpha=0.15, max_pull=0.2)
        assert pull["energy"] > 0  # 偏低时回拉向上
        assert abs(pull["energy"]) <= 0.2
    finally:
        rmtree(tmp)


def test_event_impulse_direction():
    tmp = make_tmpdir()
    try:
        engine = _engine(tmp)
        row = engine.new_row(T0)
        after = engine.apply_event_impulse(row, "user_message")
        assert after["state_vector"]["social_need"] < row["state_vector"]["social_need"]
        assert after["state_vector"]["arousal"] > row["state_vector"]["arousal"]
        neg = engine.apply_event_impulse(row, "user_negative_reply")
        assert neg["state_vector"]["valence"] < row["state_vector"]["valence"]
        # 脉冲被 clamp 在维度取值范围内
        assert all(0 <= neg["state_vector"][d] <= 1 for d in DIMS if d != "valence")
        assert -1 <= neg["state_vector"]["valence"] <= 1
        assert row["state_vector"] == engine.new_row(T0)["state_vector"]  # 不改原行
    finally:
        rmtree(tmp)


if __name__ == "__main__":
    sys.exit(run_tests(globals(), "test_state_convergence"))
