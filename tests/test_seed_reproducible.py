"""P1 验收：给定相同 role_id/time/memory_hash 等输入，seed 与演化结果可复现。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import copy_repo_configs, make_tmpdir, rmtree, run_tests

from src import chaos
from src.chaos import delta_noise, make_seed, noise_vector
from src.config import load_config, load_persona
from src.state_engine import StateEngine

T0 = 1790000000.0  # 固定时间基准，所有测试共用


def _engine(tmp):
    cfg = load_config(os.path.join(copy_repo_configs(tmp), "config.yaml"))
    persona = load_persona(os.path.join(tmp, "config", "persona.yaml"))
    return StateEngine(cfg, persona)


def test_same_inputs_same_seed():
    for t in (T0, T0 + 1, T0 + 299.5, T0 + 123456):
        a = make_seed("xiaoman", t, 300, 111, 222, "respond")
        b = make_seed("xiaoman", t, 300, 111, 222, "respond")
        assert a == b
    # 任意两个 seed 落点的噪声向量确定
    s = make_seed("xiaoman", T0, 300, 0, 0, "")
    v1, v2 = noise_vector(s), noise_vector(s)
    assert v1 == v2 and len(v1) == 7
    assert all(-1.0 <= x <= 1.0 for x in v1)


def test_different_inputs_different_seed():
    base = make_seed("xiaoman", T0, 300, 0, 0, "")
    assert make_seed("other", T0, 300, 0, 0, "") != base          # role 不同
    assert make_seed("xiaoman", T0 + 300, 300, 0, 0, "") != base  # 时间槽不同
    assert make_seed("xiaoman", T0, 300, 999, 0, "") != base      # 记忆张力不同
    assert make_seed("xiaoman", T0, 300, 0, 999, "") != base      # 关系不同
    assert make_seed("xiaoman", T0, 300, 0, 0, "respond") != base # 上次动作不同
    # 同一 tau 槽内的两个时刻必须同种子（平滑性的来源）
    assert make_seed("xiaoman", T0, 300, 0, 0, "") == make_seed("xiaoman", T0 + 50, 300, 0, 0, "")


def test_evolution_deterministic():
    tmp = make_tmpdir()
    try:
        engine = _engine(tmp)
        row = engine.new_row(T0)
        a = engine.lazy_evolve(row, T0 + 8 * 3600)
        b = engine.lazy_evolve(row, T0 + 8 * 3600)
        assert a["state_vector"] == b["state_vector"]
        assert a["noise_prev"] == b["noise_prev"]
        # 惰性补算 == 分次补算：噪声种子挂在绝对时间槽上，结果必须一致
        stepped = engine.lazy_evolve(row, T0 + 3600)
        for _ in range(7):
            stepped = engine.lazy_evolve(stepped, stepped["last_tick"] + 3600)
        for dim, v in a["state_vector"].items():
            assert abs(v - stepped["state_vector"][dim]) < 1e-9, dim
        # 演化不修改传入行
        assert row["state_vector"] == engine.new_row(T0)["state_vector"]
        # 上次动作不同 -> 种子必然不同，状态事实上不同
        c = engine.lazy_evolve(row, T0 + 8 * 3600, last_action="share_related_memory")
        assert make_seed("xiaoman", T0 + 300, 300, 0, 0, "share") != \
               make_seed("xiaoman", T0 + 300, 300, 0, 0, "")
        assert any(abs(c["state_vector"][d] - a["state_vector"][d]) > 1e-9
                   for d in a["state_vector"])
    finally:
        rmtree(tmp)


def test_delta_noise_smoothing():
    s1 = make_seed("xiaoman", T0, 300, 0, 0, "")
    s2 = make_seed("xiaoman", T0 + 300, 300, 0, 0, "")
    d1, v1 = delta_noise(s1, 0.05)
    d2, v2 = delta_noise(s2, 0.05, prev_vec=v1, smoothing=0.8)
    # 平滑后的向量仍在 [-1,1]
    assert all(-1.0 <= x <= 1.0 for x in v2)
    # 平滑保留成分
    assert any(abs(0.8 * p + 0.2 * r - q) < 1e-12 for p, r, q in zip(v1, chaos.noise_vector(s2), v2))


if __name__ == "__main__":
    sys.exit(run_tests(globals(), "test_seed_reproducible"))
