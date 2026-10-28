"""P1 验收：epsilon_t 与各维噪声分量都不超过 epsilon_base 上限。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import make_tmpdir, run_tests

from src.chaos import delta_noise, epsilon, make_seed, noise_vector
from src.utils.hash import hash_to_unit, stable_hash

T0 = 1790000000.0


def test_epsilon_bounded():
    for i in range(5000):
        s = stable_hash("xiaoman", i, "seed-test")
        e = epsilon(s, 0.05)
        assert -0.05 - 1e-12 <= e <= 0.05 + 1e-12
    # 幅度确实覆盖到接近上限（正弦不会退化为 0）
    peaks = max(abs(epsilon(stable_hash("xiaoman", i), 0.05)) for i in range(2000))
    assert peaks > 0.045


def test_delta_components_bounded():
    prev = None
    for k in range(1000):
        s = make_seed("xiaoman", T0 + k * 300, 300, k % 7, k % 3, "respond" if k % 2 else "")
        d, prev = delta_noise(s, 0.05, prev_vec=prev, smoothing=0.8)
        assert all(-0.05 - 1e-12 <= x <= 0.05 + 1e-12 for x in d), f"step {k} 越界"
    # 不带平滑同样有界
    for k in range(1000):
        s = stable_hash("xiaoman", k)
        d, _ = delta_noise(s, 0.05)
        assert all(abs(x) <= 0.05 + 1e-12 for x in d)


def test_hash_to_unit_in_range():
    vals = [hash_to_unit(1234567, i) for i in range(200)]
    assert all(-1.0 <= v <= 1.0 for v in vals)
    # 有离散度，不是常数
    assert len(set(vals)) > 100
    assert max(vals) > 0.3 and min(vals) < -0.3


def test_noise_vector_dimension_stable():
    s = stable_hash("dim-test")
    assert len(noise_vector(s)) == 7
    # 每个维度来自独立派生
    assert len(set(noise_vector(s))) == 7


if __name__ == "__main__":
    sys.exit(run_tests(globals(), "test_noise_bounded"))
