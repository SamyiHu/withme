"""混沌注入层（原则 A/B）：哈希种子 -> 平滑有界噪声。

seed_t = Hash(role_id, floor(t/seed_tau), memory_tension_hash, relation_hash, last_action)
噪声只进入状态层，幅度不超过 epsilon_base，且相邻步之间做平滑，
避免每 tick 硬随机导致"人格抽风"。
"""
import math

from .state_dims import DIMS
from .utils.hash import hash_to_unit, stable_hash


def make_seed(role_id: str, t_seconds: float, seed_tau: int,
              memory_tension_hash: int, relation_hash: int, last_action: str) -> int:
    slot = int(t_seconds // seed_tau)
    return stable_hash(role_id, slot, memory_tension_hash, relation_hash, last_action)


def epsilon(seed: int, epsilon_base: float) -> float:
    """种子映射为平滑相位，再取正弦得到 [-epsilon_base, epsilon_base] 的扰动幅度。"""
    phase = (seed % 1000003) / 1000003.0 * 2.0 * math.pi
    return epsilon_base * math.sin(phase)


def noise_vector(seed: int, dims=None):
    dims = dims or DIMS
    return [hash_to_unit(seed, i) for i in range(len(dims))]


def delta_noise(seed: int, epsilon_base: float, dims=None,
                prev_vec=None, smoothing: float = 0.8):
    """返回 (delta_vector, smoothed_vec)。|delta_i| <= epsilon_base 恒成立。"""
    dims = dims or DIMS
    raw = noise_vector(seed, dims)
    if prev_vec is None:
        vec = raw
    else:
        vec = [smoothing * p + (1.0 - smoothing) * r for p, r in zip(prev_vec, raw)]
    eps = epsilon(seed, epsilon_base)
    return [eps * v for v in vec], vec
