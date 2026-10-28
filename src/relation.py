"""关系状态：affinity / trust / familiarity + 最近主动效果。

更新永远限幅（单事件 ±0.05），防止短期反馈把关系推飞。
"""
import math

from .utils.hash import stable_hash

DEFAULT_RELATION = {"affinity": 0.5, "trust": 0.5, "familiarity": 0.3}

PER_EVENT_LIMIT = 0.05


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def update(relation: dict, sentiment: float, *, interacted: bool = True) -> dict:
    """sentiment ∈ [-1,1]：正面回复缓慢升温，负面缓慢降温。"""
    relation = dict(relation)
    sentiment = max(-1.0, min(1.0, sentiment))
    d = math.copysign(PER_EVENT_LIMIT, sentiment) * abs(sentiment)
    relation["affinity"] = clamp(relation["affinity"] + d)
    relation["trust"] = clamp(relation["trust"] + d * 0.6)
    if interacted:
        relation["familiarity"] = clamp(relation["familiarity"] + 0.01)
    return relation


def relation_hash(relation: dict) -> int:
    return stable_hash(tuple(sorted(
        (k, round(float(v), 3)) for k, v in relation.items() if isinstance(v, (int, float)))))


def to_nl(relation: dict) -> str:
    a, t, f = relation["affinity"], relation["trust"], relation["familiarity"]

    def band(v, low, mid):
        return "低" if v < low else ("中" if v < mid else "高")

    return (f"亲近度{band(a, 0.35, 0.65)}（{a:.2f}）、信任度{band(t, 0.35, 0.65)}（{t:.2f}）、"
            f"熟识度{band(f, 0.3, 0.6)}（{f:.2f}）")
