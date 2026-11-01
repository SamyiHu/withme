"""成长外环：把经历估计成目标 P*，让中心偏移 p 缓慢收敛过去。

与状态层分工（见 docs/growth.md）：
- 状态层（秒~分钟）：7 维状态向吸引子中心回拉，有界收敛；
- 成长层（天~周）：本模块，p 向用户行为拟合出的 P* 收敛，同样有界。

两个关键性质：
- 会饱和：步长 η 随证据量 n 衰减，重复同一批数据不会无限移动；
- 有包络：p 每维不超过 ±growth_envelope，且不会把中心推出维度取值范围。
"""
import math

from .state_dims import DIMS, RANGES


def _num(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def dim_envelope(dim: str, baseline: dict, envelope: float) -> float:
    """单维包络：不只看 ±E，还要保证 base+p 不超出维度取值范围。"""
    lo, hi = RANGES[dim]
    base = _num(baseline.get(dim), 0.0)
    return max(0.0, min(float(envelope), base - lo, hi - base))


def estimate_target(evidence: dict, baseline: dict, envelope: float,
                    llm_shift: dict = None) -> dict:
    """行为证据 -> 各维目标偏移（相对角色卡，单位与状态一致）。

    信号都先归一到 [0,1] 或 [-1,1]，再映射到小幅偏移；LLM 的 trait_shift
    只作为 0.3 权重的语义修正，避免模型一句话把人格带跑。
    """
    rel = evidence.get("relation") or {}
    stats = evidence.get("active_stats") or {}
    intimacy = (_num(rel.get("affinity"), 0.5) + _num(rel.get("trust"), 0.5)
                + _num(rel.get("familiarity"), 0.3)) / 3.0
    sentiment = _clamp(_num(stats.get("avg_sentiment"), 0.0), -1.0, 1.0)
    total = _num(stats.get("total"), 0.0)
    rejected = _num(stats.get("rejected"), 0.0)
    reject_rate = (rejected / total) if total > 0 else 0.0
    depth = _clamp(_num(evidence.get("depth"), 0.4), 0.0, 1.0)

    offsets = {
        "energy": 0.10 * (depth - 0.5) * 2.0,
        "social_need": 0.20 * (intimacy - 0.5) * 2.0,
        "curiosity": 0.15 * (depth - 0.5) * 2.0,
        "care": 0.20 * (intimacy - 0.5) * 2.0 - 0.15 * reject_rate,
        "expression": 0.15 * (depth - 0.5) * 2.0 - 0.15 * reject_rate,
        "arousal": 0.10 * abs(sentiment),
        "valence": 0.18 * sentiment,
    }
    shift = llm_shift or {}
    out = {}
    for dim in DIMS:
        raw = offsets.get(dim, 0.0) + 0.3 * _num(shift.get(dim), 0.0)
        env = dim_envelope(dim, baseline, envelope)
        out[dim] = _clamp(raw, -env, env)
    return out


def advance(state: dict, evidence: dict, baseline: dict, cfg,
            llm_shift: dict = None, now: float = 0.0) -> dict:
    """推进一次成长外环，返回新的 p / P* / n。

    state: {"p": {...}, "target": {...}, "n": int}
    """
    state = state or {}
    p = {d: _num((state.get("p") or {}).get(d), 0.0) for d in DIMS}
    target = {d: _num((state.get("target") or {}).get(d), 0.0) for d in DIMS}
    n = int(state.get("n") or 0)

    batch = max(1, int(evidence.get("new_user_msgs") or 1))
    est = estimate_target(evidence, baseline,
                          _num(getattr(cfg, "growth_envelope", 0.25), 0.25),
                          llm_shift)

    alpha = _num(getattr(cfg, "growth_alpha", 0.25), 0.25) * min(1.0, batch / 6.0)
    alpha = _clamp(alpha, 0.0, 1.0)
    n_new = n + batch
    # 基础步长随证据指数衰减（会饱和）；只有目标或当前值真的偏离时，才把
    # 步长抬到可塑性下限——用户行为变了还能慢慢跟上，稳定时则几乎不动。
    eta = _num(getattr(cfg, "growth_eta0", 0.5), 0.5) * math.exp(
        -n_new / max(1.0, _num(getattr(cfg, "growth_kappa", 200.0), 200.0)))
    active_gap = max(max(abs(est[d] - target[d]), abs(target[d] - p[d]))
                     for d in DIMS)
    wake_gap = max(0.0, _num(getattr(cfg, "growth_wake_gap", 0.02), 0.02))
    if active_gap >= wake_gap:
        eta = max(eta, _num(getattr(cfg, "growth_eta_min", 0.01), 0.01))
    max_step = max(0.0, _num(getattr(cfg, "growth_max_step", 0.03), 0.03))

    new_p, new_target = {}, {}
    for dim in DIMS:
        new_target[dim] = target[dim] + alpha * (est[dim] - target[dim])
        step = _clamp(eta * (new_target[dim] - p[dim]), -max_step, max_step)
        env = dim_envelope(dim, baseline,
                           _num(getattr(cfg, "growth_envelope", 0.25), 0.25))
        new_p[dim] = _clamp(p[dim] + step, -env, env)
    return {"p": new_p, "target": new_target, "n": n_new,
            "updated_at": now}


_WORD_UP = {
    "energy": "更有精神", "social_need": "更想有人陪", "curiosity": "更好奇",
    "care": "更会照顾人", "expression": "更愿意说", "arousal": "情绪更活跃",
    "valence": "心情更好",
}
_WORD_DOWN = {
    "energy": "更安静省电", "social_need": "更习惯独处", "curiosity": "更挑话题",
    "care": "关心更克制", "expression": "话更少", "arousal": "情绪更平",
    "valence": "心情更沉",
}
_DIM_CN = {
    "energy": "精力", "social_need": "社交需求", "curiosity": "好奇",
    "care": "关心", "expression": "表达欲", "arousal": "唤醒度",
    "valence": "情绪效价",
}


def profile_text(state: dict, baseline: dict, cfg, min_n: int = 40,
                 deadband: float = 0.02) -> str:
    """收敛后的长期画像摘要（提示词用，≤220 字，没长成就返回空串）。"""
    state = state or {}
    n = int(state.get("n") or 0)
    if n < min_n:
        return ""
    p = state.get("p") or {}
    moved = [(d, _num(p.get(d), 0.0)) for d in DIMS
             if abs(_num(p.get(d), 0.0)) >= deadband]
    if not moved:
        return ""
    moved.sort(key=lambda item: -abs(item[1]))
    moved = moved[:3]
    phrases = []
    for dim, value in moved:
        word = _WORD_UP[dim] if value > 0 else _WORD_DOWN[dim]
        phrases.append(f"{word}（{_DIM_CN[dim]} {value:+.2f}）")
    return "长期相处下来，我比出厂时" + "、".join(phrases) + "。核心性格和底线没变。"
