"""动机与注意力层：主动评分（5.4）+ 候选意图枚举与评分（5.5/5.6）。

原则 E：L2+ 意图必须挂在具体来源（memory / open intent）上；
没有记忆、没有未完成意图时，不可能凭空生成 L2+ 候选。
"""
from dataclasses import dataclass, field

from .intents import INTENTS
from .persona_attractor import PersonaAttractor

# 各意图的关系门槛（不满足则该候选不生成）
RELATION_GATES = {
    "share_related_memory": ("familiarity", 0.15),
    "ask_light_question": ("familiarity", 0.25),
    "remind_open_intent": ("familiarity", 0.20),
    "express_care": ("trust", 0.35),
    "shift_topic": ("affinity", 0.60),
    "propose_small_activity": ("affinity", 0.55),
}

BOUNDARY_RISK = {"shift_topic": 0.15, "propose_small_activity": 0.10}


def _bigrams(text: str) -> set:
    t = "".join(ch for ch in str(text or "") if not ch.isspace())
    return {t[i:i + 2] for i in range(len(t) - 1)} if len(t) >= 2 else {t} if t else set()


def topic_novelty(source_content: str, recent_text: str) -> float:
    """素材相对最近对话的新颖度（重叠系数，0=全新，1=完全复读）。
    字符 2-gram 重叠系数：粗但零依赖，够挡"主动消息接着刚才聊"这种复读。"""
    src, rec = _bigrams(source_content), _bigrams(recent_text)
    if not src or not rec:
        return 1.0
    return len(src & rec) / min(len(src), len(rec))


@dataclass
class Candidate:
    name: str
    level: int
    score: float
    source: dict = None            # {"type": "memory"|"intent", "id": int}
    detail: dict = field(default_factory=dict)

    def to_log(self):
        return {"intent": self.name, "level": self.level, "score": round(self.score, 4),
                "source": f"{self.source['type']}:{self.source['id']}" if self.source else None,
                **{k: round(v, 4) if isinstance(v, float) else v
                   for k, v in self.detail.items()}}


def timing_weight(hours_since_user: float) -> float:
    """主动时机权重：刚聊完不该烦人，太久没聊联系减弱。"""
    if hours_since_user < 0.1:
        return 0.30
    if hours_since_user <= 8:
        return 1.00
    if hours_since_user <= 24:
        return 0.80
    return 0.50


def active_score(state_vector: dict, relation: dict, ctx: dict, weights: dict):
    """返回 (score01, 明细)。score01 已按正向权重和归一到 [0,1]。"""
    tw = timing_weight(ctx["hours_since_user"])
    urges = {
        "social_urge": state_vector["social_need"] * relation["affinity"] * tw,
        "curiosity_urge": state_vector["curiosity"] * ctx["information_gap"],
        "care_urge": state_vector["care"] * ctx["open_intent_salience"],
        "expression_urge": state_vector["expression"] * ctx["topic_relevance"],
    }
    disturb_cost = (0.40 if ctx["quiet_hour"] else 0.0) \
        + 0.30 * min(1.0, ctx["recent_active_count"] / 4.0) \
        + 0.30 * (1.0 - tw) \
        + (0.25 if ctx.get('user_typing') else 0.0)  # 打字勿扰
    pos = sum(weights[k] for k in ("social", "curiosity", "care", "expression"))
    raw = (weights["social"] * urges["social_urge"]
           + weights["curiosity"] * urges["curiosity_urge"]
           + weights["care"] * urges["care_urge"]
           + weights["expression"] * urges["expression_urge"]
           - weights["disturb"] * disturb_cost)
    detail = {**{k: round(v, 4) for k, v in urges.items()},
              "disturb_cost": round(disturb_cost, 4), "timing_weight": tw}
    return max(0.0, min(1.0, raw / pos if pos > 0 else 0.0)), detail


def generate_candidates(state_vector: dict, relation: dict, memories: list,
                        open_intents: list, attractor: PersonaAttractor, ctx: dict,
                        recent_intent_types=(), recent_text: str = "") -> list:
    """枚举有来源支撑的候选意图并打分（5.6）。"""
    hours = ctx["hours_since_user"]
    consistency = attractor.consistency(state_vector)
    novelty_base = 1.0
    out = []

    def push(name, source, timing_fit, source_salience, note="", source_content=""):
        spec = INTENTS[name]
        if spec["needs_source"] and source is None:
            return
        gate_dim, gate_min = RELATION_GATES.get(name, (None, 0.0))
        if gate_dim and relation[gate_dim] < gate_min:
            return
        novelty = 0.4 if name in recent_intent_types else novelty_base
        if source_content and recent_text and topic_novelty(source_content, recent_text) > 0.35:
            novelty *= 0.4   # 素材就是刚才聊过的事：主动消息会变复读，重罚
        disturb = (0.5 if ctx["quiet_hour"] else 0.0) + 0.2 * min(1.0, ctx["recent_active_count"] / 4.0)
        score = ((0.4 + 0.6 * consistency)
                 * (0.7 + 0.3 * relation["affinity"])
                 * timing_fit
                 * (0.5 + 0.5 * source_salience)
                 * novelty
                 - 0.15 * disturb
                 - BOUNDARY_RISK.get(name, 0.0))
        out.append(Candidate(name, spec["level"], max(0.0, min(1.0, score)), source,
                             {"consistency": consistency, "timing_fit": timing_fit,
                              "source_salience": source_salience, "novelty": novelty,
                              "disturb": round(disturb, 3), "note": note}))

    tw = timing_weight(hours)
    # 微反应：仅在用户刚说完话的应答路径上有意义
    if hours <= 1.0 / 6:
        push("micro_reaction", None, 1.0, 0.5)
    # 未完成意图驱动的候选
    for it in open_intents[:3]:
        if hours >= 0.5:
            push("remind_open_intent", {"type": "intent", "id": it["id"]},
                 0.6 + 0.4 * tw, it["salience"], source_content=it["content"])
            push("ask_light_question", {"type": "intent", "id": it["id"]},
                 0.5 + 0.5 * tw, it["salience"], source_content=it["content"])
        if hours >= 2:
            push("express_care", {"type": "intent", "id": it["id"]},
                 0.5 + 0.5 * tw, it["salience"], source_content=it["content"])
        if hours >= 4:
            push("propose_small_activity", {"type": "intent", "id": it["id"]},
                 0.4 + 0.6 * tw, it["salience"], source_content=it["content"])
    # 记忆驱动的候选
    for m in memories[:2]:
        if hours >= 0.5:
            push("share_related_memory", {"type": "memory", "id": m["id"]},
                 0.5 + 0.5 * tw, m["salience"], source_content=m["content"])
            push("ask_light_question", {"type": "memory", "id": m["id"]},
                 0.5 + 0.5 * tw, m["salience"], source_content=m["content"])
        if hours >= 6:
            push("shift_topic", {"type": "memory", "id": m["id"]},
                 0.3 + 0.7 * tw, m["salience"], source_content=m["content"])
    return out


def best_candidate(candidates: list) -> Candidate:
    return max(candidates, key=lambda c: c.score) if candidates else None
