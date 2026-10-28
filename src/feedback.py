"""反馈闭环层：结算用户对上一次主动行为的回应。

记录：回复延迟、情绪变化、是否拒绝；并据此更新
关系状态、同类型主动的冷却倍率、以及"最近主动效果"（进入下一轮 Prompt）。
"""
import json

from . import relation as relation_mod

POSITIVE_WORDS = ["谢谢", "好的", "好呀", "嗯嗯", "喜欢", "开心", "哈哈", "有意思",
                  "可以呀", "行吧", "好啊", "对"]
REJECT_WORDS = ["别烦", "别提", "别问", "不用了", "不需要", "闭嘴", "烦", "滚", "取消",
                "拒绝", "算了", "不想", "别说了"]
MAX_MULTIPLIER = 4.0
IGNORE_LATENCY_SECONDS = 6 * 3600  # 超过 6 小时未回应视为"被忽略"


def analyze_user_reply(text: str) -> dict:
    pos = sum(1 for w in POSITIVE_WORDS if w in text)
    neg = sum(1 for w in REJECT_WORDS if w in text)
    rejected = any(w in text for w in REJECT_WORDS)
    if neg > pos:
        sentiment = -min(1.0, 0.5 + 0.5 * (neg - pos) / neg)
    elif pos > neg:
        sentiment = min(1.0, 0.5 + 0.5 * (pos - neg) / pos)
    else:
        sentiment = 0.0
    return {"sentiment": sentiment, "rejected": rejected,
            "is_question": any(mark in text for mark in ("?", "？", "吗", "呢"))}


def analyze_with_perception(text: str, perc: dict = None) -> dict:
    """词表分析 +（可选）洞察层结果融合：感知优先，词表兜底。"""
    base = analyze_user_reply(text)
    if not perc:
        return base
    sentiment = perc.get("sentiment", 0.0)
    if perc.get("intent_type") == "拒绝" and sentiment > -0.3:
        sentiment = -0.6
    return {
        "sentiment": sentiment,
        "rejected": base["rejected"] or perc.get("intent_type") == "拒绝",
        "is_question": base["is_question"] or perc.get("intent_type") == "提问",
    }


def settle_pending(agent, row: dict, text: str, now: float, perception: dict = None):
    """找到最近一条未结算的主动日志，按用户回应结算。无待结算返回 None。"""
    pending = agent.conn.execute(
        "SELECT * FROM active_logs WHERE role_id=? AND user_response_latency IS NULL"
        " AND level>=1 AND created_at>? ORDER BY id DESC LIMIT 1",
        (agent.role_id, now - 48 * 3600)).fetchone()
    if pending is None:
        return None

    latency = now - pending["created_at"]
    analysis = analyze_with_perception(text, perception)
    agent.conn.execute(
        "UPDATE active_logs SET user_response_latency=?, user_sentiment_delta=? WHERE id=?",
        (latency, analysis["sentiment"], pending["id"]))
    agent.conn.commit()

    intent = pending["intent_type"]
    level_key = f"L{pending['level']}"
    multipliers = dict(row.get("cooldown_multiplier") or {})
    # 拒绝/忽略把"此刻"锚定为各级别的冷却起点：短期内不得换个意图、
    # 也不得降级成微反应再来（刚被拒绝就连 L1 微反应也是打扰）
    row["last_active_time_by_level"] = dict(row.get("last_active_time_by_level") or {})
    for lvl in ("L1", "L2", "L3"):
        row["last_active_time_by_level"][lvl] = now
    if analysis["rejected"]:
        # 明确拒绝：同类型冷却加倍，该级别整体也加倍（有上限），并留痕到下一轮 Prompt
        multipliers[intent] = min(MAX_MULTIPLIER, max(2.0, multipliers.get(intent, 1.0) * 2.0))
        multipliers[level_key] = min(MAX_MULTIPLIER, max(2.0, multipliers.get(level_key, 1.0) * 2.0))
        row = agent.engine.apply_event_impulse(row, "user_negative_reply")
        effect = (f"上次主动（{intent}）被明确拒绝；该类与该级冷却"
                  f"×{multipliers[level_key]:.1f}")
    elif latency > IGNORE_LATENCY_SECONDS:
        multipliers[intent] = min(MAX_MULTIPLIER, multipliers.get(intent, 1.0) * 1.5)
        multipliers[level_key] = min(MAX_MULTIPLIER, multipliers.get(level_key, 1.0) * 1.5)
        row = agent.engine.apply_event_impulse(row, "user_no_reply")
        effect = f"上次主动（{intent}）长时间未获回应；该类冷却×{multipliers[intent]:.1f}"
    else:
        if analysis["sentiment"] > 0.15:
            row = agent.engine.apply_event_impulse(row, "user_positive_reply")
        effect = (f"上次主动（{intent}）在 {latency:.0f}s 后得到"
                  f"{'正面' if analysis['sentiment'] > 0.15 else '中性'}回应")

    row["cooldown_multiplier"] = multipliers
    row["last_active_effect"] = effect
    return {"row": row, "summary": {"log_id": pending["id"], "intent": intent,
                                    "latency": round(latency, 1),
                                    "sentiment": analysis["sentiment"],
                                    "rejected": analysis["rejected"], "effect": effect}}
