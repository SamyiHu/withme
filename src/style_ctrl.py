"""风格曲线控制器（用户设计：按对话反馈实时修正风格，同时保持稳定）。

concise ∈ [0.15, 0.95]，基线 0.5：
- 负反馈（嫌话多/单字敷衍）抬升 -> 输出可机检 LEN 硬约束（走 state_style
  的校验/打回/机械兜底链，不是建议而是强制）
- 积极互动（对方回得比你上一条还长）放松
- 每条消息向基线回归 10%——修正实时，但不会漂走（稳定器）
- 存在 agent 行 JSON 里：每个用户塑造自己那份小满的话痨曲线
"""


def update(bias: dict, user_text: str, last_agent_reply_len: int) -> dict:
    """每条用户消息调用一次，返回新的 bias。"""
    c = float((bias or {}).get("concise", 0.55))
    t = (user_text or "").strip()
    c += (0.5 - c) * 0.10                       # 回归基线（稳定器）
    if any(w in t for w in ("话多", "啰嗦", "唠叨", "废话", "太长", "简短",
                            "说重点", "闭嘴", "少说", "打住", "长篇大论", "重点")):
        c += 0.20                               # 明确嫌长
    elif t in ("嗯", "哦", "噢", "呵", "额", "？", "?", "...", "。。。", "6"):
        c += 0.12                               # 单字敷衍 = 隐性"你话太多"
    elif last_agent_reply_len and len(t) >= max(12, last_agent_reply_len * 1.2):
        c -= 0.10                               # 对方回得比你长且投入 = 放松
    c = max(0.15, min(0.95, c))
    return {"concise": round(c, 3)}


def constraints(concise: float) -> list:
    """concise -> 可机检硬约束（state_style.check 逐条校验，违规打回重写）。
    基线 0.55 起就有 LEN<=36 的防唠叨顶（真人 P90 也就 30 字），反馈越负越紧。"""
    c = float(concise or 0.5)
    # 档位标定自真人数据（主人 252 条实测：中位 7 字、P90 30 字）
    if c >= 0.80:
        return ["LEN<=16", "EXCL<=0"]
    if c >= 0.65:
        return ["LEN<=24"]
    if c >= 0.55:
        return ["LEN<=36"]
    return []
