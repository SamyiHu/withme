"""主动闸门层（原则 D）：分级 L0-L3、阈值、冷却、每日预算、免打扰、
场景白名单、权限与用户开关。所有判定都记录进 gate_result 供审计。
"""
import time
from dataclasses import dataclass, field

from .config import AppConfig, Persona, parse_quiet_hours
from .intents import INTENTS
from .storage import get_setting


@dataclass
class Decision:
    allowed: bool
    level: int
    intent: str
    reasons: dict = field(default_factory=dict)
    fallback: str = "silence"   # 不通过时退回 silence（闸门不过就不说话）

    def to_log(self):
        return {"allowed": self.allowed, "level": self.level, "intent": self.intent,
                "fallback": self.fallback, "checks": self.reasons}


class Gate:

    def __init__(self, config: AppConfig, persona: Persona, conn=None):
        self.config = config
        self.persona = persona
        self.conn = conn

    def decide(self, candidate, active_score: float, state_row: dict, now: float,
               source_exists=None) -> Decision:
        """source_exists: callable(source) -> bool，校验来源仍可溯源。"""
        level = candidate.level
        level_name = f"L{level}"
        reasons = {}

        # 1. 用户总开关与分级开关
        enabled = get_setting(self.conn, "proactive_enabled", True) if self.conn else True
        disabled_levels = get_setting(self.conn, "proactive_disabled_levels", {}) if self.conn else {}
        reasons["switch"] = enabled and not disabled_levels.get(level_name, False)
        if not reasons["switch"]:
            return Decision(False, level, candidate.name, reasons)

        # 2. 角色卡权限
        reasons["permission"] = bool(self.persona.active_permissions.get(level_name, False))
        if not reasons["permission"]:
            return Decision(False, level, candidate.name, reasons)

        # 3. 阈值：active_score 必须达到该级别门槛（L1 微主动门槛最低）
        threshold = self.config.gate.thresholds.get(level_name, 1.0)
        # 反思层可对 L2 阈值做有界漂移（settings.gate_shift_L2，±0.15 封顶）
        if level_name == "L2" and self.conn is not None:
            threshold = max(0.0, threshold + float(
                get_setting(self.conn, "gate_shift_L2", 0.0) or 0.0))
        reasons["threshold"] = {"score": round(active_score, 4), "need": threshold,
                                "pass": active_score >= threshold}
        if not reasons["threshold"]["pass"]:
            return Decision(False, level, candidate.name, reasons)

        # 4. 免打扰时段：L2/L3 一律拦截，L1 放行
        lt = time.localtime(now)
        quiet = parse_quiet_hours(self.persona.quiet_hours, (lt.tm_hour, lt.tm_min))
        reasons["quiet_hours"] = {"quiet": quiet, "blocked": quiet and level >= 2}
        if reasons["quiet_hours"]["blocked"]:
            return Decision(False, level, candidate.name, reasons, fallback="silence")

        # 5. 场景白名单（MVP 只有 chat/idle 两个场景，恒在白名单内）
        scene = get_setting(self.conn, "scene", "idle") if self.conn else "idle"
        reasons["scene"] = scene in self.config.gate.scene_whitelist
        if not reasons["scene"]:
            return Decision(False, level, candidate.name, reasons)

        # 6. 来源校验：L2+ 必须挂来源，且来源可溯源
        if level >= 2:
            spec = INTENTS[candidate.name]
            reasons["source"] = bool(spec["needs_source"]) is False or (
                candidate.source is not None
                and candidate.source.get("type") in spec["source_types"]
                and (source_exists is None or source_exists(candidate.source)))
            if not reasons["source"]:
                return Decision(False, level, candidate.name, reasons)

        # 7. 冷却：分级冷却 ×（意图倍率与级别倍率取大者——拒绝会同时抬高两者）
        base_cd = self.persona.cooldown_seconds.get(level_name, 10 ** 9)
        stored = state_row.get("cooldown_multiplier", {})
        multiplier = max(float(stored.get(candidate.name, 1.0)),
                         float(stored.get(level_name, 1.0)))
        elapsed_level = now - state_row.get("last_active_time_by_level", {}).get(level_name, 0.0)
        reasons["cooldown_level"] = {"elapsed": round(elapsed_level, 1),
                                     "need": base_cd * multiplier,
                                     "multiplier": multiplier,
                                     "pass": elapsed_level >= base_cd * multiplier}
        if not reasons["cooldown_level"]["pass"]:
            return Decision(False, level, candidate.name, reasons,
                            fallback="micro" if level == 1 else "silence")
        intent_cd = base_cd * 0.5
        elapsed_intent = now - state_row.get("last_active_time_by_intent", {}).get(candidate.name, 0.0)
        reasons["cooldown_intent"] = {"elapsed": round(elapsed_intent, 1), "need": intent_cd,
                                      "pass": elapsed_intent >= intent_cd}
        if not reasons["cooldown_intent"]["pass"]:
            return Decision(False, level, candidate.name, reasons)

        # 8. 每日预算（跨日自动清零）
        counts = register_budget(state_row, now)
        budget = self.persona.daily_budget.get(level_name, 0)
        reasons["budget"] = {"used": counts.get(level_name, 0), "budget": budget,
                             "pass": counts.get(level_name, 0) < budget}
        if not reasons["budget"]["pass"]:
            return Decision(False, level, candidate.name, reasons)

        return Decision(True, level, candidate.name, reasons)


def register_budget(state_row: dict, now: float) -> dict:
    """按本地日期滚动清零当日计数（纯函数，不写库）。"""
    counts = dict(state_row.get("active_count_today") or {})
    today = time.strftime("%Y-%m-%d", time.localtime(now))
    if counts.get("date") != today:
        counts = {"date": today, "L1": 0, "L2": 0, "L3": 0}
    return counts


def register_active(state_row: dict, level: int, intent_type: str, now: float) -> dict:
    """主动行为真正发出后更新状态行（冷却锚点与当日计数）。"""
    row = dict(state_row)
    row["last_active_time_by_level"] = dict(state_row.get("last_active_time_by_level") or {})
    row["last_active_time_by_level"][f"L{level}"] = now
    row["last_active_time_by_intent"] = dict(state_row.get("last_active_time_by_intent") or {})
    row["last_active_time_by_intent"][intent_type] = now
    counts = register_budget(state_row, now)
    counts[f"L{level}"] = counts.get(f"L{level}", 0) + 1
    row["active_count_today"] = counts
    return row
