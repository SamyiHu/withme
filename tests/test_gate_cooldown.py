"""P3 验收：闸门——免打扰、冷却、每日预算、权限、用户开关、来源校验全部生效。"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import make_tmpdir, rmtree, run_tests

from src.config import load_config, load_persona
from src.gate import Gate, register_active
from src.motivation import Candidate
from src.storage import get_setting, init_db, set_setting

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _at(hour: int, minute: int = 0, day_offset: int = 0) -> float:
    lt = time.localtime()
    return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday + day_offset,
                        hour, minute, 0, 0, 0, -1))


def _gate(tmp):
    conn = init_db(":memory:")
    cfg = load_config(os.path.join(ROOT, "config", "config.example.yaml"))
    persona = load_persona(os.path.join(ROOT, "config", "persona.example.yaml"))
    return Gate(cfg, persona, conn), conn


def _row(now: float) -> dict:
    return {"last_active_time_by_level": {}, "last_active_time_by_intent": {},
            "active_count_today": {"date": time.strftime("%Y-%m-%d", time.localtime(now)),
                                   "L1": 0, "L2": 0, "L3": 0},
            "cooldown_multiplier": {}}


def _cand(name="remind_open_intent", level=2, source=None):
    return Candidate(name, level, 0.8, source or {"type": "intent", "id": 1}, {})


def test_quiet_hours_block_l2_but_not_l1():
    tmp = make_tmpdir()
    try:
        gate, _ = _gate(tmp)
        night = _at(23, 30)
        d2 = gate.decide(_cand(), 0.9, _row(night), night, source_exists=lambda s: True)
        assert not d2.allowed and d2.reasons["quiet_hours"]["blocked"] is True
        micro = Candidate("micro_reaction", 1, 0.8, None, {})
        d1 = gate.decide(micro, 0.5, _row(night), night, source_exists=lambda s: True)
        assert d1.allowed, "免打扰时段只拦 L2/L3，L1 微反应放行"
        day = _at(14, 0)
        d2d = gate.decide(_cand(), 0.9, _row(day), day, source_exists=lambda s: True)
        assert d2d.allowed
    finally:
        rmtree(tmp)


def test_cooldown_blocks_then_releases():
    tmp = make_tmpdir()
    try:
        gate, _ = _gate(tmp)
        cd = gate.persona.cooldown_seconds["L2"]
        t0 = _at(14, 0)
        row = register_active(_row(t0), 2, "remind_open_intent", t0)
        early = gate.decide(_cand(), 0.9, row, t0 + cd - 10, source_exists=lambda s: True)
        assert not early.allowed and not early.reasons["cooldown_level"]["pass"]
        late = gate.decide(_cand(), 0.9, row, t0 + cd + 1, source_exists=lambda s: True)
        assert late.allowed, f"冷却 {cd}s 到期后应放行：{late.reasons}"
        # 同意图类型另有 0.5x 冷却（在分级冷却之内，这里不额外拦截）
    finally:
        rmtree(tmp)


def test_daily_budget_and_rollover():
    tmp = make_tmpdir()
    try:
        gate, _ = _gate(tmp)
        budget = gate.persona.daily_budget["L2"]
        cd = gate.persona.cooldown_seconds["L2"]
        t0 = _at(14, 0)
        row = _row(t0)
        for _ in range(budget):
            row = register_active(row, 2, "share_related_memory", t0)
        d = gate.decide(_cand(), 0.9, row, t0 + cd + 1, source_exists=lambda s: True)
        assert not d.allowed and not d.reasons["budget"]["pass"]
        assert d.reasons["budget"]["used"] == budget
        # 次日同一时刻预算自动重置（冷却也已过）
        next_day = _at(14, 0, day_offset=1)
        d2 = gate.decide(_cand(), 0.9, row, next_day, source_exists=lambda s: True)
        assert d2.allowed, f"跨日应重置预算：{d2.reasons}"
    finally:
        rmtree(tmp)


def test_permission_blocks_l3():
    tmp = make_tmpdir()
    try:
        gate, _ = _gate(tmp)
        t = _at(14, 0)
        l3 = Candidate("shift_topic", 3, 0.9, {"type": "memory", "id": 2}, {})
        d = gate.decide(l3, 0.95, _row(t), t, source_exists=lambda s: True)
        assert not d.allowed and not d.reasons["permission"], "persona 默认 L3=false 必须拦截"
    finally:
        rmtree(tmp)


def test_user_switch_blocks_everything():
    tmp = make_tmpdir()
    try:
        gate, conn = _gate(tmp)
        t = _at(14, 0)
        set_setting(conn, "proactive_enabled", False)
        d = gate.decide(_cand(), 0.9, _row(t), t, source_exists=lambda s: True)
        assert not d.allowed and d.reasons["switch"] is False
        set_setting(conn, "proactive_enabled", True)
        set_setting(conn, "proactive_disabled_levels", {"L2": True})
        d2 = gate.decide(_cand(), 0.9, _row(t), t, source_exists=lambda s: True)
        assert not d2.allowed and d2.reasons["switch"] is False, "分级关掉 L2 也必须拦截"
        d3 = gate.decide(Candidate("micro_reaction", 1, 0.8, None, {}), 0.5, _row(t), t)
        assert d3.allowed
        assert get_setting(conn, "proactive_enabled") is True
    finally:
        rmtree(tmp)


def test_source_must_exist_and_match_type():
    tmp = make_tmpdir()
    try:
        gate, _ = _gate(tmp)
        t = _at(14, 0)
        no_source = Candidate("remind_open_intent", 2, 0.8, None, {})
        d1 = gate.decide(no_source, 0.9, _row(t), t, source_exists=lambda s: True)
        assert not d1.allowed and not d1.reasons["source"], "L2+ 无来源必须拦截"
        wrong_type = Candidate("remind_open_intent", 2, 0.8, {"type": "memory", "id": 5}, {})
        d2 = gate.decide(wrong_type, 0.9, _row(t), t, source_exists=lambda s: True)
        assert not d2.allowed, "remind_open_intent 只接受 intent 来源"
        missing = _cand()  # source intent:1，但库里不存在
        d3 = gate.decide(missing, 0.9, _row(t), t, source_exists=lambda s: False)
        assert not d3.allowed and not d3.reasons["source"], "来源查无此条必须拦截（反幻觉）"
    finally:
        rmtree(tmp)


def test_rejection_multiplier_doubles_cooldown():
    tmp = make_tmpdir()
    try:
        gate, _ = _gate(tmp)
        cd = gate.persona.cooldown_seconds["L2"]
        t0 = _at(14, 0)
        row = _row(t0)
        row["cooldown_multiplier"] = {"remind_open_intent": 2.0}  # 拒绝后的倍率
        row = register_active(row, 2, "remind_open_intent", t0)
        at_cd = gate.decide(_cand(), 0.9, row, t0 + cd + 1, source_exists=lambda s: True)
        assert not at_cd.allowed, f"冷却加倍后 {cd}s 不应放行"
        at_2cd = gate.decide(_cand(), 0.9, row, t0 + 2 * cd + 1, source_exists=lambda s: True)
        assert at_2cd.allowed, f"{2 * cd}s（加倍冷却）后应放行：{at_2cd.reasons}"
        assert at_2cd.reasons["cooldown_level"]["multiplier"] == 2.0
    finally:
        rmtree(tmp)


if __name__ == "__main__":
    sys.exit(run_tests(globals(), "test_gate_cooldown"))
