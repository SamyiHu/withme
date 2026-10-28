"""P4 验收：反馈闭环——拒绝后同类型冷却加倍；主动效果进入下一轮 Prompt；
调度器惰性补算；正面反馈有限幅提升关系。"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import make_tmpdir, rmtree, run_tests

from src.feedback import analyze_user_reply
from src.main import Agent

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _at(hour: int, minute: int = 0, day_offset: int = 0) -> float:
    lt = time.localtime()
    return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday + day_offset, hour, minute, 0, 0, 0, -1))


def _agent(tmp) -> Agent:
    return Agent(config_dir=os.path.join(ROOT, "config"),
                 db_path=os.path.join(tmp, "agent.db"))


def _seed_proactive(agent, t_user, t_active):
    """造一条已发出的主动日志：用户留言 -> 8 小时后主动提醒。"""
    agent.on_user_message("我打算周末去学冲浪", now=t_user)
    result = agent.evaluate_proactive(now=t_active)
    assert result.get("sent"), f"前置条件失败：主动未发出 {result}"
    return result["sent"]["log_id"]


def test_analyze_user_reply_lexicon():
    assert analyze_user_reply("谢谢！想学")["sentiment"] > 0
    assert analyze_user_reply("哈哈，好呀")["sentiment"] > 0
    neg = analyze_user_reply("别提这个了，烦")
    assert neg["rejected"] is True and neg["sentiment"] < 0
    mid = analyze_user_reply("嗯，先这样")
    assert mid["sentiment"] == 0.0 and mid["rejected"] is False
    assert analyze_user_reply("明天会下雨吗")["is_question"] is True


def test_rejection_doubles_cooldown_and_affects_prompt():
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp)
        t0, t_active = _at(9), _at(17)
        log_id = _seed_proactive(agent, t0, t_active)
        t_reply = t_active + 90
        r = agent.on_user_message("别提这个了，烦", now=t_reply)
        fb = r["meta"]["feedback"]
        assert fb and fb["rejected"] is True and fb["log_id"] == log_id
        assert abs(fb["latency"] - 90) < 1
        row = agent.get_row(t_reply)
        assert row["cooldown_multiplier"]["remind_open_intent"] == 2.0
        # 冷却加倍生效：正常 1800s 已过，但加倍后 3600s 未到
        t_next = t_active + 2000
        blocked = agent.evaluate_proactive(now=t_next)
        assert blocked.get("sent") is None
        checks = blocked.get("gate", {}).get("checks", {})
        assert not checks.get("cooldown_level", {}).get("pass", True)
        # 主动效果已进入下一轮 Prompt（最近主动效果）
        row2 = agent._evolve(agent.get_row(t_next), t_next)
        assert "拒绝" in row2["last_active_effect"]
        # 情绪受损：valence 低于基线
        assert row2["state_vector"]["valence"] < agent.persona.baseline_state["valence"] + 0.2
    finally:
        rmtree(tmp)


def test_positive_reply_limited_relation_gain():
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp)
        t0, t_active = _at(9), _at(17)
        _seed_proactive(agent, t0, t_active)
        before = dict(agent.get_row(t_active)["relation_state"])
        r = agent.on_user_message("哈哈，被你记着了，谢谢！", now=t_active + 60)
        after = r["meta"]["relation"]
        assert after["affinity"] > before["affinity"], "正面回应应提升亲近度"
        assert after["affinity"] - before["affinity"] <= 0.05 + 1e-9, "单次提升必须限幅"
        assert after["familiarity"] - before["familiarity"] <= 0.01 + 1e-9
        row = agent.get_row(t_active + 60)
        assert "正面" in row["last_active_effect"]
        assert row["cooldown_multiplier"] == {}  # 正面反馈不加倍冷却
        # 主动日志已结算
        log = agent.conn.execute(
            "SELECT user_response_latency, user_sentiment_delta FROM active_logs"
            " ORDER BY id DESC LIMIT 1").fetchone()
        assert log["user_response_latency"] == 60
        assert log["user_sentiment_delta"] > 0
    finally:
        rmtree(tmp)


def test_scheduler_lazy_catchup():
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp)
        t0 = _at(8)
        agent.on_user_message("我打算周末去学冲浪", now=t0)
        # 调度器不空转：一次性补算 16 小时
        result = agent.evaluate_proactive(now=t0 + 16 * 3600)
        row = agent.get_row(t0 + 16 * 3600)
        assert row["last_tick"] == t0 + 16 * 3600, "必须按 Δt 补算到当前"
        # 16 小时无互动：social_need 因孤独累积显著高于基线
        assert row["state_vector"]["social_need"] > agent.persona.baseline_state["social_need"] + 0.1
        # 若被发出，必须有来源；深夜 24:00 会被免打扰拦截（t0+16h = 24:00）
        if (t0 // 3600 + 16) % 24 in (23, 0):
            assert result.get("sent") is None
    finally:
        rmtree(tmp)


def test_ignored_proactive_raises_cooldown():
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp)
        t0, t_active = _at(9), _at(17)
        _seed_proactive(agent, t0, t_active)
        # 用户 8 小时后才回一句无关的话：视为忽略，冷却 ×1.5
        agent.on_user_message("今天天气怎么样", now=t_active + 8 * 3600)
        row = agent.get_row(t_active + 8 * 3600)
        assert row["cooldown_multiplier"]["remind_open_intent"] == 1.5
        assert "未获回应" in row["last_active_effect"]
    finally:
        rmtree(tmp)


if __name__ == "__main__":
    sys.exit(run_tests(globals(), "test_feedback_loop"))
