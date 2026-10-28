"""路线图批次验收（离线）：时间感、打字勿扰、反思调闸门、多步手部、成本护栏、
初识锚点、judge 抽样。"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import make_tmpdir, rmtree, run_tests

from src import motivation
from src.config import load_config
from src.llm_client import MockLLMClient
from src.main import Agent
from src.prompt_builder import PromptBuilder, relative_age
from src.storage import get_setting, set_setting

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOW = 1790000000.0


def _agent(tmp, client=None, **cfg_over) -> Agent:
    agent = Agent(config_dir=os.path.join(ROOT, "config"),
                  db_path=os.path.join(tmp, "agent.db"), llm_client=client)
    for section, key, val in cfg_over.values() if cfg_over else ():
        pass
    for path, val in cfg_over.items():
        section, key = path.split(".")
        setattr(getattr(agent.config, section), key, val)
    return agent


def _at(hour):
    lt = time.localtime()
    return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, hour, 0, 0, 0, 0, -1))


def test_relative_age_labels():
    assert relative_age(NOW - 3600, NOW) == "今天"
    assert relative_age(NOW - 1.5 * 86400, NOW) == "昨天"
    assert relative_age(NOW - 5 * 86400, NOW) == "5天前"
    assert relative_age(NOW - 10 * 86400, NOW) == "上周"
    assert relative_age(NOW - 40 * 86400, NOW) == "上个月"
    assert relative_age(None, NOW) == ""


def test_memory_time_annotation_in_prompt():
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp)
        builder = agent.builder
        mem_old = {"id": 1, "type": "event", "content": "老朋友的事", "salience": 0.6,
                   "created_at": NOW - 10 * 86400}
        mem_new = {"id": 2, "type": "event", "content": "昨天的事", "salience": 0.6,
                   "created_at": NOW - 100}
        row = agent.engine.new_row(NOW)
        prompt = builder.build(intent="respond", source=None,
                               state_vector=row["state_vector"],
                               relation_state=row["relation_state"],
                               memories=[mem_old, mem_new], open_intents=[],
                               permission_level="L2", last_active_effect="x", now=NOW)
        memories = prompt["相关记忆"]
        assert memories[0].get("时间") == "上周"
        assert memories[1].get("时间") == "今天"
        assert "记忆条目会标注时间" in str(prompt["表达风格"])  # 时间感规则随人设下发
    finally:
        rmtree(tmp)


def test_first_chat_anchor_and_known_days():
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp)
        agent.on_user_message("第一次见面，你好呀", now=NOW)
        assert get_setting(agent.conn, "first_chat_ts") == NOW
        s = agent.status(now=NOW + 7200)
        assert s["known_days"] == 0
        assert agent.status(now=NOW + 3 * 86400)["known_days"] == 3
    finally:
        rmtree(tmp)


def test_typing_raises_disturb():
    weights = {"social": 0.35, "curiosity": 0.15, "care": 0.25,
               "expression": 0.15, "disturb": 0.10}
    sv = {"energy": 0.6, "social_need": 0.8, "curiosity": 0.7, "care": 0.7,
          "expression": 0.55, "arousal": 0.4, "valence": 0.35}
    rel = {"affinity": 0.6, "trust": 0.6, "familiarity": 0.4}
    base = {"hours_since_user": 3.0, "information_gap": 0.6,
            "open_intent_salience": 0.7, "topic_relevance": 0.5,
            "quiet_hour": False, "recent_active_count": 0}
    idle, _ = motivation.active_score(sv, rel, base, weights)
    typing, _ = motivation.active_score(sv, rel, {**base, "user_typing": True}, weights)
    assert idle > typing, "用户正在打字时主动评分必须更低"


def test_typing_flag_flows_from_settings():
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp)
        set_setting(agent.conn, "last_typing_ts", 1000.0)
        row = agent.engine.new_row(1030.0)
        ctx = agent._ctx(row, 1030.0)
        assert ctx["user_typing"] is True
        ctx2 = agent._ctx(row, 1200.0)
        assert ctx2["user_typing"] is False, "90 秒外不算打字中"
    finally:
        rmtree(tmp)


def test_gate_shift_from_reflection():
    tmp = make_tmpdir()
    try:
        from tests.test_reflection import FakeReflectClient
        out = json.dumps({"insights": [], "new_rule": "",
                          "trait_shift": {}, "gate_shift": {"L2_threshold": 0.05}})
        agent = _agent(tmp, FakeReflectClient([out]))
        agent.maybe_reflect(force=True, now=NOW)
        assert abs(get_setting(agent.conn, "gate_shift_L2") - 0.05) < 1e-9
        # 闸门生效：0.36 分在默认阈值 0.35 之上、偏移后 0.40 之下 -> 被拦
        from src.motivation import Candidate
        cand = Candidate("remind_open_intent", 2, 0.8, {"type": "intent", "id": 1}, {})
        d = agent.gate.decide(cand, 0.36, agent.engine.new_row(_at(14)), _at(14),
                              source_exists=lambda s: True)
        assert not d.allowed and abs(d.reasons["threshold"]["need"] - 0.40) < 1e-6
        # reset 一并清理
        agent.reset_growth()
        assert get_setting(agent.conn, "gate_shift_L2") in (0.0, None)
    finally:
        rmtree(tmp)


def test_multi_step_hand_action():
    tmp = make_tmpdir()
    try:
        from tests.test_perception import FakeAnalysisClient
        raw = json.dumps({"sentiment": 0.1, "emotion": "平静", "intent_type": "闲聊",
                          "topics": ["提醒"], "urgency": 0.4, "cancel_topics": [],
                          "defer_topics": [], "done_topics": [],
                          "hand_action": [
                              {"name": "reminder.set",
                               "params": {"text": "喝水", "when": "30分钟后"}},
                              {"name": "note.add", "params": {"text": "今晚早点睡"}}]})
        agent = _agent(tmp, FakeAnalysisClient(raw))
        r = agent.on_user_message("半小时后提醒我喝水，顺便记一下今晚早点睡", now=NOW)
        hand = r["meta"]["hand"]
        assert hand["ok"] and hand["action"] == "reminder.set+note.add"
        assert "喝水" in hand["summary"] and "早点睡" in hand["summary"]
        assert agent.conn.execute("SELECT COUNT(*) c FROM reminders"
                                  ).fetchone()["c"] == 1
        assert agent.conn.execute("SELECT COUNT(*) c FROM hands_log"
                                  ).fetchone()["c"] >= 2
    finally:
        rmtree(tmp)


def test_cost_degrade_guards():
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp)
        agent.config.cost.daily_token_budget = 10  # 极小预算
        agent.client.last_usage = {"prompt_tokens": 100, "completion_tokens": 5}
        agent._record_usage("respond")
        assert agent._cost_degraded() is True
        # 降级：主动被拦、反思跳过
        assert agent.evaluate_proactive(now=_at(14)).get("reason") == "cost budget exceeded"
        assert agent.maybe_reflect(now=_at(15)) is None
        # 核心聊天不受影响
        r = agent.on_user_message("还在吗", now=_at(15) + 60)
        assert r["reply"]
    finally:
        rmtree(tmp)


def test_judge_fetch_pairs():
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from judge_eval import fetch_pairs
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp)
        agent._log_message("user", "第一句", {}, 1.0)
        agent._log_message("agent", "回一", {}, 2.0)
        agent._log_message("user", "第二句", {}, 3.0)
        agent._log_message("agent", "", {}, 4.0)  # 空回复应被跳过
        agent._log_message("agent", "回二", {}, 5.0)
        pairs = fetch_pairs(os.path.join(tmp, "agent.db"), 10)
        assert pairs == [("第一句", "回一"), ("第二句", "回二")]
    finally:
        rmtree(tmp)


if __name__ == "__main__":
    sys.exit(run_tests(globals(), "test_roadmap"))
