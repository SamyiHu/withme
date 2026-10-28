"""P3 验收：动机评分与端到端主动流——纯随机不得发起 L2+；
每个主动意图必须挂来源；评分与闸门明细写入 active_logs。"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import make_tmpdir, rmtree, run_tests

from src import motivation
from src.main import Agent

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _at(hour: int, minute: int = 0, day_offset: int = 0) -> float:
    lt = time.localtime()
    return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday + day_offset, hour, minute, 0, 0, 0, -1))


def _agent(tmp) -> Agent:
    return Agent(config_dir=os.path.join(ROOT, "config"),
                 db_path=os.path.join(tmp, "agent.db"))


def test_topic_novelty_penalizes_echo():
    """主动候选素材若是刚聊过的事，话题新颖度罚分——防主动消息变复读。"""
    from src.motivation import topic_novelty, generate_candidates
    from src.persona_attractor import PersonaAttractor

    echo = topic_novelty("周末去学冲浪，先练平衡", "你之前说周末去学冲浪现在怎么样了")
    fresh = topic_novelty("周末去学冲浪，先练平衡", "今天午饭吃了什么，食堂人好多")
    assert echo > 0.35 > fresh, (echo, fresh)

    sv = {"energy": 0.8, "social_need": 0.8, "curiosity": 0.8, "arousal": 0.5,
          "valence": 0.3, "care": 0.5, "expression": 0.5}
    relation = {"affinity": 0.9, "trust": 0.9, "familiarity": 0.9}
    mem = {"id": 1, "type": "event", "content": "周末去学冲浪，先练平衡", "salience": 0.8}
    class _StubAttractor:
        def consistency(self, sv):
            return 0.5
    attractor = _StubAttractor()
    ctx = {"hours_since_user": 6.0, "quiet_hour": False, "recent_active_count": 0}
    common = dict(state_vector=sv, relation=relation, open_intents=[],
                  attractor=attractor, ctx=ctx, recent_intent_types=())
    echo_cands = generate_candidates(memories=[mem], recent_text="周末去学冲浪怎么样了", **common)
    fresh_cands = generate_candidates(memories=[mem], recent_text="食堂话题", **common)
    def _score(name, cands):
        return max((c.score for c in cands if c.name == name), default=0)
    assert _score("share_related_memory", echo_cands) <            _score("share_related_memory", fresh_cands), "回声素材必须被罚分"


def test_active_score_drivers():
    weights = {"social": 0.35, "curiosity": 0.15, "care": 0.25, "expression": 0.15, "disturb": 0.10}
    sv = {"energy": 0.6, "social_need": 0.85, "curiosity": 0.7, "care": 0.7,
          "expression": 0.55, "arousal": 0.4, "valence": 0.35}
    rel = {"affinity": 0.5, "trust": 0.5, "familiarity": 0.3}
    # 用户离开 8 小时 + 有未完成意图：分高
    away = {"hours_since_user": 8.0, "information_gap": 0.6, "open_intent_salience": 0.65,
            "topic_relevance": 0.5, "quiet_hour": False, "recent_active_count": 0}
    score_away, detail = motivation.active_score(sv, rel, away, weights)
    # 刚聊完 1 分钟 + 深夜：分低
    fresh = {"hours_since_user": 0.01, "information_gap": 0.3, "open_intent_salience": 0.0,
             "topic_relevance": 0.5, "quiet_hour": True, "recent_active_count": 3}
    score_fresh, _ = motivation.active_score(sv, rel, fresh, weights)
    assert score_away > 0.32, f"离开8h场景应过 L2 阈值：{score_away}"
    assert score_fresh < score_away
    assert 0.0 <= score_fresh <= 1.0
    assert set(detail) >= {"social_urge", "curiosity_urge", "care_urge", "expression_urge",
                           "disturb_cost"}


def test_no_memory_no_l2_candidates():
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp)
        noon = _at(12)
        result = agent.on_user_message("在吗", now=noon - 3600)  # 不含意图标记的短消息
        # 只产生低显著度/短消息记忆，不足以支撑 L2+：主动必须无来源
        r = agent.evaluate_proactive(now=noon)
        assert r.get("sent") is None
        logs = agent.conn.execute("SELECT COUNT(*) c FROM active_logs").fetchone()["c"]
        assert logs == 0, "无记忆/无意图时不得发起任何主动（原则 E）"
    finally:
        rmtree(tmp)


def test_proactive_reminder_end_to_end():
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp)
        t0 = _at(10)
        r1 = agent.on_user_message("最近在忙项目，不过还想学冲浪来着", now=t0)
        assert r1["reply"] and not r1["addon"], "刚说完话的应答路径不应搭主动"
        # 8 小时后（12+8=18 点，不在免打扰时段）
        r2 = agent.evaluate_proactive(now=t0 + 8 * 3600)
        assert r2.get("sent"), f"应当发起主动：{r2}"
        sent = r2["sent"]
        assert sent["level"] == 2 and sent["intent"] == "remind_open_intent"
        assert sent["text"]
        # 审计：评分明细 + 闸门明细 + 来源
        row = agent.conn.execute("SELECT * FROM active_logs WHERE id=?",
                                 (sent["log_id"],)).fetchone()
        detail = json.loads(row["score_detail"])
        assert "active_score" in detail and "urges" in detail
        assert row["source_type"] == "intent" and row["source_id"] is not None
        gate = json.loads(row["gate_result"])
        assert gate["allowed"] is True
        # 立即再 tick：分级冷却拦截
        r3 = agent.evaluate_proactive(now=t0 + 8 * 3600 + 60)
        assert r3.get("sent") is None
        assert "cooldown" in str(r3.get("gate", {}).get("checks", {}))
    finally:
        rmtree(tmp)


def test_reply_addon_after_long_absence():
    tmp = make_tmpdir()
    try:
        agent = _agent(tmp)
        t0 = _at(9)
        agent.on_user_message("我打算周末去学冲浪", now=t0)
        # 离开 8 小时后回来：应答 + 主动搭话（remind 作为附加层）
        r = agent.on_user_message("我回来啦", now=t0 + 8 * 3600)
        assert r["reply"]
        if r["addon"]:
            assert r["addon_meta"]["level"] >= 2
            logs = agent.conn.execute(
                "SELECT COUNT(*) c FROM active_logs WHERE intent_type=?",
                (r["addon_meta"]["intent"],)).fetchone()["c"]
            assert logs >= 1, "主动附加层必须落 active_logs"
        # 状态已被保存（last_tick 推进）
        row = agent.get_row(t0 + 8 * 3600)
        assert row["last_tick"] == t0 + 8 * 3600
    finally:
        rmtree(tmp)


def test_seed_reproducible_reply():
    tmp1, tmp2 = make_tmpdir(), make_tmpdir()
    try:
        a1, a2 = _agent(tmp1), _agent(tmp2)
        # 两个独立 agent（不同库）在相同时间与相同输入下渲染一致
        t = _at(15, 30)
        r1 = a1.on_user_message("今天天气不错", now=t)
        r2 = a2.on_user_message("今天天气不错", now=t)
        assert r1["reply"] == r2["reply"], "同 seed 渲染必须可复现"
    finally:
        rmtree(tmp1)
        rmtree(tmp2)


if __name__ == "__main__":
    sys.exit(run_tests(globals(), "test_proactive_flow"))
