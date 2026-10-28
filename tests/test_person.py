"""人物卡 + 信任引擎 + 记忆分人验收。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import run_tests

from src.config import load_persona
from src.memory import MemoryStore
from src.person import PersonStore
from src.storage import init_db

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _stack():
    conn = init_db(":memory:")
    persona = load_persona(os.path.join(ROOT, "config", "persona.example.yaml"))
    # example 模板的 creator 留空（开源导出安全）；本文件要测创造者卡，注入演示身份
    persona.creator = dict(persona.creator or {}, name="阿澄", wxid="creator_demo")
    return conn, PersonStore(conn, persona.role_id, persona), MemoryStore(conn)


def test_creator_card_defaults_and_trust():
    conn, persons, _ = _stack()
    try:
        pid = persons.ensure_creator(now=1000.0)
        card = persons.get(pid)
        assert card["kind"] == "creator"
        assert card["authority"] == "creator"
        assert card["trust_identity"] == 1.0 and card["attachment"] == 1.0
        # 创造者的批评不降低信任、不抬高戒备
        before = persons.summary(pid)
        persons.observe(pid, "你这句说得不对", from_owner=True, now=1001.0)
        after = persons.summary(pid)
        assert after["信任"] == before["信任"] and after["戒备"] == before["戒备"]
    finally:
        conn.close()


def test_member_trust_moves_with_events():
    conn, persons, _ = _stack()
    try:
        pid = persons.ensure("wxid_a", "胖达", "group_member", now=1000.0)
        start = persons.summary(pid)
        persons.observe(pid, "谢谢，帮我看看这个", from_owner=False, now=1001.0)
        helped = persons.summary(pid)
        assert helped["信任"] > start["信任"]
        persons.observe(pid, "你就是个傻逼，滚", from_owner=False, now=1002.0)
        insulted = persons.summary(pid)
        assert insulted["信任"] < helped["信任"] and insulted["戒备"] > helped["戒备"]
        # 事件可回查
        rows = conn.execute("SELECT event_type FROM person_events WHERE person_id=?",
                            (pid,)).fetchall()
        assert {r["event_type"] for r in rows} >= {"help", "insult"}
    finally:
        conn.close()


def test_memory_scoped_by_person_and_scope():
    conn, persons, mem = _stack()
    try:
        a = persons.ensure("wxid_a", "甲", "group_member", now=1000.0)
        b = persons.ensure("wxid_b", "乙", "group_member", now=1000.0)
        ma = mem.add("xiaoman", "event", "我喜欢吃火锅", person_id=a, scope="group:g1",
                     now=1001.0)
        mb = mem.add("xiaoman", "event", "我喜欢吃火锅", person_id=b, scope="group:g1",
                     now=1001.0)
        assert ma != mb, "不同人说同样的话不能被去重合并"
        hits_a = mem.retrieve("xiaoman", "火锅", 5, 0.1, now=1002.0,
                              person_id=a, scope="group:g1")
        assert hits_a and all(h["person_id"] == a for h in hits_a)
        assert all(h["person_id"] != b for h in hits_a)
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(run_tests(globals(), name="test_person"))
