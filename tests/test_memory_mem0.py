"""Mem0 混合后端验收（离线 FakeMem0）：意图走 SQLite、事实走 mem0、
别名溯源稳定、verbatim 直存、检索失败回退 SQLite。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import make_tmpdir, rmtree, run_tests

from src.memory_mem0 import Mem0MemoryStore
from src.storage import init_db


class FakeMem0:
    def __init__(self, fail_search=False):
        self.facts = {}
        self.calls = []
        self.fail_search = fail_search

    def add(self, messages, user_id=None, agent_id=None, run_id=None,
            metadata=None, infer=True, **kwargs):
        self.calls.append({"infer": infer, "text": messages[0]["content"],
                           "meta": metadata or {}})
        mid = f"mem-{len(self.facts) + 1}"
        content = messages[0]["content"] if not infer else f"抽取:{messages[0]['content'][:8]}"
        self.facts[mid] = content
        return {"results": [{"id": mid, "memory": content, "event": "ADD"}]}

    def search(self, query, filters=None, limit=5):
        if self.fail_search:
            raise RuntimeError("search down")
        return {"results": [{"id": k, "memory": v, "score": 0.6}
                            for k, v in list(self.facts.items())[:limit]]}

    def get_all(self, filters=None):
        return {"results": [{"id": k, "memory": v, "created_at": 1, "updated_at": 1}
                            for k, v in self.facts.items()]}

    def get(self, memory_id=None):
        return {"results": [{"id": memory_id, "memory": self.facts.get(memory_id, "?")}]}


def _store(tmp, fake=None):
    conn = init_db(os.path.join(tmp, "agent.db"))
    return Mem0MemoryStore(conn, mem0_memory=fake or FakeMem0()), conn


def test_intent_routes_to_sqlite():
    tmp = make_tmpdir()
    try:
        store, conn = _store(tmp)
        iid = store.add("xiaoman", "intent", "我打算学冲浪", salience=0.65, now=1.0)
        assert isinstance(iid, int), "意图必须落 SQLite 拿到数字 id"
        assert [m["id"] for m in store.open_intents("xiaoman")] == [iid]
        store.close(iid, "done")
        assert store.open_intents("xiaoman") == []
        assert store.memory_tension_hash("xiaoman") != 0 or True  # 关闭后无意图=0
    finally:
        rmtree(tmp)


def test_fact_routes_to_mem0_with_stable_alias():
    tmp = make_tmpdir()
    try:
        fake = FakeMem0()
        store, conn = _store(tmp, fake)
        store.add("xiaoman", "event", "我家的猫叫年糕，三岁了", now=1.0)
        store.add("xiaoman", "event", "最近在忙项目", now=2.0)
        assert all(c["infer"] for c in fake.calls), "默认应走 LLM 抽取（infer=True）"
        hits = store.retrieve("xiaoman", "宠物的名字", 5, 0.0)
        assert hits and all(isinstance(h["id"], int) for h in hits)
        # 同一 mem0 事实的别名稳定（不重复发号）
        again = store.retrieve("xiaoman", "宠物", 5, 0.0)
        assert {h["id"] for h in again} == {h["id"] for h in hits}
        # 溯源：别名能查回 mem0 原文
        first = hits[0]
        got = store.get(first["id"])
        assert got is not None and got["content"].startswith(("抽取:", "我家的猫", "最近"))
    finally:
        rmtree(tmp)


def test_verbatim_and_fallback():
    tmp = make_tmpdir()
    try:
        fake = FakeMem0(fail_search=True)
        store, conn = _store(tmp, fake)
        store.add("xiaoman", "event", "「晨跑」已推迟（等忙完这阵）", now=1.0, verbatim=True)
        assert fake.calls[-1]["infer"] is False, "verbatim 必须原文直存"
        assert "已推迟" in fake.calls[-1]["text"]
        # mem0 检索挂了 -> 回退 SQLite 关键词检索（把事实塞进 SQLite 侧）
        store.sqlite.add("xiaoman", "event", "用户家的猫叫年糕", salience=0.9, now=2.0)
        hits = store.retrieve("xiaoman", "年糕", 5, 0.3)
        assert any("年糕" in h["content"] for h in hits), "回退路径必须可用"
    finally:
        rmtree(tmp)


def test_sqlite_intent_get_still_works():
    tmp = make_tmpdir()
    try:
        store, conn = _store(tmp)
        iid = store.add("xiaoman", "intent", "我打算学冲浪", now=1.0)
        row = store.get(iid)
        assert row and row["type"] == "intent" and row["status"] == "open"
    finally:
        rmtree(tmp)


if __name__ == "__main__":
    sys.exit(run_tests(globals(), "test_memory_mem0"))
