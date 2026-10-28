"""P2/P3 验收：记忆可溯源——主动引用必须能查到 ID，检索不到不得编造。"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import make_tmpdir, run_tests

from src.llm_client import MockLLMClient
from src.memory import MemoryStore
from src.storage import init_db


def _store(tmp):
    conn = init_db(os.path.join(tmp, "agent.db"))
    return MemoryStore(conn), conn


def test_add_get_roundtrip():
    tmp = make_tmpdir()
    try:
        store, conn = _store(tmp)
        mid = store.add("xiaoman", "event", "用户说这周要交项目报告", salience=0.7,
                        tags=["工作"], source_message_id=3, now=1000.0)
        m = store.get(mid)
        assert m["id"] == mid and m["type"] == "event"
        assert m["content"] == "用户说这周要交项目报告"
        assert m["tags"] == ["工作"] and m["status"] == "open"
        assert store.get(99999) is None
        conn.close()
    finally:
        import shutil; shutil.rmtree(tmp, ignore_errors=True)


def test_retrieve_stale_prefers_old_untouched():
    """翻旧账：显著度高且久未提起的优先，刚用过的靠后。主动消息防复读的素材层。"""
    tmp = make_tmpdir()
    try:
        store, conn = _store(tmp)
        old_a = store.add("xiaoman", "event", "用户的老家在肇庆", salience=0.7, now=1000.0)
        old_b = store.add("xiaoman", "event", "用户喜欢在周末钓鱼", salience=0.7, now=1000.0)
        fresh = store.add("xiaoman", "event", "用户刚聊过桂林山水", salience=0.9, now=2000.0)
        store.touch([fresh], 2000.0)   # 刚被用过
        out = store.retrieve_stale("xiaoman", 2, now=3000.0)
        ids = [m["id"] for m in out]
        assert set(ids) == {old_a, old_b}, f"刚用过的 fresh 不该进翻旧账: {ids}"
        conn.close()
    finally:
        import shutil; shutil.rmtree(tmp, ignore_errors=True)


def test_retrieve_ranking_and_filters():
    tmp = make_tmpdir()
    try:
        store, conn = _store(tmp)
        a = store.add("xiaoman", "event", "用户想学冲浪，说等夏天去海边", salience=0.8,
                      tags=["运动"], now=1.0)
        b = store.add("xiaoman", "intent", "用户打算周末整理房间", salience=0.6, now=2.0)
        store.add("xiaoman", "reflection", "用户似乎不喜欢被连续追问", salience=0.4, now=3.0)
        low_id = store.add("xiaoman", "event", "无关低显著度记忆", salience=0.1, now=4.0)
        # 相关度：查询词与 a 内容重叠
        hits = store.retrieve("xiaoman", "冲浪 海边", top_k=5, min_salience=0.3, now=5.0)
        assert hits and hits[0]["id"] == a, hits
        assert all(h["id"] != low_id for h in hits), "低于 min_salience 的必须被过滤"
        assert all(h["salience"] >= 0.3 for h in hits)
        # top_k 生效
        assert len(store.retrieve("xiaoman", "", top_k=2, min_salience=0.3)) == 2
        # 空查询按显著度排序
        ranked = store.retrieve("xiaoman", "", top_k=5, min_salience=0.3)
        assert [h["id"] for h in ranked][:2] == [a, b]
        # touch 更新 last_used_at
        assert store.get(a)["last_used_at"] == 5.0
        # 其他角色的记忆不可见
        other = store.add("other_role", "event", "别人的冲浪计划", salience=0.9, now=6.0)
        assert all(h["id"] != other for h in store.retrieve("xiaoman", "冲浪", 5, 0.3))
        conn.close()
    finally:
        import shutil; shutil.rmtree(tmp, ignore_errors=True)


def test_open_intents_and_tension_hash():
    tmp = make_tmpdir()
    try:
        store, conn = _store(tmp)
        assert store.open_intents("xiaoman") == []
        h0 = store.memory_tension_hash("xiaoman")
        i1 = store.add("xiaoman", "intent", "用户想学冲浪", salience=0.8, now=1.0)
        h1 = store.memory_tension_hash("xiaoman")
        assert h1 != h0, "出现未完成意图后，张力哈希必须变化（种子随记忆改变）"
        intents = store.open_intents("xiaoman")
        assert [m["id"] for m in intents] == [i1]
        store.close(i1)
        assert store.open_intents("xiaoman") == []
        assert store.memory_tension_hash("xiaoman") == h0
        conn.close()
    finally:
        import shutil; shutil.rmtree(tmp, ignore_errors=True)


def test_no_fabricated_memory_reference():
    """mock 渲染 share_related_memory 时，引用的 [M#] 必须来自给定列表，
    且列表中每个 id 都能回查到记忆原文。"""
    tmp = make_tmpdir()
    try:
        conn = init_db(os.path.join(tmp, "agent.db"))
        store = MemoryStore(conn)
        id_a = store.add("xiaoman", "event", "用户想学冲浪", salience=0.8)
        id_b = store.add("xiaoman", "event", "用户提到想吃火锅", salience=0.6)
        given = [{"id": id_a, "type": "event", "salience": 0.8, "内容": "用户想学冲浪"},
                 {"id": id_b, "type": "event", "salience": 0.6, "内容": "用户提到想吃火锅"}]
        client = MockLLMClient({"catchphrases": ["嗯哼", "诶"]})
        prompt = {
            "相关记忆": given,
            "未完成意图": [],
            "当前状态": {"数值": {"energy": 0.6}},
            "本次任务": {"意图": "share_related_memory", "来源": f"memory:{id_a}",
                         "指令": "只渲染该意图"},
        }
        for seed in range(30):
            text = client.render(prompt, seed)
            refs = {int(x) for x in re.findall(r"\[M(\d+)\]", text)}
            assert refs and refs <= {id_a, id_b}, f"seed={seed} 引用了未提供的记忆: {refs}"
        # 溯源链路：prompt 中的每个记忆 id 都能查回原文
        for m in given:
            row = store.get(m["id"])
            assert row is not None and row["content"] == m["内容"]
        conn.close()
    finally:
        import shutil; shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(run_tests(globals(), "test_memory_traceable"))
