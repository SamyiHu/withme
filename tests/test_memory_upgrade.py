"""记忆升级验收：三层字面去重（换说法不算新记忆）、检索时间衰减、
mem0 攒批提炼（节流/失败保险/批内与存量去重）。全部离线假件。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import make_tmpdir, rmtree, run_tests

from src.config import MemoryConfig, config_from_dict
from src.memory import MemoryStore, edit_distance, fact_similar, normalize_fact
from src.memory_mem0 import Mem0MemoryStore
from src.storage import init_db

DAY = 86400.0


# ---------------- 三层去重：判定函数 ----------------

def test_normalize_and_similarity_layers():
    assert normalize_fact("用户说：他家的猫叫[M12]年糕！") == normalize_fact(
        "用户说他家的猫叫年糕")
    assert fact_similar("用户想学冲浪", "用户想学冲浪")
    # 包含关系（更完整的表述算同一件事）
    assert fact_similar("老王是群主", "老王是群主，管得很松")
    # 编辑距离：中间换一个字
    assert fact_similar("用户下周三要去体检", "用户下周四要去体检")
    # 不同的事：不合并
    assert not fact_similar("用户想学冲浪", "用户想吃火锅")
    # 长度差太多直接否
    assert not fact_similar("好的", "用户答应周日下午一起去逛宜家买书架")
    assert edit_distance("abc", "abd") == 1 and edit_distance("", "ab") == 2


# ---------------- 去重入库：换说法不重复、重复陈述刷新 ----------------

def test_add_dedup_rephrase_and_touch():
    tmp = make_tmpdir()
    try:
        store = MemoryStore(init_db(os.path.join(tmp, "a.db")))
        a = store.add("r", "event", "用户家的猫叫年糕，三岁了", salience=0.7, now=100.0)
        # 换说法（标点/语序微调） -> 同一条
        b = store.add("r", "event", "用户家的猫叫年糕，三岁了～", now=200.0)
        assert b == a, "换说法的重复陈述应命中既有记忆"
        assert store.get(a)["last_used_at"] == 200.0, "重复陈述应刷新 last_used_at"
        n = store.conn.execute("SELECT COUNT(*) c FROM memories").fetchone()["c"]
        assert n == 1
        # 完全不同的事 -> 新增
        c = store.add("r", "event", "用户下周要交项目报告", now=201.0)
        assert c != a and store.conn.execute(
            "SELECT COUNT(*) c FROM memories").fetchone()["c"] == 2
    finally:
        rmtree(tmp)


def test_add_dedup_ignores_closed_intents():
    tmp = make_tmpdir()
    try:
        store = MemoryStore(init_db(os.path.join(tmp, "a.db")))
        i1 = store.add("r", "intent", "用户打算周末整理房间", salience=0.7, now=1.0)
        store.close(i1, "done")  # 这件事已经完成
        # 用户几周后又提出同一个计划 -> 必须是新意图（不能命中已关闭的）
        i2 = store.add("r", "intent", "用户打算周末整理房间", now=2.0)
        assert i2 != i1
        assert [m["id"] for m in store.open_intents("r")] == [i2]
    finally:
        rmtree(tmp)


# ---------------- 检索时间衰减 ----------------

def test_retrieve_decay_ranking():
    tmp = make_tmpdir()
    try:
        conn = init_db(os.path.join(tmp, "a.db"))
        store = MemoryStore(conn, halflife_days=1.0)  # 半衰期 1 天
        old = store.add("r", "event", "用户喜欢冲浪和海边", salience=0.9, now=0.0)
        new = store.add("r", "event", "用户最近喜欢冲浪", salience=0.5, now=3 * DAY)
        # 两者都与查询相关；旧记忆 3 天未被触碰 -> 衰减到底，新的排前面
        hits = store.retrieve("r", "冲浪", top_k=5, min_salience=0.3, now=3 * DAY + 60)
        assert hits[0]["id"] == new and any(h["id"] == old for h in hits)
        # 关闭衰减 -> 显著度高者回到第一
        store.halflife_days = 0.0
        hits2 = store.retrieve("r", "冲浪", top_k=5, min_salience=0.3, now=3 * DAY + 61)
        assert hits2[0]["id"] == old
        # touch 复活：旧记忆被检索命中后 last_used_at 刷新，衰减重算
        store.halflife_days = 1.0
        store.touch([old], 3 * DAY + 62)
        hits3 = store.retrieve("r", "冲浪", top_k=5, min_salience=0.3, now=3 * DAY + 63)
        assert hits3[0]["id"] == old, "刚被触碰的高显著度记忆应重新排到最前"
    finally:
        rmtree(tmp)


def test_decay_floor_keeps_old_relevant():
    tmp = make_tmpdir()
    try:
        conn = init_db(os.path.join(tmp, "a.db"))
        store = MemoryStore(conn, halflife_days=1.0)
        old = store.add("r", "event", "用户对花生严重过敏", salience=0.6, now=0.0)
        junk = store.add("r", "event", "用户随口提过一次天气不错", salience=0.55, now=100 * DAY)
        hits = store.retrieve("r", "过敏 花生", top_k=5, min_salience=0.3, now=100 * DAY + 60)
        assert hits and hits[0]["id"] == old, "衰减有下限：老而强相关的记忆不能输给新闲聊"
    finally:
        rmtree(tmp)


# ---------------- mem0 攒批提炼 ----------------

class FakeMem0:
    def __init__(self):
        self.calls = []

    def add(self, messages, user_id=None, metadata=None, infer=True, **kw):
        self.calls.append({"text": messages[0]["content"], "infer": infer})
        return {"results": [{"id": f"m{len(self.calls)}",
                             "memory": messages[0]["content"], "event": "ADD"}]}

    def search(self, query, filters=None, limit=5):
        return {"results": []}

    def get_all(self, filters=None):
        return {"results": []}


class FakeLLM:
    def __init__(self, out=""):
        self.out = out
        self.calls = []

    def generate_response(self, messages, **kw):
        self.calls.append(messages)
        if isinstance(self.out, Exception):
            raise self.out
        return self.out


def _batch_store(tmp, **mcfg_over):
    fake = FakeMem0()
    cfg = MemoryConfig(batch_extract=True, batch_min_items=3,
                       batch_min_minutes=0.0, batch_max_fail=2)
    for k, v in mcfg_over.items():
        setattr(cfg, k, v)
    conn = init_db(os.path.join(tmp, "a.db"))
    store = Mem0MemoryStore(conn, mem0_memory=fake, mcfg=cfg)
    store._extract_llm = FakeLLM()
    return store, fake


def test_batch_extract_flow_and_dedup():
    tmp = make_tmpdir()
    try:
        store, fake = _batch_store(tmp)
        # 直连（无 mcfg）不受影响——见 test_memory_mem0 既有断言
        for i in range(3):
            store.add("r", "event", f"用户闲聊消息{i}", now=1000.0 + i)
        assert fake.calls == [], "攒批模式下候选先进缓冲，不逐条入库"
        assert store._pending_count("r") == 3
        # 未到水位（batch_min_items=3 已到，但先验证节流的间隔分支）
        store.mcfg.batch_min_items = 5
        assert store.maybe_extract("r") == 0 and store._pending_count("r") == 3
        store.mcfg.batch_min_items = 3
        # 提炼：LLM 给出两条（其中一条是前一条的微转述）-> 只入库 1 条
        store._extract_llm.out = ("- 用户养了一只猫叫年糕，今年三岁\n"
                                  "用户养了一只猫叫年糕，三岁\n"
                                  + "无意义的超长行" * 30)
        n = store.maybe_extract("r")
        assert n == 1, "批内换说法重复应被去掉"
        assert len(fake.calls) == 1 and fake.calls[0]["infer"] is False
        assert store._pending_count("r") == 0, "已审视的候选应出队"
        # 再攒一批：与已记事实微转述（标点/措辞微调）重复 -> 不新增；
        # 远距离改写按设计放行（阈值保守，宁可漏合并也不错杀——语义合并留给 mem0 向量侧）
        for i in range(3):
            store.add("r", "event", f"更多闲聊{i}", now=2000.0 + i)
        store._extract_llm.out = "用户养了一只猫，叫年糕，今年三岁。"
        assert store.maybe_extract("r") == 0, "与存量的微转述重复，不应新增"
        # "无" -> 成功出队但零新增
        for i in range(3):
            store.add("r", "event", f"水消息{i}", now=3000.0 + i)
        store._extract_llm.out = "无"
        assert store.maybe_extract("r") == 0 and store._pending_count("r") == 0
    finally:
        rmtree(tmp)


def test_batch_extract_failure_insurance():
    tmp = make_tmpdir()
    try:
        store, fake = _batch_store(tmp)
        store._extract_llm.out = RuntimeError("llm down")
        for i in range(3):
            store.add("r", "event", f"消息{i}", now=100.0 + i)
        assert store.maybe_extract("r") == -1
        assert store._pending_count("r") == 3, "失败必须保留缓冲"
        assert store._batch_state("r")["fails"] == 1
        # 连续失败第二次后自动停用：第三次不再调 LLM
        store.maybe_extract("r")  # 第 2 次失败
        assert store._batch_state("r")["fails"] == 2
        store._extract_llm.calls.clear()
        store._extract_llm.out = "- 用户喜欢爬山"
        assert store.maybe_extract("r") == 0
        assert store._extract_llm.calls == [], "停用后不应再调用 LLM"
        # force 越过停用：恢复成功并复位
        assert store.maybe_extract("r", force=True) == 1
        assert store._batch_state("r")["fails"] == 0
        assert store._pending_count("r") == 0
    finally:
        rmtree(tmp)


def test_config_memory_new_fields():
    cfg = config_from_dict({"memory": {"salience_halflife_days": 7.5,
                                       "batch_min_items": 4}})
    assert cfg.memory.salience_halflife_days == 7.5
    assert cfg.memory.batch_min_items == 4 and cfg.memory.batch_extract is True
    try:
        config_from_dict({"memory": {"salience_halflife_days": -1}})
        raise AssertionError("负半衰期应被拒绝")
    except Exception as e:
        assert "salience_halflife_days" in str(e)


if __name__ == "__main__":
    sys.exit(run_tests(globals(), name="test_memory_upgrade"))
