"""记忆层：四类记忆（event/relation/intent/reflection）的写入、检索与溯源。

MVP 不引入向量库：检索 = 显著度 × 关键词相关度（子串/标签/二元组）× 时间衰减，
确定性排序，逐条可溯源到 memory id（原则 E / 风险防护"幻觉记忆"）。

写入侧三层字面去重（归一化相等 / 包含 / 编辑距离，吸收自 wxhook 社区实践）：
"同一件事换个说法"不再重复入库，重复陈述会刷新既有记忆（touch）而非新增行。
"""
import json
import re

from .utils.hash import stable_hash

MEMORY_TYPES = ("event", "relation", "intent", "reflection")

# 去重候选窗口：只跟最近的记忆比（全表编辑距离在大库上不可接受）
DEDUP_WINDOW = 200


class MemoryStore:

    def __init__(self, conn, halflife_days: float = 14.0):
        """halflife_days：检索排序的时间衰减半衰期（天）；0 = 关闭衰减。"""
        self.conn = conn
        self.halflife_days = float(halflife_days or 0.0)

    def add(self, role_id: str, mtype: str, content: str, salience: float = 0.5,
            tags=None, source_message_id=None, status: str = "open", now: float = 0.0,
            verbatim: bool = False, dedup: bool = True, person_id: int = None,
            scope: str = None) -> int:
        """verbatim 仅供 mem0 后端区分"原文直存/LLM抽取"，SQLite 实现忽略。
        与既有 open 状态记忆语义重复时：不新增行，touch 既有记忆并返回其 id。
        dedup=False 用于"引用包裹"式复合文本（如「意图」已取消（原因））——
        它天然包含被引用的原文，包含式去重会误杀，须显式关闭。"""
        assert mtype in MEMORY_TYPES, mtype
        if dedup:
            dup = self._find_duplicate(role_id, content, person_id=person_id, scope=scope)
            if dup is not None:
                if now and now > 0:
                    self.touch([dup], now)
                return dup
        cur = self.conn.execute(
            "INSERT INTO memories(role_id, type, content, source_message_id, salience,"
            " created_at, last_used_at, status, tags, person_id, scope)"
            " VALUES(?,?,?,?,?,?,NULL,?,?,?,?)",
            (role_id, mtype, content, source_message_id, float(salience), now, status,
             json.dumps(tags or [], ensure_ascii=False), person_id, scope))
        self.conn.commit()
        return cur.lastrowid

    def _find_duplicate(self, role_id: str, content: str, person_id=None, scope=None):
        """在最近 DEDUP_WINDOW 条 open 记忆里找"讲的是同一件事"的条目。

        去重必须限定在同一个人/同一作用域内，否则 A 和 B 说同样的话会被合并。
        """
        norm = normalize_fact(content)
        if not norm:
            return None
        sql = ("SELECT id, content FROM memories WHERE role_id=? AND status='open'"
               " AND " + ("person_id=?" if person_id is not None else "person_id IS NULL"))
        params = [role_id]
        if person_id is not None:
            params.append(person_id)
        if scope is not None:
            sql += " AND scope=?"
            params.append(scope)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(DEDUP_WINDOW)
        rows = self.conn.execute(sql, params).fetchall()
        for row in rows:
            other = normalize_fact(row["content"])
            if other and fact_similar(norm, other):
                return row["id"]
        return None

    def get(self, memory_id: int):
        row = self.conn.execute("SELECT * FROM memories WHERE id=?", (memory_id,)).fetchone()
        return _row_to_dict(row) if row else None

    def retrieve(self, role_id: str, query: str, top_k: int, min_salience: float,
                 now: float = None, person_id: int = None, scope: str = None) -> list:
        """按 显著度×相关度×时间衰减 排序取 top_k。query 为空时退化为按显著度取。
        衰减只作用于排序（不删数据）：半衰期 halflife_days 天未被触碰的记忆权重减半，
        下限 0.35——老而相关的记忆仍能被高相关度拉回来。"""
        sql = "SELECT * FROM memories WHERE role_id=? AND status != 'expired'"
        params = [role_id]
        if person_id is not None:
            sql += " AND person_id=?"
            params.append(person_id)
        if scope is not None:
            sql += " AND scope=?"
            params.append(scope)
        rows = self.conn.execute(sql, params).fetchall()
        scored = []
        for row in rows:
            row_d = _row_to_dict(row)
            if row_d["salience"] < min_salience:
                continue
            score = row_d["salience"] * _relevance(query, row_d)
            score *= self._decay_factor(row_d, now)
            scored.append((score, row_d))
        scored.sort(key=lambda t: (-t[0], -t[1]["id"]))
        picked = [d for _, d in scored[:top_k]]
        if picked and now is not None:
            self.touch([d["id"] for d in picked], now)
        return picked

    def _decay_factor(self, memory: dict, now: float) -> float:
        if not self.halflife_days or now is None:
            return 1.0
        ref = memory["last_used_at"] if memory["last_used_at"] is not None else memory["created_at"]
        if ref is None:  # 无时间信息不衰减
            return 1.0
        age_days = max(0.0, (now - ref) / 86400.0)
        return max(0.35, 0.5 ** (age_days / self.halflife_days))

    def retrieve_stale(self, role_id: str, top_k: int, now: float = None) -> list:
        """翻旧账：显著度高、且久未被提起的记忆（主动消息"突然想起来"的素材）。
        与 retrieve 的区别：不按 query 相关度，按 新颖度反向 排——越久没聊过越优先。"""
        rows = self.conn.execute(
            "SELECT * FROM memories WHERE role_id=? AND status != 'expired'"
            " AND type != 'intent' AND salience >= 0.5", (role_id,)).fetchall()
        items = [_row_to_dict(r) for r in rows]
        items.sort(key=lambda d: (d["last_used_at"] or d["created_at"] or 0,
                                  -d["salience"]))
        return items[:top_k]

    def open_intents(self, role_id: str) -> list:
        rows = self.conn.execute(
            "SELECT * FROM memories WHERE role_id=? AND type='intent' AND status='open'"
            " ORDER BY salience DESC, id DESC", (role_id,)).fetchall()
        return [_row_to_dict(r) for r in rows]

    def close(self, memory_id: int, status: str = "done"):
        assert status in ("done", "expired")
        self.conn.execute("UPDATE memories SET status=? WHERE id=?", (status, memory_id))
        self.conn.commit()

    def touch(self, memory_ids, now: float):
        for mid in memory_ids:
            self.conn.execute("UPDATE memories SET last_used_at=? WHERE id=?", (now, mid))
        self.conn.commit()

    def memory_tension_hash(self, role_id: str) -> int:
        """未完成意图的"张力"指纹：有惦记的事时，心跳噪声随之改变。"""
        open_items = [(m["id"], round(m["salience"], 3), m["content"])
                      for m in self.open_intents(role_id)]
        return stable_hash(open_items) if open_items else 0


# ---------------- 三层字面去重（吸收自 wxhook 社区实践，零依赖） ----------------

# 注意 ASCII 引号必须留在字符类里：用三引号原始串，避免相邻拼接丢掉 r 前缀
_STRIP_PUNCT = re.compile(r"""[\s，。！？!?~…,.、""''（）()\[\]{}:：;；]""")
_STRIP_BRACKET = re.compile(r"\[[^\]]{0,20}\]")
_STRIP_AT = re.compile(r"@[^\s@]{1,24}")


def normalize_fact(s: str) -> str:
    """记忆条目归一化：去 [M#] 标记/@人/标点空白/大小写，限长 220。"""
    s = _STRIP_BRACKET.sub("", str(s or ""))
    s = _STRIP_AT.sub("", s)
    return _STRIP_PUNCT.sub("", s).lower()[:220]


def fact_similar(a_norm: str, b_norm: str) -> bool:
    """两条归一化后的记忆是否"讲的是同一件事"：相等 / 包含 / 编辑距离>=0.8。
    阈值偏保守：宁可漏合并，不把两件不同的事错并。"""
    if not a_norm or not b_norm:
        return False
    if a_norm == b_norm:
        return True
    if len(a_norm) >= 4 and len(b_norm) >= 4 and (a_norm in b_norm or b_norm in a_norm):
        return True
    max_len = max(len(a_norm), len(b_norm))
    if abs(len(a_norm) - len(b_norm)) / max_len > 0.5:
        return False  # 长度差太多，直接否
    return 1 - edit_distance(a_norm, b_norm) / max_len >= 0.8


def edit_distance(a: str, b: str) -> int:
    """Levenshtein 编辑距离（两行滚动数组）。"""
    m, n = len(a), len(b)
    if not m:
        return n
    if not n:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        ai = a[i - 1]
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (0 if ai == b[j - 1] else 1))
        prev = cur
    return prev[n]


def _row_to_dict(row) -> dict:
    return {
        "id": row["id"], "role_id": row["role_id"], "type": row["type"],
        "content": row["content"], "source_message_id": row["source_message_id"],
        "salience": row["salience"], "created_at": row["created_at"],
        "last_used_at": row["last_used_at"], "status": row["status"],
        "tags": json.loads(row["tags"]),
        "person_id": (row["person_id"] if "person_id" in row.keys() else None),
        "scope": (row["scope"] if "scope" in row.keys() else None),
    }


def _bigrams(text: str) -> set:
    text = "".join(text.split())
    return {text[i:i + 2] for i in range(len(text) - 1)} if len(text) >= 2 else ({text} if text else set())


def _relevance(query: str, memory: dict) -> float:
    query = (query or "").strip()
    if not query:
        return 1.0
    content = memory["content"]
    if query in content or content in query:
        return 1.5
    if any(tag and tag in query for tag in memory["tags"]):
        return 1.2
    qb, cb = _bigrams(query), _bigrams(content)
    if qb and cb:
        ratio = len(qb & cb) / len(qb)
        if ratio > 0:
            return 0.5 + ratio
    return 0.3
