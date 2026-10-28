"""Mem0 记忆后端：事实记忆（event/reflection）走 mem0（本地中文向量语义检索 +
qdrant 本地持久化），意图生命周期仍走 SQLite（open/done/expired 是闸门语义，
mem0 没有）。

事实写入默认**攒批提炼**（吸收自 wxhook 社区实践）：候选消息先进 mem_pending
缓冲，攒够 batch_min_items 条且距上次提炼 >= batch_min_minutes 才做一次 LLM
提炼（比逐条 infer 省 token；prompt 带已有记忆防"换说法重复"）。提炼结果过
三层字面去重后原文直存（infer=False）。失败保险：空返回重试一次；失败保留
缓冲不清空；连续 batch_max_fail 次自动停用（force 可越过）。

依赖：pip install mem0ai fastembed；国内网络需 HF_ENDPOINT=https://hf-mirror.com。
LLM 提炼走桥接（自定义 BridgeLLM，system 并入 user），embedding 本地 ONNX。
"""
import json
import os
import threading
import time
import urllib.request

from .memory import MemoryStore, fact_similar, normalize_fact
from .perception import extract_json

MEM0_DIMS = 512  # BAAI/bge-small-zh-v1.5 的维度


class BridgeLLM:
    """mem0 的桥接 LLM 适配器：桥接丢 system 角色，消息合并进 user，只回 JSON。"""

    def __init__(self, base_url: str, model: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def generate_response(self, messages, response_format=None, tools=None,
                          tool_choice="auto", **kwargs):
        parts = [f"【{m.get('role')}】\n{m['content']}" for m in messages if m.get("content")]
        text = "\n\n".join(parts) + "\n\n只输出要求的 JSON 对象，禁止任何其他文字。"
        body = {"model": self.model,
                "messages": [{"role": "user", "content": text}],
                "temperature": 0.1, "max_tokens": 1200}
        last_err = None
        for attempt in range(2):
            req = urllib.request.Request(
                self.base_url + "/chat/completions",
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {self.api_key}"},
                method="POST")
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    raw = json.loads(resp.read().decode("utf-8"))["choices"][0]["message"]["content"]
                obj = extract_json(raw)
                return json.dumps(obj, ensure_ascii=False) if obj is not None else raw
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(1.0)
        raise RuntimeError(f"mem0 桥接 LLM 调用失败：{last_err!r}")


_MEM0_CLIENTS = {}   # data_dir(abs) -> Memory：进程内单例。qdrant 本地盘一个路径
                     # 只允许一个客户端实例（跨进程靠文件锁，同进程第二实例也拒），
                     # 多会话必须共享同一个 mem0（写入自带 role_id 过滤，互不串)
_MEM0_BUILD_LOCK = threading.Lock()


def build_memory(data_dir: str, collection: str = "withme"):
    os.environ.setdefault("MEM0_TELEMETRY", "False")  # 遥测存储跨进程共享路径会锁冲突
    """构造 mem0 实例：qdrant 本地盘 + fastembed 中文向量 + 桥接 LLM。"""
    from mem0 import Memory
    from mem0.configs.base import MemoryConfig, VectorStoreConfig
    from mem0.embeddings.configs import EmbedderConfig

    key = (os.path.abspath(data_dir), collection)
    with _MEM0_BUILD_LOCK:
        if key in _MEM0_CLIENTS:
            return _MEM0_CLIENTS[key]
        cfg = MemoryConfig(
            vector_store=VectorStoreConfig(provider="qdrant", config={
                "collection_name": collection, "embedding_model_dims": MEM0_DIMS,
                "path": os.path.join(data_dir, "mem0_qdrant")}),
            llm={"provider": "openai", "config": {"api_key": "dummy"}},  # 占位，随后换成桥接
            embedder=EmbedderConfig(provider="fastembed",
                                    config={"model": "BAAI/bge-small-zh-v1.5"}),
            history_db_path=os.path.join(data_dir, "mem0_history.db"),
        )
        mem = Memory(cfg)
        mem.llm = BridgeLLM(
            os.environ.get("AGENT_LLM_BASE_URL", ""),
            os.environ.get("AGENT_LLM_MODEL", ""),
            os.environ.get("LLM_API_KEY", ""))
        _MEM0_CLIENTS[key] = mem
        return mem


class Mem0MemoryStore:
    """与 MemoryStore 同构的混合后端：意图走 SQLite，事实走 mem0。
    mcfg 传入时按其 batch_* 攒批提炼；不传（直连/测试）保持逐条 infer 的旧行为。"""

    def __init__(self, conn, data_dir: str = None, mem0_memory=None, mcfg=None,
                 session_scope: str = None):
        self.conn = conn
        self.mcfg = mcfg
        # 会话隔离：qdrant 库按 user_id 过滤，全部会话共用一个 role_id 会互相
        # 串记忆（家人群召回私聊的事）。主库（主人私聊）沿用裸 role_id 保历史，
        # 其余会话用 role_id:sid。None/'' /main 都视为主库。
        self.user_scope = None if not session_scope or session_scope in ("main", "default")             else str(session_scope)
        self.sqlite = MemoryStore(conn, halflife_days=getattr(mcfg, "salience_halflife_days", 14.0))
        self.mem = mem0_memory
        if self.mem is None:
            self.mem = build_memory(data_dir or os.path.join(os.getcwd(), "data"))
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS mem_aliases ("
            " alias INTEGER PRIMARY KEY AUTOINCREMENT,"
            " mem0_id TEXT UNIQUE NOT NULL,"
            " content TEXT NOT NULL,"
            " created_at REAL NOT NULL)")
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS mem_pending ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " role_id TEXT NOT NULL,"
            " content TEXT NOT NULL,"
            " created_at REAL NOT NULL)")
        self.conn.commit()
        # 测试可注入提炼 LLM（生产用 mem.llm = BridgeLLM）
        self._extract_llm = None

    def _uid(self, role_id: str) -> str:
        return f"{role_id}:{self.user_scope}" if self.user_scope else role_id

    # ---------------- 别名（保住 [M12] 式溯源显示） ----------------

    def _alias_of(self, mem0_id: str, content: str) -> int:
        row = self.conn.execute("SELECT alias FROM mem_aliases WHERE mem0_id=?",
                                (mem0_id,)).fetchone()
        if row:
            return row["alias"]
        cur = self.conn.execute(
            "INSERT INTO mem_aliases(mem0_id, content, created_at) VALUES(?,?,?)",
            (mem0_id, content, time.time()))
        self.conn.commit()
        return cur.lastrowid

    def _alias_to_mem0(self, alias: int):
        return self.conn.execute("SELECT * FROM mem_aliases WHERE alias=?",
                                 (alias,)).fetchone()

    def _recent_facts(self, role_id: str, limit: int = 24) -> list:
        rows = self.conn.execute(
            "SELECT content FROM mem_aliases ORDER BY alias DESC LIMIT ?", (limit,)).fetchall()
        return [r["content"] for r in rows]

    # ---------------- 写入 ----------------

    def add(self, role_id: str, mtype: str, content: str, salience: float = 0.5,
            tags=None, source_message_id=None, status: str = "open", now: float = 0.0,
            verbatim: bool = False, dedup: bool = True, person_id: int = None,
            scope: str = None):
        if mtype == "intent":
            return self.sqlite.add(role_id, mtype, content, salience, tags,
                                   source_message_id, status, now, dedup=dedup,
                                   person_id=person_id, scope=scope)
        # 攒批模式：候选消息进缓冲，由 maybe_extract 批量提炼（省 token）
        if (self.mcfg is not None and self.mcfg.batch_extract and not verbatim):
            self.conn.execute(
                "INSERT INTO mem_pending(role_id, content, created_at) VALUES(?,?,?)",
                (role_id, content, now or time.time()))
            self.conn.commit()
            return None
        return self._add_fact_now(role_id, mtype, content, verbatim,
                                  source_message_id, person_id=person_id,
                                  scope=scope)

    def _add_fact_now(self, role_id: str, mtype: str, content: str, verbatim: bool,
                      source_message_id=None, person_id=None, scope=None):
        metadata = {"message_id": source_message_id, "type": mtype,
                    "person_id": person_id, "scope": scope}
        try:
            result = self.mem.add(
                [{"role": "user", "content": content}],
                user_id=self._uid(role_id), metadata=metadata,
                infer=not verbatim)
            for item in (result or {}).get("results", []):
                if item.get("id"):
                    self._alias_of(item["id"], item.get("memory", content))
        except Exception as e:  # noqa: BLE001  事实抽取失败不阻塞聊天，回退 SQLite 原文
            print(f"[mem0] 事实入库失败，回退 SQLite：{e!r}")
            return self.sqlite.add(role_id, "event", content, 0.5, None,
                                   source_message_id, "open", 0.0,
                                   person_id=person_id, scope=scope)
        return None

    # ---------------- 批量提炼（节流 + 全套失败保险） ----------------

    def _pending_count(self, role_id: str) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) c FROM mem_pending WHERE role_id=?",
            (role_id,)).fetchone()["c"]

    def _batch_state(self, role_id: str) -> dict:
        try:
            from .storage import get_setting
            states = get_setting(self.conn, "mem0_batch_state", {}) or {}
            return states.get(role_id) or {"last_at": 0.0, "fails": 0}
        except Exception:  # noqa: BLE001
            return {"last_at": 0.0, "fails": 0}

    def _set_batch_state(self, role_id: str, state: dict):
        try:
            from .storage import get_setting, set_setting
            states = get_setting(self.conn, "mem0_batch_state", {}) or {}
            states[role_id] = state
            set_setting(self.conn, "mem0_batch_state", states)
        except Exception:  # noqa: BLE001  状态写不进不影响提炼本身
            pass

    def maybe_extract(self, role_id: str, force: bool = False) -> int:
        """攒够候选后做一次批量提炼。返回新增事实条数；失败返回 -1（缓冲保留）。
        节流：>= batch_min_items 条候选 且 距上次 >= batch_min_minutes；
        连续失败 >= batch_max_fail 次自动停用（force=True 越过节流与停用）。"""
        if self.mcfg is None or not self.mcfg.batch_extract:
            return 0
        state = self._batch_state(role_id)
        now = time.time()
        if not force:
            if state.get("fails", 0) >= self.mcfg.batch_max_fail:
                return 0  # 自动停用中（连续失败），等 force 手动唤醒
            if now - state.get("last_at", 0.0) < self.mcfg.batch_min_minutes * 60:
                return 0
            if self._pending_count(role_id) < self.mcfg.batch_min_items:
                return 0
        rows = self.conn.execute(
            "SELECT id, content FROM mem_pending WHERE role_id=?"
            " ORDER BY id ASC LIMIT 24", (role_id,)).fetchall()
        if not rows:
            return 0
        llm = getattr(self.mem, "llm", None) or self._extract_llm
        if llm is None:
            return -1  # 没有可用 LLM：缓冲保留，不丢数据

        dialogue = "\n".join("- " + r["content"][:120] for r in rows)
        existing = self._recent_facts(role_id)
        sys = "你是一个记忆整理器。只输出事实条目，每行一条，不要编号、不要解释。"
        usr = ("从下面的对话记录里提炼 1~4 条**值得长期记住**的事实（用户身份/偏好/习惯/"
               "重要事件/答应过别人的事）。\n每条写完整一点（30~100 字）：带上是谁、什么场合、"
               "什么细节，别只写半句。\n只记长期有用的；日常闲聊不要记。没有值得记的就只输出：无\n"
               + ("\n【已经记过下面这些，不要重复提炼同一件事（换个说法也算重复）】\n"
                  + "\n".join("- " + f[:120] for f in existing) + "\n" if existing else "")
               + "\n对话记录：\n" + dialogue)
        text = ""
        try:
            text = str(llm.generate_response(
                [{"role": "system", "content": sys}, {"role": "user", "content": usr}]) or "")
            if not text.strip():  # 空返回降级重试（system 并进 user 再问一次）
                text = str(llm.generate_response(
                    [{"role": "user", "content": sys + "\n\n" + usr}]) or "")
        except Exception as e:  # noqa: BLE001  失败：保留缓冲，记一次失败
            self._set_batch_state(role_id, {"last_at": state.get("last_at", 0.0),
                                            "fails": state.get("fails", 0) + 1})
            print(f"[mem0] 批量提炼失败（缓冲保留，第 {state.get('fails', 0) + 1} 次）：{e!r}")
            return -1

        facts = []
        lines = []
        for line in text.splitlines():
            t = line.strip().lstrip("-•·").strip()
            t = t.lstrip("0123456789.、） ").strip()
            if t.startswith("{") and '"facts"' in t:
                # 提炼 LLM 偶尔整行输出 JSON：拆出内部条目，别把 JSON 本身存成事实
                try:
                    inner = json.loads(t).get("facts") or []
                    lines.extend(str(x) for x in inner)
                except ValueError:
                    pass
                continue
            lines.append(t)
        for line in lines:
            t = line.strip().lstrip("-•·").strip()
            t = t.lstrip("0123456789.、） ").strip()
            if not t or t == "无" or len(t) > 200:
                continue
            n = normalize_fact(t)
            if any(fact_similar(n, normalize_fact(x)) for x in facts):
                continue  # 批内去重
            if any(fact_similar(n, normalize_fact(x)) for x in existing):
                continue  # 与已记过的重复（换说法也算）
            facts.append(t)
        for f in facts:
            self._add_fact_now(role_id, "event", f, verbatim=True)
        # 成功（含"无可记"）：已审视的候选出队，状态复位
        ids = [r["id"] for r in rows]
        self.conn.execute(
            "DELETE FROM mem_pending WHERE id IN (%s)" % ",".join("?" * len(ids)), ids)
        self.conn.commit()
        self._set_batch_state(role_id, {"last_at": now, "fails": 0})
        if facts:
            print(f"[mem0] 批量提炼：{len(rows)} 条候选 -> 新增 {len(facts)} 条事实")
        return len(facts)

    # ---------------- 检索 ----------------

    def retrieve(self, role_id: str, query: str, top_k: int, min_salience: float,
                 now: float = None, person_id: int = None, scope: str = None) -> list:
        query = (query or "").strip()
        try:
            if not query:
                items = self.mem.get_all(filters={"user_id": self._uid(role_id)}).get("results", [])
                picked = sorted(items, key=lambda m: m.get("updated_at") or 0,
                                reverse=True)[:top_k]
                scored = [(0.5, m) for m in picked]
            else:
                res = self.mem.search(query=query, filters={"user_id": self._uid(role_id)},
                                      limit=top_k).get("results", [])
                meaningful = [r for r in res if r.get("score", 0) >= 0.35]
                # 全都低于阈值就空手而归：硬塞最不相关的一条 = 给模型喂错记忆
                scored = [(r.get("score", 0.5), r) for r in meaningful]
        except Exception as e:  # noqa: BLE001  检索失败回退 SQLite 关键词检索
            print(f"[mem0] 语义检索失败，回退 SQLite：{e!r}")
            return self.sqlite.retrieve(role_id, query, top_k, min_salience, now,
                                        person_id=person_id, scope=scope)
        out = []
        for score, r in scored[:top_k]:
            meta = r.get("metadata") or {}
            out.append({
                "id": self._alias_of(r["id"], r["memory"]),
                "type": meta.get("type", "event"),
                "content": r["memory"],
                "salience": round(min(1.0, max(score, 0.35) * 1.6), 2),
                "tags": [], "status": "open",
                "source_message_id": meta.get("message_id"),
                "created_at": r.get("created_at"), "last_used_at": None,
            })
        return out

    def retrieve_stale(self, role_id: str, top_k: int, now: float = None) -> list:
        """翻旧账：向量库全量里按 updated_at 最久远的优先（mem0 无显著度元数据，
        用"最久没被更新"近似"最久没聊过"）。"""
        try:
            items = self.mem.get_all(filters={"user_id": self._uid(role_id)}).get("results", [])
        except Exception:  # noqa: BLE001  向量库失败就退化到 sqlite 侧
            return self.sqlite.retrieve_stale(role_id, top_k, now)
        items = sorted(items, key=lambda m: m.get("updated_at") or 0)[:top_k]
        out = []
        for m in items:
            if not isinstance(m, dict) or not m.get("memory"):
                continue
            out.append({"id": self._alias_of(m["id"], m["memory"]),
                        "type": (m.get("metadata") or {}).get("type", "event"),
                        "content": m["memory"], "salience": 0.5, "tags": [],
                        "status": "open", "source_message_id": None,
                        "created_at": m.get("created_at"), "last_used_at": None})
        return out

    # ---------------- 意图生命周期（SQLite 原样） ----------------

    def open_intents(self, role_id: str) -> list:
        return self.sqlite.open_intents(role_id)

    def close(self, memory_id: int, status: str = "done"):
        self.sqlite.close(memory_id, status)

    def touch(self, memory_ids, now: float):
        self.sqlite.touch(memory_ids, now)

    def memory_tension_hash(self, role_id: str) -> int:
        return self.sqlite.memory_tension_hash(role_id)

    # ---------------- 溯源 ----------------

    def get(self, memory_id: int):
        row = self.sqlite.get(memory_id)
        if row is not None:
            return row
        alias = self._alias_to_mem0(memory_id)
        if alias is None:
            return None
        content = alias["content"]
        created = alias["created_at"]
        try:
            detail = self.mem.get(memory_id=alias["mem0_id"])
            items = detail.get("results", []) if isinstance(detail, dict) else []
            if items:
                content = items[0].get("memory", content)
                created = items[0].get("created_at", created)
        except Exception:  # noqa: BLE001  本地别名里存有原文，查不回 mem0 也能溯源
            pass
        return {"id": memory_id, "role_id": None, "type": "event", "content": content,
                "source_message_id": None, "salience": 0.5, "created_at": created,
                "last_used_at": None, "status": "open", "tags": []}
