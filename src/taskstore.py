"""持久化任务与出站队列。

骨架来自 Super-agent 的两个做法：
- 任务（ticket/assignment 的轻量版）：先落库再执行，进程重启后能看出哪些
  任务被中断，而不是只活在线程字典里；
- 出站队列（outbox）：消息发送失败不直接丢，落库后按退避重试。

这里只做 WithMe 需要的最小字段，不引入完整的 Ticket/Mandate 领域模型。
"""
import hashlib
import json
import sqlite3
import threading
import time
import uuid

TASK_STATUSES = ("queued", "running", "ready", "failed", "cancelled")
ACTIVE_STATUSES = ("queued", "running")
OUTBOX_STATUSES = ("pending", "sent", "failed", "dropped")


def _now() -> float:
    return time.time()


def _json(value, default):
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


class _DbBacked:
    """后台任务/出站线程与主线程共用同一个库时，各用各的连接。

    Agent 的共享连接是 check_same_thread=False 的，但 sqlite 连接跨线程并发
    写并不安全；这里按线程惰性开连接（WAL + busy_timeout），主线程和后台
    song-worker 互不抢同一个连接。
    """

    def __init__(self, conn):
        self._shared = conn
        self._db_path = ""
        try:
            for row in conn.execute("PRAGMA database_list").fetchall():
                if row[1] == "main":
                    self._db_path = row[2] or ""
                    break
        except Exception:  # noqa: BLE001  取不到路径就退回共享连接
            self._db_path = ""
        self._local = threading.local()

    def _conn(self):
        if not self._db_path:
            return self._shared
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, timeout=5.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn = conn
        return conn

    @property
    def conn(self):
        return self._conn()


class TaskStore(_DbBacked):
    """每个会话库一份；写操作串行化，读操作直接查。"""

    def __init__(self, conn, role_id: str, lock=None, owner: str = ""):
        super().__init__(conn)
        self.role_id = role_id
        self.owner = owner or ""
        self._lock = lock or threading.Lock()
        # fail_orphans 最近一次实际标失败的任务行；桥启动后用它补一条人话通知。
        self.last_orphans = []

    def create(self, kind: str, params: dict = None, idempotency_key: str = None,
               status: str = "queued") -> dict:
        """新建任务；同一 idempotency_key 的活跃任务直接复用（防重复点歌）。"""
        params = params or {}
        with self._lock:
            if idempotency_key:
                row = self.conn.execute(
                    "SELECT * FROM tasks WHERE role_id=? AND idempotency_key=?"
                    " AND status IN ('queued','running') ORDER BY created_at DESC LIMIT 1",
                    (self.role_id, idempotency_key)).fetchone()
                if row is not None:
                    return dict(row)
            now = _now()
            task_id = uuid.uuid4().hex
            self.conn.execute(
                "INSERT INTO tasks(id, role_id, kind, stage, status, params, result,"
                " error, idempotency_key, owner, attempts, heartbeat_at, created_at, updated_at)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (task_id, self.role_id, kind, "", status, _json(params, "{}"),
                 None, None, idempotency_key, self.owner, 0, now, now, now))
            self.conn.commit()
        return self.get(task_id)

    def get(self, task_id: str) -> dict:
        row = self.conn.execute(
            "SELECT * FROM tasks WHERE id=? AND role_id=?",
            (task_id, self.role_id)).fetchone()
        return dict(row) if row is not None else {}

    def update(self, task_id: str, **fields) -> dict:
        allowed = {"stage", "status", "params", "result", "error", "attempts"}
        sets, values = [], []
        for key in allowed:
            if key not in fields:
                continue
            value = fields[key]
            if key in ("params", "result"):
                value = _json(value, "{}" if key == "params" else "null")
            sets.append(f"{key}=?")
            values.append(value)
        if not sets:
            return self.get(task_id)
        now = _now()
        sets += ["updated_at=?", "heartbeat_at=?"]
        values += [now, now, task_id, self.role_id]
        with self._lock:
            self.conn.execute(
                "UPDATE tasks SET " + ", ".join(sets) +
                " WHERE id=? AND role_id=?", values)
            self.conn.commit()
        return self.get(task_id)

    def heartbeat(self, task_id: str, stage: str = None) -> None:
        now = _now()
        with self._lock:
            if stage is None:
                self.conn.execute(
                    "UPDATE tasks SET heartbeat_at=?, updated_at=? WHERE id=? AND role_id=?",
                    (now, now, task_id, self.role_id))
            else:
                self.conn.execute(
                    "UPDATE tasks SET heartbeat_at=?, updated_at=?, stage=?"
                    " WHERE id=? AND role_id=?",
                    (now, now, stage, task_id, self.role_id))
            self.conn.commit()

    def fail_orphans(self, before: float = None, reason: str = "进程重启，任务中断",
                     owner: str = None) -> int:
        """把上一个进程留下的活跃任务标失败；默认只动进程启动前的。

        返回条数（兼容旧调用）；实际被改的行留在 ``last_orphans``，桥可以用它
        给用户补一句“刚才那首没唱完”，而不是只在健康页里静静变红。
        """
        cutoff = _now() if before is None else float(before)
        where = ("role_id=? AND status IN ('queued','running') AND heartbeat_at<?")
        args = [self.role_id, cutoff]
        if owner is not None:
            where += " AND owner=?"
            args.append(str(owner))
        with self._lock:
            rows = [dict(r) for r in self.conn.execute(
                "SELECT * FROM tasks WHERE " + where, args).fetchall()]
            now = _now()
            cur = self.conn.execute(
                "UPDATE tasks SET status='failed', error=?, updated_at=?, heartbeat_at=?"
                " WHERE " + where, [reason, now, now] + args)
            self.conn.commit()
        self.last_orphans = rows
        return int(cur.rowcount or 0)

    def stale(self, older_than: float = 900.0, statuses=("queued", "running"),
              owner: str = None) -> list:
        """心跳超过 older_than 秒的活跃任务（只读，给健康页/看门狗判断卡死）。"""
        marks = ",".join("?" for _ in statuses)
        cutoff = _now() - max(0.0, float(older_than))
        sql = ("SELECT * FROM tasks WHERE role_id=? AND status IN (" + marks + ")"
               " AND heartbeat_at<?")
        args = [self.role_id] + list(statuses) + [cutoff]
        if owner is not None:
            sql += " AND owner=?"
            args.append(str(owner))
        rows = self.conn.execute(sql + " ORDER BY heartbeat_at ASC", args).fetchall()
        return [dict(r) for r in rows]

    def fail_stale(self, older_than: float = 900.0,
                   reason: str = "任务超时无心跳，已中断",
                   owner: str = None) -> list:
        """把卡死的活跃任务标失败，返回被标失败的行（供桥通知用户）。"""
        cutoff = _now() - max(0.0, float(older_than))
        where = ("role_id=? AND status IN ('queued','running') AND heartbeat_at<?")
        args = [self.role_id, cutoff]
        if owner is not None:
            where += " AND owner=?"
            args.append(str(owner))
        with self._lock:
            rows = [dict(r) for r in self.conn.execute(
                "SELECT * FROM tasks WHERE " + where + " ORDER BY heartbeat_at ASC",
                args).fetchall()]
            if rows:
                now = _now()
                self.conn.execute(
                    "UPDATE tasks SET status='failed', error=?, updated_at=?, heartbeat_at=?"
                    " WHERE " + where, [reason, now, now] + args)
                self.conn.commit()
        return rows

    def counts(self) -> dict:
        rows = self.conn.execute(
            "SELECT status, COUNT(*) c FROM tasks WHERE role_id=? GROUP BY status",
            (self.role_id,)).fetchall()
        return {r["status"]: int(r["c"]) for r in rows}

    def recent(self, limit: int = 20) -> list:
        rows = self.conn.execute(
            "SELECT * FROM tasks WHERE role_id=? ORDER BY created_at DESC LIMIT ?",
            (self.role_id, max(1, min(int(limit), 200)))).fetchall()
        return [dict(r) for r in rows]


class OutboxStore(_DbBacked):
    """出站消息的持久化重试队列；发送失败才入队，成功后标记 sent。"""

    def __init__(self, conn, role_id: str, lock=None):
        super().__init__(conn)
        self.role_id = role_id
        self._lock = lock or threading.Lock()

    def enqueue(self, room: str, kind: str, payload: dict,
                dedupe_key: str = None, max_attempts: int = 6) -> int:
        with self._lock:
            if dedupe_key:
                row = self.conn.execute(
                    "SELECT id FROM outbox WHERE role_id=? AND dedupe_key=?"
                    " AND status='pending' ORDER BY id DESC LIMIT 1",
                    (self.role_id, dedupe_key)).fetchone()
                if row is not None:
                    return int(row["id"])
            now = _now()
            cur = self.conn.execute(
                "INSERT INTO outbox(role_id, room, kind, payload, status, attempts,"
                " next_retry_at, last_error, dedupe_key, created_at, updated_at)"
                " VALUES(?,?,?,?, 'pending', 0, ?, NULL, ?, ?, ?)",
                (self.role_id, room, kind, _json(payload, "{}"),
                 now, dedupe_key, now, now))
            self.conn.commit()
            return int(cur.lastrowid)

    def due(self, limit: int = 20) -> list:
        rows = self.conn.execute(
            "SELECT * FROM outbox WHERE role_id=? AND status='pending'"
            " AND next_retry_at<=? ORDER BY id ASC LIMIT ?",
            (self.role_id, _now(), max(1, min(int(limit), 100)))).fetchall()
        return [dict(r) for r in rows]

    def mark_sent(self, item_id: int) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE outbox SET status='sent', updated_at=? WHERE id=? AND role_id=?",
                (_now(), int(item_id), self.role_id))
            self.conn.commit()

    def retry(self, item_id: int, error: str, delay: float = 5.0,
              max_attempts: int = 6) -> str:
        """失败一次；达到上限转 failed 并返回 failed，否则 pending。"""
        with self._lock:
            row = self.conn.execute(
                "SELECT attempts FROM outbox WHERE id=? AND role_id=?",
                (int(item_id), self.role_id)).fetchone()
            attempts = (int(row["attempts"]) if row is not None else 0) + 1
            status = "failed" if attempts >= max_attempts else "pending"
            now = _now()
            self.conn.execute(
                "UPDATE outbox SET status=?, attempts=?, next_retry_at=?,"
                " last_error=?, updated_at=? WHERE id=? AND role_id=?",
                (status, attempts, now + max(0.0, float(delay)), str(error)[:300],
                 now, int(item_id), self.role_id))
            self.conn.commit()
            return status

    def mark_failed(self, item_id: int, error: str) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE outbox SET status='failed', last_error=?, updated_at=?"
                " WHERE id=? AND role_id=?",
                (str(error)[:300], _now(), int(item_id), self.role_id))
            self.conn.commit()

    def counts(self) -> dict:
        rows = self.conn.execute(
            "SELECT status, COUNT(*) c FROM outbox WHERE role_id=? GROUP BY status",
            (self.role_id,)).fetchall()
        return {r["status"]: int(r["c"]) for r in rows}

    def recent(self, limit: int = 20) -> list:
        rows = self.conn.execute(
            "SELECT * FROM outbox WHERE role_id=? ORDER BY id DESC LIMIT ?",
            (self.role_id, max(1, min(int(limit), 200)))).fetchall()
        return [dict(r) for r in rows]


class TriggerStore(_DbBacked):
    """入站触发账本：只审计，不参与权限判断。

    ``log`` 在同一条文本 1.5 秒内重复到达时返回 False（微信补收/重启回放
    的典型重复），桥据此不再走一遍 Agent；正常重复发言的间隔通常大于 1.5s。
    """

    def __init__(self, conn, role_id: str, lock=None):
        super().__init__(conn)
        self.role_id = role_id
        self._lock = lock or threading.Lock()

    def log(self, room: str, sender: str, kind: str, decision: str,
            reason: str = "", addressed: bool = False, text: str = "",
            dedupe_window: float = 1.5) -> bool:
        text = str(text or "")
        digest = hashlib.sha1(
            ("%s|%s|%s" % (room, sender, text)).encode("utf-8")).hexdigest()
        now = _now()
        with self._lock:
            if decision == "respond" and text:
                row = self.conn.execute(
                    "SELECT id FROM triggers WHERE role_id=? AND text_hash=?"
                    " AND created_at>=? ORDER BY id DESC LIMIT 1",
                    (self.role_id, digest, now - max(0.0, dedupe_window))).fetchone()
                if row is not None:
                    self.conn.execute(
                        "INSERT INTO triggers(role_id, room, sender, kind, decision,"
                        " reason, addressed, text_hash, preview, created_at)"
                        " VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (self.role_id, room, sender, kind, "duplicate",
                         "1.5s 内重复投递", 1 if addressed else 0, digest,
                         text[:120], now))
                    self.conn.commit()
                    return False
            self.conn.execute(
                "INSERT INTO triggers(role_id, room, sender, kind, decision,"
                " reason, addressed, text_hash, preview, created_at)"
                " VALUES(?,?,?,?,?,?,?,?,?,?)",
                (self.role_id, room, sender, kind, decision, str(reason or ""),
                 1 if addressed else 0, digest, text[:120], now))
            self.conn.commit()
        return True

    def recent(self, limit: int = 20) -> list:
        rows = self.conn.execute(
            "SELECT * FROM triggers WHERE role_id=? ORDER BY id DESC LIMIT ?",
            (self.role_id, max(1, min(int(limit), 200)))).fetchall()
        return [dict(r) for r in rows]
