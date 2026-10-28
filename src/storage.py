"""SQLite 初始化与迁移。schema 单一来源是 db/schema.sql。"""
import json
import os
import sqlite3

SCHEMA_VERSION = 6

# 未来版本迁移：追加 (目标版本, SQL) 即可；init_db 会按序应用
MIGRATIONS = [
    # v4：群聊多说话者结构。sender 只分 user/agent；speaker/speaker_role 才是
    # "这句话是谁说的"（owner/member/agent），避免所有群友都被当成同一个"你"。
    (4,
     "ALTER TABLE messages ADD COLUMN speaker TEXT;"
     "ALTER TABLE messages ADD COLUMN speaker_role TEXT;"
     "ALTER TABLE messages ADD COLUMN reply_to INTEGER;"),
    # v6：审核 Agent（双 Agent 的第二个人）留痕：每条出门的话都记原稿/病灶/
    # 改写要求/终稿，控制台可回溯"这句话是谁放行的"。
    (6, "CREATE TABLE IF NOT EXISTS review_logs (id INTEGER PRIMARY KEY AUTOINCREMENT, role_id TEXT NOT NULL, purpose TEXT NOT NULL, mode TEXT NOT NULL, issues TEXT NOT NULL DEFAULT '[]', correction TEXT, original TEXT NOT NULL, final TEXT NOT NULL, retried INTEGER NOT NULL DEFAULT 0, llm_used INTEGER NOT NULL DEFAULT 0, llm_failed INTEGER NOT NULL DEFAULT 0, latency_ms INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL);"
        "CREATE INDEX IF NOT EXISTS idx_review_logs_role_time ON review_logs(role_id, created_at DESC);"),
    (5, "ALTER TABLE memories ADD COLUMN person_id INTEGER;"
        "ALTER TABLE memories ADD COLUMN scope TEXT;"
        "CREATE TABLE IF NOT EXISTS persons (id INTEGER PRIMARY KEY AUTOINCREMENT, role_id TEXT NOT NULL, wxid TEXT NOT NULL, name TEXT NOT NULL DEFAULT '', kind TEXT NOT NULL DEFAULT 'member', first_seen REAL NOT NULL, last_seen REAL NOT NULL, UNIQUE(role_id, wxid));"
        "CREATE TABLE IF NOT EXISTS person_cards (person_id INTEGER PRIMARY KEY, affinity REAL NOT NULL DEFAULT 0.3, familiarity REAL NOT NULL DEFAULT 0.1, trust_identity REAL NOT NULL DEFAULT 0.5, trust_benevolence REAL NOT NULL DEFAULT 0.5, trust_competence REAL NOT NULL DEFAULT 0.4, trust_disclosure REAL NOT NULL DEFAULT 0.1, caution REAL NOT NULL DEFAULT 0.3, attachment REAL NOT NULL DEFAULT 0.0, authority TEXT NOT NULL DEFAULT 'none', updated_at REAL NOT NULL DEFAULT 0, version INTEGER NOT NULL DEFAULT 1);"
        "CREATE TABLE IF NOT EXISTS person_events (id INTEGER PRIMARY KEY AUTOINCREMENT, role_id TEXT NOT NULL, person_id INTEGER NOT NULL, event_type TEXT NOT NULL, weight REAL NOT NULL DEFAULT 0, content TEXT NOT NULL DEFAULT '', source_message_id INTEGER, value_before TEXT, value_after TEXT, created_at REAL NOT NULL);"
        "CREATE TABLE IF NOT EXISTS person_facts (id INTEGER PRIMARY KEY AUTOINCREMENT, role_id TEXT NOT NULL, person_id INTEGER NOT NULL, content TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'claim', confidence REAL NOT NULL DEFAULT 0.5, source_message_id INTEGER, created_at REAL NOT NULL);"),
]


def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_db_path() -> str:
    return os.path.join(project_root(), "data", "agent.db")


def init_db(path: str = None, threaded: bool = False) -> sqlite3.Connection:
    """threaded=True 时允许多线程共用连接（调用方需自行串行化，如 Agent 的锁）。"""
    path = path or default_db_path()
    if path != ":memory:":
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=not threaded)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version < SCHEMA_VERSION:
        schema_path = os.path.join(project_root(), "db", "schema.sql")
        with open(schema_path, "r", encoding="utf-8") as f:
            conn.executescript(f.read())
        # version=0 是全新库：schema.sql 已是当前版本，不再重放历史 ALTER。
        # 只有从旧版本升级的库才按序执行 MIGRATIONS。
        if version > 0:
            for target_version, sql in MIGRATIONS:
                if version < target_version:
                    conn.executescript(sql)
        conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        conn.commit()
    return conn


# ---------------- settings kv ----------------

def get_setting(conn: sqlite3.Connection, key: str, default=None):
    row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return json.loads(row["value"]) if row else default


def set_setting(conn: sqlite3.Connection, key: str, value):
    conn.execute(
        "INSERT INTO settings(key, value) VALUES(?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, json.dumps(value, ensure_ascii=False)))
    conn.commit()


# ---------------- states 行 ----------------

def load_state(conn: sqlite3.Connection, role_id: str):
    row = conn.execute("SELECT data FROM states WHERE role_id=?", (role_id,)).fetchone()
    return json.loads(row["data"]) if row else None


def save_state(conn: sqlite3.Connection, role_id: str, data: dict, now: float):
    conn.execute(
        "INSERT INTO states(role_id, data, updated_at) VALUES(?,?,?) "
        "ON CONFLICT(role_id) DO UPDATE SET data=excluded.data, updated_at=excluded.updated_at",
        (role_id, json.dumps(data, ensure_ascii=False, sort_keys=True), now))
    conn.commit()
