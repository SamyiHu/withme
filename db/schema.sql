-- SQLite schema v1。storage.py 通过 PRAGMA user_version 做版本迁移。
CREATE TABLE IF NOT EXISTS settings (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

-- 每角色一行；state_vector/relation_state 等打包为 JSON 存 data 列
CREATE TABLE IF NOT EXISTS states (
  role_id    TEXT PRIMARY KEY,
  data       TEXT NOT NULL,
  updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS memories (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  role_id           TEXT NOT NULL,
  type              TEXT NOT NULL CHECK (type IN ('event','relation','intent','reflection')),
  content           TEXT NOT NULL,
  source_message_id INTEGER,
  salience          REAL NOT NULL DEFAULT 0.5,
  created_at        REAL NOT NULL,
  last_used_at      REAL,
  status            TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open','done','expired')),
  tags              TEXT NOT NULL DEFAULT '[]',
  person_id         INTEGER,   -- v5：这条记忆关于谁
  scope             TEXT       -- v5：private / group:<room> / global
);
CREATE INDEX IF NOT EXISTS idx_memories_role_status ON memories(role_id, status, salience DESC);

CREATE TABLE IF NOT EXISTS active_logs (
  id                    INTEGER PRIMARY KEY AUTOINCREMENT,
  role_id               TEXT NOT NULL,
  level                 INTEGER NOT NULL,
  intent_type           TEXT NOT NULL,
  source_type           TEXT,              -- intent | memory | event | null
  source_id             INTEGER,           -- 溯源：主动行为必须能指回来源
  rendered_text         TEXT NOT NULL,
  score_detail          TEXT NOT NULL,     -- 评分明细 JSON
  gate_result           TEXT NOT NULL,     -- 闸门判定明细 JSON
  user_response_latency REAL,
  user_sentiment_delta  REAL,
  created_at            REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_active_logs_role_time ON active_logs(role_id, created_at DESC);

CREATE TABLE IF NOT EXISTS messages (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  role_id    TEXT NOT NULL,
  sender     TEXT NOT NULL CHECK (sender IN ('user','agent')),
  content    TEXT NOT NULL,
  meta       TEXT,
  created_at REAL NOT NULL,
  -- v4：多说话者结构。sender 只分 user/agent；speaker/speaker_role 才是
  -- "这句话是谁说的"（owner/member/agent）。旧库由 storage.MIGRATIONS 迁移。
  speaker      TEXT,
  speaker_role TEXT,
  reply_to     INTEGER
);
CREATE INDEX IF NOT EXISTS idx_messages_role_time ON messages(role_id, created_at DESC);

-- v5：人物卡 / 信任账本 / 事实与声明。卡片是记忆的物化视图，不存聊天流水。
CREATE TABLE IF NOT EXISTS persons (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  role_id    TEXT NOT NULL,
  wxid       TEXT NOT NULL,
  name       TEXT NOT NULL DEFAULT '',
  kind       TEXT NOT NULL DEFAULT 'member'
             CHECK (kind IN ('creator','friend','group_member','agent','stranger')),
  first_seen REAL NOT NULL,
  last_seen  REAL NOT NULL,
  UNIQUE(role_id, wxid)
);
CREATE TABLE IF NOT EXISTS person_cards (
  person_id         INTEGER PRIMARY KEY,
  affinity          REAL NOT NULL DEFAULT 0.3,
  familiarity       REAL NOT NULL DEFAULT 0.1,
  trust_identity    REAL NOT NULL DEFAULT 0.5,
  trust_benevolence REAL NOT NULL DEFAULT 0.5,
  trust_competence  REAL NOT NULL DEFAULT 0.4,
  trust_disclosure  REAL NOT NULL DEFAULT 0.1,
  caution           REAL NOT NULL DEFAULT 0.3,
  attachment        REAL NOT NULL DEFAULT 0.0,
  authority         TEXT NOT NULL DEFAULT 'none',
  updated_at        REAL NOT NULL DEFAULT 0,
  version           INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS person_events (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  role_id           TEXT NOT NULL,
  person_id         INTEGER NOT NULL,
  event_type        TEXT NOT NULL,
  weight            REAL NOT NULL DEFAULT 0,
  content           TEXT NOT NULL DEFAULT '',
  source_message_id INTEGER,
  value_before      TEXT,
  value_after       TEXT,
  created_at        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_person_events_person ON person_events(person_id, created_at DESC);
CREATE TABLE IF NOT EXISTS person_facts (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  role_id           TEXT NOT NULL,
  person_id         INTEGER NOT NULL,
  content           TEXT NOT NULL,
  status            TEXT NOT NULL DEFAULT 'claim' CHECK (status IN ('fact','claim')),
  confidence        REAL NOT NULL DEFAULT 0.5,
  source_message_id INTEGER,
  created_at        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_person_facts_person ON person_facts(person_id, status);

-- 反思层审计：每次人格成长的证据、产出与应用结果
CREATE TABLE IF NOT EXISTS reflections (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  role_id    TEXT NOT NULL,
  created_at REAL NOT NULL,
  evidence   TEXT NOT NULL,
  result     TEXT NOT NULL,
  applied    TEXT NOT NULL
);

-- 手部（行动层）：用户要求的定时提醒
CREATE TABLE IF NOT EXISTS reminders (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  role_id           TEXT NOT NULL,
  text              TEXT NOT NULL,
  due_at            REAL NOT NULL,
  status            TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open','done','cancelled')),
  created_at        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reminders_due ON reminders(role_id, status, due_at);

-- 手部执行留痕：LLM 提议 -> 引擎执行 -> 结果，全部可审计
CREATE TABLE IF NOT EXISTS hands_log (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  role_id    TEXT NOT NULL,
  action     TEXT NOT NULL,
  params     TEXT NOT NULL,
  result     TEXT NOT NULL,
  created_at REAL NOT NULL
);

-- v6：审核 Agent（双 Agent 模式的第二个人）留痕。每条出门的话记原稿、
-- 病灶、改写要求与终稿；控制台能回溯"这句话是谁放行的"。
CREATE TABLE IF NOT EXISTS review_logs (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  role_id    TEXT NOT NULL,
  purpose    TEXT NOT NULL,              -- respond | proactive | chime | reminder
  mode       TEXT NOT NULL,              -- off | rule | llm
  issues     TEXT NOT NULL DEFAULT '[]', -- 病灶清单 JSON
  correction TEXT,                       -- 交给互动 Agent 的改写要求
  original   TEXT NOT NULL,              -- 互动 Agent 原稿
  final      TEXT NOT NULL,              -- 终稿（过审后真正发出去的）
  retried    INTEGER NOT NULL DEFAULT 0, -- 是否打回重写过
  llm_used   INTEGER NOT NULL DEFAULT 0,
  llm_failed INTEGER NOT NULL DEFAULT 0,
  latency_ms INTEGER NOT NULL DEFAULT 0,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_review_logs_role_time ON review_logs(role_id, created_at DESC);
