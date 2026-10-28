"""把 SQLite 存量记忆回填进 mem0（只回填值得记的：显著度>=0.5 的事实与生命周期备注；
意图不回填——它们继续走 SQLite 的未完成意图机制）。用法：

  HF_ENDPOINT=https://hf-mirror.com AGENT_LLM_BASE_URL=... AGENT_LLM_MODEL=... \
  LLM_API_KEY=... python tools/backfill_mem0.py [--db data/agent.db]
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import sqlite3  # noqa: E402

from src.memory_mem0 import Mem0MemoryStore  # noqa: E402
from src.storage import init_db, project_root  # noqa: E402


def main():
    db = sys.argv[sys.argv.index("--db") + 1] if "--db" in sys.argv else \
        os.path.join(project_root(), "data", "agent.db")
    conn = init_db(db)
    rows = conn.execute(
        "SELECT id, role_id, content, salience, created_at FROM memories"
        " WHERE type='event' AND (salience>=0.5 OR content LIKE '「%')"
        " ORDER BY id").fetchall()
    print(f"待回填事实 {len(rows)} 条")
    store = Mem0MemoryStore(conn, data_dir=os.path.join(project_root(), "data"))
    ok = 0
    for r in rows:
        try:
            store.add(r["role_id"] or "xiaoman", "event", r["content"],
                      salience=r["salience"], now=r["created_at"] or 0.0,
                      verbatim=True)
            ok += 1
            print(f"  ✓ M{r['id']} {r['content'][:38]}")
        except Exception as e:  # noqa: BLE001
            print(f"  ✗ M{r['id']} {e!r}")
    print(f"回填完成 {ok}/{len(rows)}；验证：store.retrieve('晨跑 运动') ->")
    for m in store.retrieve("xiaoman", "晨跑 运动", 3, 0.0):
        print(f"  [M{m['id']}] {m['content'][:44]}")


if __name__ == "__main__":
    main()
