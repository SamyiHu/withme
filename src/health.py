"""运行健康快照：给控制台一个“哪里需要人”的页面。

不追求完整监控系统，只聚合最容易静默坏掉的几块：
- 桥进程心跳（data/health/bridge.json，桥每 15s 写一次）
- 微信 hook 登录态（桥心跳里带）
- 语音换色 worker（GET :8090/health）
- 各会话库里的任务 / 出站队列积压

健康检查本身永远不抛异常：某一块读不到就标 warn/error，不能让控制台跟着挂。
"""
import json
import os
import sqlite3
import time
import urllib.request

from .storage import project_root

BRIDGE_STALE_SECONDS = 45.0
TASK_STALE_SECONDS = 1200.0
WORKER_URL = "http://127.0.0.1:8090"


def _read_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def _worker_check(url: str = WORKER_URL) -> dict:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=1.5) as r:
            data = json.loads(r.read().decode("utf-8") or "{}")
        return {"ok": bool(data.get("ready")), "detail": "ready" if data.get("ready") else "未就绪"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "detail": "连接失败：%s" % str(e)[:120]}


def _db_snapshot(path: str, label: str) -> dict:
    """只读聚合一个会话库；表不存在（老库）就返回空。"""
    out = {"label": label, "path": path, "tasks": {}, "outbox": {},
           "triggers": {}, "recent_tasks": [], "recent_triggers": [],
           "failed_outbox": [], "oldest_active_age": 0.0}
    if not os.path.exists(path):
        return out
    conn = None
    try:
        conn = sqlite3.connect(path, timeout=1.0)
        conn.row_factory = sqlite3.Row
        names = {r["name"] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if "tasks" in names:
            for r in conn.execute(
                    "SELECT status, COUNT(*) c FROM tasks GROUP BY status").fetchall():
                out["tasks"][r["status"]] = int(r["c"])
            out["recent_tasks"] = [dict(r) for r in conn.execute(
                "SELECT id, kind, stage, status, error, created_at, updated_at"
                " FROM tasks ORDER BY created_at DESC LIMIT 8").fetchall()]
            oldest = conn.execute(
                "SELECT MIN(heartbeat_at) t FROM tasks"
                " WHERE status IN ('queued','running')").fetchone()
            if oldest and oldest["t"]:
                out["oldest_active_age"] = max(0.0, time.time() - float(oldest["t"]))
        if "outbox" in names:
            for r in conn.execute(
                    "SELECT status, COUNT(*) c FROM outbox GROUP BY status").fetchall():
                out["outbox"][r["status"]] = int(r["c"])
            out["failed_outbox"] = [dict(r) for r in conn.execute(
                "SELECT id, room, kind, attempts, last_error, updated_at"
                " FROM outbox WHERE status='failed' ORDER BY id DESC LIMIT 8").fetchall()]
        if "triggers" in names:
            for r in conn.execute(
                    "SELECT decision, COUNT(*) c FROM triggers GROUP BY decision").fetchall():
                out["triggers"][r["decision"]] = int(r["c"])
            out["recent_triggers"] = [dict(r) for r in conn.execute(
                "SELECT id, room, sender, decision, reason, addressed, preview, created_at"
                " FROM triggers ORDER BY id DESC LIMIT 8").fetchall()]
    except Exception as e:  # noqa: BLE001  单个库坏了不拖垮整体
        out["error"] = repr(e)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
    return out


def _db_paths(data_dir: str) -> list:
    paths = []
    default = os.path.join(data_dir, "agent.db")
    if os.path.exists(default):
        paths.append((default, "主库"))
    users = os.path.join(data_dir, "wechat_users")
    if os.path.isdir(users):
        for name in sorted(os.listdir(users)):
            path = os.path.join(users, name, "agent.db")
            if os.path.exists(path):
                paths.append((path, name))
    return paths


def snapshot(data_dir: str = None, worker_url: str = WORKER_URL) -> dict:
    now = time.time()
    data_dir = data_dir or os.path.join(project_root(), "data")
    bridge = _read_json(os.path.join(data_dir, "health", "bridge.json"))
    age = (now - float(bridge.get("at") or 0)) if bridge else None
    if not bridge:
        bridge_check = {"name": "微信桥", "level": "error",
                        "detail": "没有心跳文件（桥没启动过/从未写入）"}
    elif age is not None and age > BRIDGE_STALE_SECONDS:
        bridge_check = {"name": "微信桥", "level": "error",
                        "detail": "心跳已 %.0f 秒未更新" % age}
    else:
        bridge_check = {"name": "微信桥", "level": "ok",
                        "detail": "%d 个会话，心跳 %.0f 秒前" % (
                            int(bridge.get("agents") or 0), age or 0)}
    hook = bridge.get("hook")
    if hook is True:
        hook_check = {"name": "微信 hook", "level": "ok", "detail": "已登录"}
    elif hook is False:
        hook_check = {"name": "微信 hook", "level": "error", "detail": "未登录"}
    else:
        hook_check = {"name": "微信 hook", "level": "warn", "detail": "心跳里没有登录态"}

    worker = _worker_check(worker_url)
    worker_check = {"name": "语音 worker", "level": "ok" if worker["ok"] else "warn",
                    "detail": worker["detail"]}

    dbs = [_db_snapshot(path, label) for path, label in _db_paths(data_dir)]
    task_counts, outbox_counts = {}, {}
    trigger_counts = {}
    recent_tasks, failed_outbox, recent_triggers = [], [], []
    oldest_active_age = 0.0
    for db in dbs:
        oldest_active_age = max(oldest_active_age,
                                float(db.get("oldest_active_age") or 0))
        for k, v in (db.get("tasks") or {}).items():
            task_counts[k] = task_counts.get(k, 0) + int(v)
        for k, v in (db.get("outbox") or {}).items():
            outbox_counts[k] = outbox_counts.get(k, 0) + int(v)
        for k, v in (db.get("triggers") or {}).items():
            trigger_counts[k] = trigger_counts.get(k, 0) + int(v)
        for row in db.get("recent_triggers") or []:
            row = dict(row)
            row["db"] = db["label"]
            recent_triggers.append(row)
        for row in db.get("recent_tasks") or []:
            row = dict(row)
            row["db"] = db["label"]
            recent_tasks.append(row)
        for row in db.get("failed_outbox") or []:
            row = dict(row)
            row["db"] = db["label"]
            failed_outbox.append(row)
    recent_tasks.sort(key=lambda r: float(r.get("created_at") or 0), reverse=True)
    failed_outbox.sort(key=lambda r: float(r.get("updated_at") or 0), reverse=True)
    recent_triggers.sort(key=lambda r: float(r.get("created_at") or 0), reverse=True)

    running = int(task_counts.get("running") or 0)
    failed = int(task_counts.get("failed") or 0)
    pending = int(outbox_counts.get("pending") or 0)
    ofailed = int(outbox_counts.get("failed") or 0)
    if oldest_active_age > TASK_STALE_SECONDS:
        task_level = "error"
        task_detail = "有任务 %.0f 秒无心跳（疑似卡死）" % oldest_active_age
    elif failed or ofailed:
        task_level = "warn"
        task_detail = "失败任务 %d，失败出站 %d" % (failed, ofailed)
    elif running or pending:
        task_level = "warn"
        task_detail = "运行中任务 %d，待补发 %d" % (running, pending)
    else:
        task_level = "ok"
        task_detail = "无积压"
    task_check = {"name": "任务 / 出站队列", "level": task_level, "detail": task_detail}

    checks = [bridge_check, hook_check, worker_check, task_check]
    worst = "ok"
    for c in checks:
        if c["level"] == "error":
            worst = "error"
            break
        if c["level"] == "warn":
            worst = "warn"
    return {
        "ok": worst == "ok",
        "level": worst,
        "at": now,
        "checks": checks,
        "bridge": bridge,
        "tasks": {"counts": task_counts, "recent": recent_tasks[:20],
                  "oldest_active_age": round(oldest_active_age, 1)},
        "outbox": {"counts": outbox_counts, "failed": failed_outbox[:20]},
        "triggers": {"counts": trigger_counts, "recent": recent_triggers[:20]},
        "dbs": dbs,
    }
