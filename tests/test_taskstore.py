"""任务/出站队列与健康快照的回归测试。"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import make_tmpdir, rmtree, run_tests

from src.storage import init_db
from src.taskstore import OutboxStore, TaskStore


def test_task_lifecycle_and_idempotency():
    tmp = make_tmpdir()
    try:
        conn = init_db(os.path.join(tmp, "a.db"))
        ts = TaskStore(conn, "role", owner="bridge")
        t1 = ts.create("song.cover", {"title": "晴天"}, idempotency_key="晴天")
        t2 = ts.create("song.cover", {"title": "晴天"}, idempotency_key="晴天")
        assert t1["id"] == t2["id"], "同一活跃任务必须复用"
        ts.update(t1["id"], status="running", stage="converting")
        assert ts.get(t1["id"])["stage"] == "converting"
        ts.update(t1["id"], status="ready", result={"path": "x.mp3"})
        assert json.loads(ts.get(t1["id"])["result"])["path"] == "x.mp3"
        assert ts.counts().get("ready") == 1
        # ready 之后同一 key 可以重新建（不是活跃状态）
        t3 = ts.create("song.cover", {"title": "晴天"}, idempotency_key="晴天")
        assert t3["id"] != t1["id"]
    finally:
        rmtree(tmp)


def test_fail_orphans_only_own_owner():
    tmp = make_tmpdir()
    try:
        conn = init_db(os.path.join(tmp, "a.db"))
        bridge = TaskStore(conn, "role", owner="bridge")
        serve = TaskStore(conn, "role", owner="serve")
        a = bridge.create("song.sing")
        b = serve.create("song.sing")
        n = bridge.fail_orphans(before=time.time() + 1, owner="bridge")
        assert n == 1
        assert bridge.get(a["id"])["status"] == "failed"
        assert serve.get(b["id"])["status"] == "queued", "不能误杀 serve 的任务"
    finally:
        rmtree(tmp)


def test_outbox_retry_and_dedupe():
    tmp = make_tmpdir()
    try:
        conn = init_db(os.path.join(tmp, "a.db"))
        ob = OutboxStore(conn, "role")
        i1 = ob.enqueue("room", "text", {"text": "hi"}, dedupe_key="k")
        i2 = ob.enqueue("room", "text", {"text": "hi"}, dedupe_key="k")
        assert i1 == i2
        due = ob.due()
        assert len(due) == 1
        assert ob.retry(i1, "boom", delay=0) == "pending"
        assert ob.due()[0]["attempts"] == 1
        ob.mark_sent(i1)
        assert not ob.due()
        assert ob.counts().get("sent") == 1
    finally:
        rmtree(tmp)


def test_health_snapshot_flags_stale_bridge_and_tasks():
    from src.health import snapshot
    tmp = make_tmpdir()
    try:
        os.makedirs(os.path.join(tmp, "health"))
        with open(os.path.join(tmp, "health", "bridge.json"), "w", encoding="utf-8") as f:
            json.dump({"at": time.time() - 600, "agents": 2, "hook": False}, f)
        conn = init_db(os.path.join(tmp, "agent.db"))
        ts = TaskStore(conn, "role", owner="bridge")
        t = ts.create("song.cover")
        ts.update(t["id"], status="failed", error="boom")
        snap = snapshot(data_dir=tmp, worker_url="http://127.0.0.1:1")
        levels = {c["name"]: c["level"] for c in snap["checks"]}
        assert levels["微信桥"] == "error", levels
        assert levels["微信 hook"] == "error", levels
        assert snap["tasks"]["counts"].get("failed") == 1
        assert snap["level"] in ("warn", "error")
    finally:
        rmtree(tmp)


def test_bridge_text_failure_goes_to_outbox_and_drains():
    """发送三次失败 -> 落 outbox；下一轮补发成功 -> sent。"""
    import threading
    from types import SimpleNamespace
    from src.wechat_bridge import WeChatBridge
    tmp = make_tmpdir()
    try:
        conn = init_db(os.path.join(tmp, "agent.db"))
        ob = OutboxStore(conn, "role")
        bridge = WeChatBridge.__new__(WeChatBridge)
        bridge.stop = threading.Event()
        bridge.log = lambda msg: None
        bridge.cfg = SimpleNamespace(wechat=SimpleNamespace(group_send_max_per_minute=6))
        bridge.pool = SimpleNamespace(agents={"room": SimpleNamespace(outbox=ob)})
        bridge._send_log, bridge._last_send = {}, {}
        bridge._send_text_raw = lambda text, peer: False
        bridge._send_text("掉线时这条不能丢", "room")
        due = ob.due()
        assert len(due) == 1, due
        bridge._chat_enabled = lambda peer: True
        bridge._send_text_raw = lambda text, peer: True
        assert bridge._drain_outbox() == 1
        assert not ob.due()
        assert ob.counts().get("sent") == 1
    finally:
        rmtree(tmp)


def test_recover_unanswered_skips_any_agent_reply_after():
    """重启补处理只补真正没人接的消息；插嘴/主动也算接住，避免补出第二条。"""
    import time
    from types import SimpleNamespace
    from src.wechat_bridge import WeChatBridge
    tmp = make_tmpdir()
    try:
        conn = init_db(os.path.join(tmp, "agent.db"))
        now = time.time()
        conn.execute("INSERT INTO messages(role_id,sender,content,created_at)"
                     " VALUES('role','user','已答过',?)", (now - 100,))
        conn.execute("INSERT INTO messages(role_id,sender,content,created_at)"
                     " VALUES('role','agent','插嘴也算答',?)", (now - 90,))
        conn.execute("INSERT INTO messages(role_id,sender,content,created_at)"
                     " VALUES('role','user','真正漏掉的',?)", (now - 80,))
        conn.commit()
        bridge = WeChatBridge.__new__(WeChatBridge)
        bridge.log = lambda msg: None
        bridge._started_at = now
        bridge.pool = SimpleNamespace(agents={"room": SimpleNamespace(
            role_id="role", conn=conn)})
        bridge._chat_enabled = lambda peer: True
        calls = []
        bridge._multi_chat = lambda text, *a, **k: calls.append((text, k))
        n = WeChatBridge._recover_unanswered(bridge)
        assert n == 1, calls
        assert calls[0][0] == "真正漏掉的", calls
        assert calls[0][1].get("replay_message_id") == 3
    finally:
        rmtree(tmp)


if __name__ == "__main__":
    sys.exit(run_tests(globals(), "taskstore"))
