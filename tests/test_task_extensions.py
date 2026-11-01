"""任务/出站/触发/权限的扩展回归：P0 失败路径 + P1 异步 + P2 授权。"""
import json
import os
import sys
import threading
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import make_tmpdir, rmtree, run_tests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _hands(tmp):
    from src.config import load_config
    from src.hands import Hands
    from src.storage import init_db
    conn = init_db(os.path.join(tmp, "a.db"))
    cfg = load_config(os.path.join(ROOT, "config", "config.yaml"))
    return Hands(cfg, conn, "role"), conn


def test_orphans_and_stale_are_reported_and_failed():
    tmp = make_tmpdir()
    try:
        from src.taskstore import TaskStore
        from src.storage import init_db
        conn = init_db(os.path.join(tmp, "a.db"))
        ts = TaskStore(conn, "role", owner="bridge")
        a = ts.create("song.cover", {"title": "晴天"})
        ts.fail_orphans(before=time.time() + 1, owner="bridge")
        assert ts.last_orphans and ts.last_orphans[0]["id"] == a["id"]
        assert ts.get(a["id"])["status"] == "failed"

        b = ts.create("song.sing", {"title": "慢歌"})
        old = time.time() - 5000
        conn.execute("UPDATE tasks SET status='running', heartbeat_at=? WHERE id=?",
                     (old, b["id"]))
        conn.commit()
        stale = ts.stale(older_than=1200)
        assert [x["id"] for x in stale] == [b["id"]]
        failed = ts.fail_stale(older_than=1200)
        assert [x["id"] for x in failed] == [b["id"]]
        assert ts.get(b["id"])["status"] == "failed"
    finally:
        rmtree(tmp)


def test_media_outbox_drains_image():
    from src.taskstore import OutboxStore
    from src.storage import init_db
    from src.wechat_bridge import WeChatBridge
    tmp = make_tmpdir()
    try:
        conn = init_db(os.path.join(tmp, "a.db"))
        ob = OutboxStore(conn, "role")
        p = os.path.join(tmp, "x.jpg")
        with open(p, "wb") as f:
            f.write(b"jpg")
        item = ob.enqueue("room", "image", {"path": p}, dedupe_key="i")
        bridge = WeChatBridge.__new__(WeChatBridge)
        bridge.stop = threading.Event()
        bridge.log = lambda msg: None
        bridge.cfg = SimpleNamespace(wechat=SimpleNamespace(group_send_max_per_minute=6))
        bridge.pool = SimpleNamespace(agents={"room": SimpleNamespace(outbox=ob)})
        bridge._chat_enabled = lambda peer: True
        sent_paths = []
        bridge.channel = SimpleNamespace(
            send_image=lambda path, peer: (sent_paths.append(path), 0)[1])
        assert WeChatBridge._drain_outbox(bridge) == 1
        assert sent_paths == [p]
        assert ob.counts().get("sent") == 1
        assert item
    finally:
        rmtree(tmp)


def test_async_hand_returns_pending_and_marks_task_ready():
    tmp = make_tmpdir()
    try:
        hands, _ = _hands(tmp)
        done = threading.Event()
        seen = []

        def sender(info):
            seen.append(info)
            done.set()

        hands.async_sender = sender
        hands.register_action("demo.slow", {}, "test",
                              lambda p, now: {"ok": True, "summary": "做完了"})
        hands._async_actions.add("demo.slow")
        out = hands.execute("demo.slow", {"x": 1})
        assert out.get("pending") is True, out
        assert done.wait(5), "后台动作没有回调"
        assert seen[0]["result"]["ok"] is True
        task = hands.tasks.recent(1)[0]
        assert task["status"] == "ready", task
    finally:
        rmtree(tmp)


def test_bilibili_watch_is_async_when_sender_present():
    tmp = make_tmpdir()
    try:
        hands, _ = _hands(tmp)
        done = threading.Event()
        seen = []
        hands.async_sender = lambda info: (seen.append(info), done.set())
        hands._actions["bilibili.watch"]["handler"] = lambda p, now: {
            "ok": True, "summary": "视频讲的是测试内容"}
        hands.turn_context = {"is_owner": True, "requester": "owner",
                              "text": "看看这个视频"}
        out = hands.execute("bilibili.watch", {"url": "BV1test"})
        assert out.get("pending") is True, out
        assert done.wait(5), "bilibili.watch 没有异步回调"
        assert seen[0]["result"]["summary"] == "视频讲的是测试内容"
        task = hands.tasks.recent(1)[0]
        assert task["kind"] == "bilibili.watch" and task["status"] == "ready"
        params = json.loads(task["params"])
        assert params.get("_request") == "看看这个视频"
    finally:
        rmtree(tmp)


def test_shell_is_owner_only_and_task_status_cancel():
    tmp = make_tmpdir()
    try:
        hands, _ = _hands(tmp)
        hands.turn_context = {"is_owner": False, "requester": "u2"}
        r = hands.execute("shell.run", {"command": "echo hi"})
        assert r["ok"] is False and "创造者" in r["summary"]
        t = hands.tasks.create("song.cover", {"title": "晴天",
                                             "_meta": {"requester": "u1"}})
        hands.turn_context = {"is_owner": False, "requester": "u2"}
        out = hands.execute("task.cancel", {"kind": "song"})
        assert out["ok"] is False, out
        assert hands.tasks.get(t["id"])["status"] == "queued"
        hands.turn_context = {"is_owner": True, "requester": "owner"}
        st = hands.execute("task.status", {})
        assert "任务" in st["summary"]
        out2 = hands.execute("task.cancel", {"kind": "song"})
        assert out2["ok"] is True, out2
        assert hands.tasks.get(t["id"])["status"] == "cancelled"
    finally:
        rmtree(tmp)


def test_task_retry_replays_failed_song():
    tmp = make_tmpdir()
    try:
        hands, _ = _hands(tmp)
        failed = hands.tasks.create("song.cover", {"title": "晴天"})
        hands.tasks.update(failed["id"], status="failed", error="boom")
        calls = []
        old = hands._song_cover
        hands._song_cover = lambda p, now: (calls.append(p),
                                           {"ok": True, "summary": "重试中"})[1]
        hands.turn_context = {"is_owner": True, "requester": "owner"}
        try:
            out = hands.execute("task.retry", {"kind": "song"})
        finally:
            hands._song_cover = old
        assert out["ok"] is True and "重试" in out["summary"], out
        assert calls and calls[0]["title"] == "晴天", calls
    finally:
        rmtree(tmp)


def test_trigger_store_dedupes_rapid_redelivery():
    from src.taskstore import TriggerStore
    from src.storage import init_db
    tmp = make_tmpdir()
    try:
        conn = init_db(os.path.join(tmp, "a.db"))
        ts = TriggerStore(conn, "role")
        assert ts.log("room", "u1", "text", "respond", "accepted", True, "在吗") is True
        assert ts.log("room", "u1", "text", "respond", "accepted", True, "在吗") is False
        assert ts.recent(1)[0]["decision"] == "duplicate"
    finally:
        rmtree(tmp)


def test_health_flags_stale_running_task():
    from src.health import snapshot
    from src.storage import init_db
    tmp = make_tmpdir()
    try:
        os.makedirs(os.path.join(tmp, "health"))
        with open(os.path.join(tmp, "health", "bridge.json"), "w", encoding="utf-8") as f:
            json.dump({"at": time.time(), "agents": 1, "hook": True}, f)
        conn = init_db(os.path.join(tmp, "agent.db"))
        conn.execute("INSERT INTO tasks(id, role_id, kind, status, params, owner,"
                     " heartbeat_at, created_at, updated_at)"
                     " VALUES('t','role','song.sing','running','{}','bridge',?,?,?)",
                     (time.time() - 5000, time.time(), time.time()))
        conn.commit()
        snap = snapshot(data_dir=tmp, worker_url="http://127.0.0.1:1")
        checks = {c["name"]: c for c in snap["checks"]}
        assert checks["任务 / 出站队列"]["level"] == "error", checks
        assert "无心跳" in checks["任务 / 出站队列"]["detail"]
    finally:
        rmtree(tmp)


def test_agent_renders_task_result_with_persona():
    tmp = make_tmpdir()
    try:
        from src.main import Agent

        class FakeClient:
            name = "fake"
            last_usage = {"prompt_tokens": 0, "completion_tokens": 0,
                          "estimated": True}

            def __init__(self):
                self.prompts = []

            def render(self, prompt, seed):
                self.prompts.append(prompt)
                return "看完了，那个视频讲的是测试内容，还挺有意思"

            def render_raw(self, text, seed, **kw):
                return '{"pass": true, "issues": [], "fix": ""}'

        client = FakeClient()
        agent = Agent(config_dir=os.path.join(ROOT, "config"),
                      db_path=os.path.join(tmp, "a.db"), llm_client=client)
        agent.apply_config({"reviewer": {"mode": "off"}})
        task = agent.hands.tasks.create("bilibili.watch", {
            "_request": "看看这个视频", "_meta": {"requester": "owner"}})
        out = agent.on_task_result({
            "action": "bilibili.watch", "task_id": task["id"],
            "result": {"ok": True, "summary": "视频标题：测试；UP主：某人"}})
        assert out["reply"] and "测试内容" in out["reply"], out
        sections = client.prompts[-1]
        assert "后台任务结果" in sections, sections.keys()
        assert "看看这个视频" in str(sections["后台任务结果"])
        row = agent.conn.execute(
            "SELECT meta FROM messages WHERE sender='agent' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row and "task_result" in (row["meta"] or "")
    finally:
        rmtree(tmp)


def test_web_search_async_but_chime_can_force_sync():
    tmp = make_tmpdir()
    try:
        hands, _ = _hands(tmp)
        done = threading.Event()
        seen = []
        hands.async_sender = lambda info: (seen.append(info), done.set())
        hands._actions["web.search"]["handler"] = lambda p, now: {
            "ok": True, "summary": "搜到一条测试结果"}
        hands.turn_context = {"is_owner": True, "requester": "owner",
                              "text": "帮我搜一下"}
        out = hands.execute("web.search", {"query": "测试"})
        assert out.get("pending") is True, out
        assert done.wait(5), "web.search 没有异步回调"
        assert seen[0]["result"]["summary"] == "搜到一条测试结果"
        # 插嘴路径显式 _sync，必须当场拿结果，不能变成两条消息
        sync = hands.execute("web.search", {"query": "测试", "_sync": True})
        assert sync.get("pending") is not True and sync["summary"] == "搜到一条测试结果"
    finally:
        rmtree(tmp)


def test_bridge_sends_bilibili_result_through_agent():
    from src.wechat_bridge import WeChatBridge
    sent = []

    class A:
        def on_task_result(self, info):
            return {"reply": "看完了，讲的是测试内容"}

    b = WeChatBridge.__new__(WeChatBridge)
    b.log = lambda msg: None
    b.pool = SimpleNamespace(agents={"room": A()})
    b._send_reply = lambda text, peer: sent.append((text, peer))
    WeChatBridge._send_async_result(
        b, "room", {"action": "bilibili.watch",
                    "result": {"ok": True, "summary": "系统摘要"}})
    assert sent == [("看完了，讲的是测试内容", "room")], sent


if __name__ == "__main__":
    sys.exit(run_tests(globals(), "task_extensions"))
