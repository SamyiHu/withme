"""I/O 面验收：HTTP API 全链路——聊天、状态、增量轮询、主动消息、认证、开关。"""
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import run_tests

from src.main import Agent
from src.server import make_server

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _at(hour: int, minute: int = 0) -> float:
    lt = time.localtime()
    return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, hour, minute, 0, 0, 0, -1))


class _Server:
    def __init__(self, tmp):
        self.tmp = tmp
        self.agent = Agent(config_dir=os.path.join(ROOT, "config"),
                           db_path=os.path.join(tmp, "agent.db"), threaded=True)
        self.httpd = make_server(self.agent, "127.0.0.1", 0)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def url(self, path):
        return f"http://127.0.0.1:{self.port}{path}"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def _req(method, url, body=None, headers=None):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8") if body is not None else None,
        headers={"Content-Type": "application/json", **(headers or {})}, method=method)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def test_health_and_page():
    tmp = tempfile.mkdtemp()
    srv = _Server(tmp)
    try:
        code, health = _req("GET", srv.url("/health"))
        assert code == 200 and health["ok"] and health["role_id"] == "xiaoman"
        with urllib.request.urlopen(srv.url("/"), timeout=10) as resp:
            page = resp.read().decode("utf-8")
        assert "<html" in page
        # 网页客户端（src/web/index.html）不入库：有则含角色名，无则回落到 API 导览页
        import os as _os
        _web = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                             "src", "web", "index.html")
        if _os.path.exists(_web):
            assert "小满" in page
        else:
            assert "WithMe" in page and "/health" in page
        # 后台控制台：页面 + 参数 JSON
        with urllib.request.urlopen(srv.url("/console"), timeout=10) as resp:
            cpage = resp.read().decode("utf-8")
        assert "后台控制台" in cpage
        code, cdata = _req("GET", srv.url("/console-data"))
        assert code == 200 and "config" in cdata and "status" in cdata
        assert "people" in cdata and "memory" in cdata
        code, traj = _req("GET", srv.url("/trajectory?limit=20"))
        assert code == 200 and isinstance(traj.get("items"), list)
        # 未知路由 404
        try:
            _req("GET", srv.url("/nope"))
            raise AssertionError("应 404")
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        srv.close()
        shutil.rmtree(tmp, ignore_errors=True)


def test_chat_status_messages_pending():
    tmp = tempfile.mkdtemp()
    srv = _Server(tmp)
    try:
        code, r = _req("POST", srv.url("/chat"), {"text": "我打算周末去学冲浪"})
        assert code == 200 and r["reply"]
        assert isinstance(r["meta"]["message_id"], int)
        code, s = _req("GET", srv.url("/status"))
        assert code == 200 and s["state_vector"]["energy"] > 0
        assert s["proactive_enabled"] is True and s["permission_level"].startswith("L")
        code, m = _req("GET", srv.url("/messages?limit=10"))
        assert code == 200 and len(m["items"]) >= 2
        assert m["items"][-1]["sender"] == "agent"
        # 控制台参数可调：合法值 200，非法值 400
        code, c = _req("POST", srv.url("/config"),
                        {"tick": {"fast_seconds": 30},
                         "gate": {"thresholds": {"L2": 0.4}}})
        assert code == 200 and c.get("llm")
        try:
            _req("POST", srv.url("/config"), {"tick": {"fast_seconds": 99999}})
            raise AssertionError("非法 tick 应 400")
        except urllib.error.HTTPError as e:
            assert e.code == 400
        # 空文本 400
        try:
            _req("POST", srv.url("/chat"), {"text": "  "})
            raise AssertionError("空文本应 400")
        except urllib.error.HTTPError as e:
            assert e.code == 400
    finally:
        srv.close()
        shutil.rmtree(tmp, ignore_errors=True)


def test_proactive_appears_in_pending():
    tmp = tempfile.mkdtemp()
    srv = _Server(tmp)
    try:
        # 受控时间喂消息（/chat 的 HTTP 链路由其它用例覆盖），HTTP 侧只验证增量轮询面
        srv.agent.on_user_message("我打算周末去学冲浪", now=_at(9))
        last = srv.agent.messages_after(0)
        after = last[-1]["id"] if last else 0
        result = srv.agent.evaluate_proactive(now=_at(17))
        assert result.get("sent"), f"前置条件：主动应发出 {result}"
        code, pend = _req("GET", srv.url(f"/pending?after_id={after}"))
        assert code == 200 and pend["items"], "主动消息必须出现在增量轮询里"
        m = pend["items"][0]
        assert m["sender"] == "agent", m
        assert m["meta"]["proactive"]["level"] == 2, m["meta"]
        assert m["meta"]["proactive"]["log_id"] == result["sent"]["log_id"], "可溯源到 active_logs"
        # 游标推进后不再重复下发
        code, pend2 = _req("GET", srv.url(f"/pending?after_id={pend['last_id']}"))
        assert pend2["items"] == [], "游标之后不应再有消息"
    finally:
        srv.close()
        shutil.rmtree(tmp, ignore_errors=True)


def test_auth_token_enforced():
    tmp = tempfile.mkdtemp()
    os.environ["AGENT_API_TOKEN"] = "secret-token"
    srv = _Server(tmp)
    try:
        try:
            _req("GET", srv.url("/status"))
            raise AssertionError("无令牌应 401")
        except urllib.error.HTTPError as e:
            assert e.code == 401
        code, s = _req("GET", srv.url("/status"),
                       headers={"Authorization": "Bearer secret-token"})
        assert code == 200 and s["role_id"]
        # /health 与网页保持开放（静态壳，数据仍需令牌）
        code, h = _req("GET", srv.url("/health"))
        assert code == 200
    finally:
        srv.close()
        shutil.rmtree(tmp, ignore_errors=True)
        os.environ.pop("AGENT_API_TOKEN", None)


def test_tick_and_switches():
    tmp = tempfile.mkdtemp()
    srv = _Server(tmp)
    try:
        code, r = _req("POST", srv.url("/tick"))
        assert code == 200 and ("sent" in r or "reason" in r)
        # 关闭总开关 -> 调度直通拒绝，状态可见
        code, _ = _req("POST", srv.url("/proactive"), {"enabled": False})
        assert code == 200
        code, s = _req("GET", srv.url("/status"))
        assert s["proactive_enabled"] is False
        r2 = srv.agent.evaluate_proactive()
        assert r2.get("sent") is None and "disabled" in r2.get("reason", "")
        _req("POST", srv.url("/proactive"), {"enabled": True})
        # 分级开关
        code, _ = _req("POST", srv.url("/level"), {"level": "L2", "enabled": False})
        assert code == 200
        s2 = srv.agent.status()
        assert "L2" not in s2["permission_level"] or s2["permission_level"] != "L2"
        _req("POST", srv.url("/level"), {"level": "L2", "enabled": True})
        # 非法入参 400
        try:
            _req("POST", srv.url("/level"), {"level": "L9", "enabled": True})
            raise AssertionError("非法 level 应 400")
        except urllib.error.HTTPError as e:
            assert e.code == 400
    finally:
        srv.close()
        shutil.rmtree(tmp, ignore_errors=True)



def test_presence_endpoint():
    """POST /presence 写入 typing 时间戳（回归：曾因缺失 set_setting 导入返回 500）。"""
    tmp = tempfile.mkdtemp()
    srv = _Server(tmp)
    try:
        code, r = _req("POST", srv.url("/presence"), {"typing": True})
        assert code == 200 and r.get("ok") is True, r
        row = srv.agent.conn.execute(
            "SELECT value FROM settings WHERE key='last_typing_ts'").fetchone()
        assert row is not None
    finally:
        srv.close()
        shutil.rmtree(tmp, ignore_errors=True)


def test_concurrent_requests_serialized():
    tmp = tempfile.mkdtemp()
    srv = _Server(tmp)
    try:
        results = []

        def worker(i):
            try:
                code, r = _req("POST", srv.url("/chat"), {"text": f"并发消息 {i} 测试一下"})
                results.append(code == 200 and bool(r["reply"]))
            except Exception:  # noqa: BLE001
                results.append(False)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(results) and len(results) == 6, "并发请求必须全部成功（锁串行化）"
    finally:
        srv.close()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(run_tests(globals(), "test_server_io"))
