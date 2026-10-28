"""API 标准化验收：/api/v1 前缀等价、JSON 契约（坏 body 400、坏查询参数 400）、
错误形状统一、/health 版本握手、版本化路径的鉴权。"""
import json
import os
import shutil
import sys
import tempfile
import threading
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import run_tests

from src.main import Agent
from src.server import make_server

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
        self.thread.join(timeout=5)  # 等 serve_forever 线程退出，避免端口/连接残留串味


def _req(method, url, body=None, headers=None):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8") if body is not None else None,
        headers={"Content-Type": "application/json", **(headers or {})}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def _raw(method, url, data: bytes):
    """发原始字节 body（构造坏 JSON 用），返回 (status, 解析后的 json 或 None)。"""
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return e.code, None


def test_api_v1_prefix_equivalence():
    """版本前缀是纯别名：/api/v1/X 与 /X 响应逐字节同构。"""
    tmp = tempfile.mkdtemp()
    srv = _Server(tmp)
    try:
        for bare in ("/health", "/status", "/messages?limit=3", "/pending?after_id=0"):
            _, a = _req("GET", srv.url(bare))
            _, b = _req("GET", srv.url("/api/v1" + bare))
            _, c = _req("GET", srv.url("/api" + bare))
            assert a == b == c, bare
        # 版本前缀下的写操作
        code, out = _req("POST", srv.url("/api/v1/proactive"), body={"enabled": True})
        assert code == 200 and out["proactive_enabled"] is True
        code, out = _req("POST", srv.url("/api/v1/level"), body={"level": "L2", "enabled": True})
        assert code == 200 and out == {"level": "L2", "enabled": True}
    finally:
        srv.close()
        shutil.rmtree(tmp, ignore_errors=True)


def test_health_version_handshake():
    tmp = tempfile.mkdtemp()
    srv = _Server(tmp)
    try:
        code, h = _req("GET", srv.url("/health"))
        assert code == 200 and h["ok"] and h["api"] == "v1"
    finally:
        srv.close()
        shutil.rmtree(tmp, ignore_errors=True)


def test_invalid_json_body_400():
    """坏 JSON / 非 dict body 一律 400 invalid json body，不再静默变 {}。"""
    tmp = tempfile.mkdtemp()
    srv = _Server(tmp)
    try:
        for raw in (b"{bad json", b"[1,2]", b'"text"', b"\xff\xfe"):
            code, err = _raw("POST", srv.url("/proactive"), raw)
            assert code == 400 and err["error"] == "invalid json body", (raw, code, err)
        # 空 body = {} 等价无参调用：/tick 不需要参数，应正常 200
        code, out = _raw("POST", srv.url("/api/v1/tick"), b"")
        assert code == 200 and isinstance(out, dict)
    finally:
        srv.close()
        shutil.rmtree(tmp, ignore_errors=True)


def test_bad_query_param_400():
    """limit / after_id 传非整数回 400（此前会漏进 500）。"""
    tmp = tempfile.mkdtemp()
    srv = _Server(tmp)
    try:
        for url in ("/messages?limit=abc", "/pending?after_id=x",
                    "/api/v1/trajectory?limit=1.5"):
            code, err = _req("GET", srv.url(url))
            assert code == 400 and "整数" in err["error"], (url, code, err)
    finally:
        srv.close()
        shutil.rmtree(tmp, ignore_errors=True)


def test_error_shapes_uniform():
    """错误统一 {"error": …} 形状；参数校验错误信息可读。"""
    tmp = tempfile.mkdtemp()
    srv = _Server(tmp)
    try:
        code, err = _req("GET", srv.url("/no-such-path"))
        assert code == 404 and err == {"error": "not found"}
        code, err = _req("GET", srv.url("/api/v1/no-such-path"))
        assert code == 404 and err == {"error": "not found"}
        code, err = _req("POST", srv.url("/proactive"), body={"enabled": "yes"})
        assert code == 400 and err["error"] == "enabled (bool) required"
        code, err = _req("POST", srv.url("/level"), body={"level": "L9", "enabled": True})
        assert code == 400 and "L1-L3" in err["error"]
    finally:
        srv.close()
        shutil.rmtree(tmp, ignore_errors=True)


def test_auth_on_versioned_paths():
    """鉴权对版本化路径同样生效。"""
    tmp = tempfile.mkdtemp()
    os.environ["AGENT_API_TOKEN"] = "secret-token"
    srv = _Server(tmp)
    try:
        code, _ = _req("GET", srv.url("/api/v1/status"))
        assert code == 401, f"版本化路径无令牌应 401，实际 {code}"
        code, _ = _req("GET", srv.url("/api/v1/status"),
                       headers={"X-API-Token": "secret-token"})
        assert code == 200
        # 设了令牌后 500 只回通用信息（本测试只验证字段形状，不触发真 500）
        code, h = _req("GET", srv.url("/health"))
        assert code == 200 and h["ok"]
    finally:
        srv.close()
        shutil.rmtree(tmp, ignore_errors=True)
        os.environ.pop("AGENT_API_TOKEN", None)


if __name__ == "__main__":
    sys.exit(run_tests(globals(), "test_server_api"))
