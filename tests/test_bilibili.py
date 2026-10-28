"""B 站手部验收：链接解析（含短链）、三路数据合成观后素材、评论区排序、
失败降级（弹幕挂了仍能"看"）。离线：本地假服务器 + 注入 fetch。"""
import json
import os
import sys
import threading
import zlib
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import make_tmpdir, rmtree, run_tests

from src import bilibili
from src.hands import Hands
from src.config import load_config
from src.storage import init_db

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VIEW = {"code": 0, "data": {
    "aid": 80433022, "cid": 137649199, "title": "【官方 MV】测试视频",
    "owner": {"name": "测试UP"}, "desc": "简介一句话", "duration": 213,
    "pubdate": 1580000000,
    "stat": {"view": 107535558, "danmaku": 151081, "like": 1200000,
             "coin": 800000, "reply": 99000}}}
REPLIES = {"code": 0, "data": {"replies": [
    {"content": {"message": "怀旧满分"}, "like": 500},
    {"content": {"message": "经典永流传\n换行也截断"}, "like": 1200},
    {"content": {"message": "泪目"}, "like": 80},
]}}
DAN_XML = ('<d><d p="1.0,1,25,16777215,0,0,0">你被骗了</d>'
           '<d p="2.0,1,25,16777215,0,0,0">你被骗了</d>'
           '<d p="3.0,1,25,16777215,0,0,0">恭喜破亿</d>'
           '<d p="4.0,1,25,16777215,0,0,0">欢迎回来</d></d>')


class FakeBili(BaseHTTPRequestHandler):
    danmaku_down = False

    def do_GET(self):
        if "b23.tv" in self.path or self.path.startswith("/short"):
            self.send_response(302)
            self.send_header("Location", "https://www.bilibili.com/video/BV1GJ411x7h7")
            self.end_headers()
            return
        if "web-interface/view" in self.path:
            body = json.dumps(VIEW).encode()
        elif "/x/v2/reply" in self.path:
            body = json.dumps(REPLIES).encode()
        elif self.path.endswith(".xml"):
            if self.danmaku_down:
                self.send_response(500)
                self.end_headers()
                return
            body = zlib.compress(DAN_XML.encode())[2:-4]  # raw deflate
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


_server = None


def _local(url):
    """把 bilibili 域名的 URL 改写为本地假服务器地址。"""
    global _server
    if _server is None:
        httpd = HTTPServer(("127.0.0.1", 0), FakeBili)
        _server = httpd
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
    from urllib.parse import urlsplit
    parts = urlsplit(url)
    query = f"?{parts.query}" if parts.query else ""
    return f"http://127.0.0.1:{_server.server_address[1]}{parts.path}{query}"


_ORIG_FETCH = bilibili.default_fetch


def _fetch(url, binary=False):
    return _ORIG_FETCH(_local(url), binary=binary)


def _short_fetch(url, binary=False):
    if "b23.tv" in url:
        return "", "https://www.bilibili.com/video/BV1GJ411x7h7?p=1"
    return _fetch(url, binary)


def test_parse_bvid():
    assert bilibili.parse_bvid("https://www.bilibili.com/video/BV1GJ411x7h7?p=1") == "BV1GJ411x7h7"
    assert bilibili.parse_bvid("BV1GJ411x7h7") == "BV1GJ411x7h7"
    assert bilibili.parse_bvid("看这个 https://b23.tv/abc123 谢谢", _short_fetch) == "BV1GJ411x7h7"
    try:
        bilibili.parse_bvid("https://www.bilibili.com/video/av170001")
        raise AssertionError("av 号应提示不支持")
    except ValueError:
        pass
    try:
        bilibili.parse_bvid("没有链接", _fetch)
        raise AssertionError("无 BV 应报错")
    except ValueError:
        pass


def test_watch_summary_composition():
    s = bilibili.watch_summary("https://www.bilibili.com/video/BV1GJ411x7h7", _fetch)
    assert "【官方 MV】测试视频" in s and "测试UP" in s
    assert "3分钟" in s
    assert "你被骗了（×2）" in s, "弹幕热词应聚合计数"
    assert "经典永流传" in s, "高赞评论应入素材"
    assert "107535558" in s and "151081" in s


def test_comments_summary_sorted_by_like():
    s = bilibili.comments_summary("BV1GJ411x7h7", _fetch)
    assert "经典永流传" in s and "赞1200" in s
    first = s.split("\n")[1]
    assert "经典永流传" in first, "按点赞排序，最高赞应排第一"


def test_danmaku_failure_degrades():
    FakeBili.danmaku_down = True
    try:
        s = bilibili.watch_summary("BV1GJ411x7h7", _fetch)
        assert "测试视频" in s and "弹幕读取失败" in s, "弹幕挂了也应给出基础信息+评论"
    finally:
        FakeBili.danmaku_down = False


def test_hands_actions_wired():
    tmp = make_tmpdir()
    try:
        conn = init_db(os.path.join(tmp, "agent.db"))
        cfg = load_config(os.path.join(ROOT, "config", "config.example.yaml"))
        hands = Hands(cfg, conn, "xiaoman")
        orig = bilibili.default_fetch
        bilibili.default_fetch = _fetch
        try:
            r = hands.execute("bilibili.watch",
                              {"url": "https://www.bilibili.com/video/BV1GJ411x7h7"})
            assert r["ok"] and "测试视频" in r["summary"]
            r2 = hands.execute("bilibili.comments", {"url": "BV1GJ411x7h7"})
            assert r2["ok"] and "经典永流传" in r2["summary"]
            r3 = hands.execute("bilibili.watch", {"url": "不是链接"})
            assert not r3["ok"]
            names = [m["name"] for m in hands.manifest()]
            assert "bilibili.watch" in names and "bilibili.comments" in names
        finally:
            bilibili.default_fetch = orig
    finally:
        rmtree(tmp)


if __name__ == "__main__":
    code = run_tests(globals(), "test_bilibili")
    if _server:
        _server.shutdown()
    sys.exit(code)
