"""手部（行动层）验收：时间解析、动作执行与安全门、感知提议接线、到点触发。"""
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import make_tmpdir, repo_config_dir, rmtree, run_tests

from src.config import load_config
from src.hands import Hands
from src.llm_client import MockLLMClient
from src.main import Agent
from src.storage import init_db
from src.timeparse import parse_when

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T0 = 1790000000.0  # 2026-09-22 周二（UTC）附近，具体以本地时区断言宽松处理


def _hands(tmp, **cfg_over):
    conn = init_db(os.path.join(tmp, "agent.db"))
    cfg = load_config(os.path.join(ROOT, "config", "config.example.yaml"))
    for k, v in cfg_over.items():
        setattr(cfg.hands, k, v)
    return Hands(cfg, conn, "xiaoman", memory=None, notifier=lambda t, b: True), conn


def test_timeparse_relative_and_absolute():
    assert parse_when("30分钟后", T0) == T0 + 1800
    assert parse_when("1.5小时后", T0) == T0 + 5400
    assert parse_when("两天后", T0) == T0 + 2 * 86400
    assert parse_when("十分钟后", T0) == T0 + 600
    # 绝对时间
    ts = parse_when("2030-01-02 09:00", T0)
    assert ts is not None and ts > T0
    # 只有时刻：今天已过则明天
    lt = time.localtime(T0)
    ts2 = parse_when("23:59", T0)
    assert ts2 is not None and ts2 > T0
    # 解析不了
    assert parse_when("下次轮回之时", T0) is None
    assert parse_when("", T0) is None


def test_timeparse_period_words():
    ts = parse_when("明天早上9点", T0)
    assert ts is not None and ts > T0
    lt = time.localtime(ts)
    assert lt.tm_hour == 9, "早上9点应为 9 点"
    ts2 = parse_when("明天晚上8点半", T0)
    lt2 = time.localtime(ts2)
    assert lt2.tm_hour == 20 and lt2.tm_min == 30, "晚上8点半 -> 20:30"
    ts3 = parse_when("下午3点", T0)
    assert time.localtime(ts3).tm_hour == 15


def test_reminder_set_list_cancel():
    tmp = make_tmpdir()
    try:
        hands, conn = _hands(tmp)
        r = hands.execute("reminder.set", {"text": "吃药", "when": "30分钟后"}, now=T0)
        assert r["ok"] and r["reminder_id"] == 1, r
        due = conn.execute("SELECT * FROM reminders").fetchone()
        assert due["text"] == "吃药" and abs(due["due_at"] - (T0 + 1800)) < 1
        # 时间听不懂
        r2 = hands.execute("reminder.set", {"text": "x", "when": "改天吧"}, now=T0)
        assert not r2["ok"] and "没听懂" in r2["summary"]
        # list / cancel
        rl = hands.execute("reminder.list", {}, now=T0)
        assert r2 is not None and "吃药" in rl["summary"]
        rc = hands.execute("reminder.cancel", {"text": "吃药"}, now=T0)
        assert rc["ok"]
        assert conn.execute("SELECT COUNT(*) c FROM reminders WHERE status='open'"
                            ).fetchone()["c"] == 0
        # 留痕
        assert conn.execute("SELECT COUNT(*) c FROM hands_log").fetchone()["c"] >= 4
    finally:
        rmtree(tmp)


def test_note_and_safety_gates():
    tmp = make_tmpdir()
    try:
        hands, conn = _hands(tmp)
        # shell 默认禁用
        r = hands.execute("shell.run", {"command": "echo hi"}, now=T0)
        assert not r["ok"] and "allow_shell" in r["summary"]
        # 非法 URL 协议被拒
        r2 = hands.execute("web.fetch", {"url": "ftp://x.com"}, now=T0)
        assert not r2["ok"] and "http" in r2["summary"]
        # 手部总开关
        hands.cfg.hands.enabled = False
        r3 = hands.execute("reminder.set", {"text": "x", "when": "1小时后"}, now=T0)
        assert not r3["ok"]
        hands.cfg.hands.enabled = True
        # manifest 随 allow_shell 联动
        names = [m["name"] for m in hands.manifest()]
        assert "shell.run" not in names
        hands.cfg.hands.allow_shell = True
        assert any(m["name"] == "shell.run" for m in hands.manifest())
    finally:
        rmtree(tmp)


def test_web_fetch_local():
    tmp = make_tmpdir()
    try:
        class Page(BaseHTTPRequestHandler):
            def do_GET(self):
                body = b"<html><script>bad()</script><p>hello " + b"x" * 100 + \
                       b"</p></html>"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        httpd = HTTPServer(("127.0.0.1", 0), Page)
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        hands, conn = _hands(tmp, web_max_bytes=50_000)
        r = hands.execute("web.fetch", {"url": f"http://127.0.0.1:{port}/"}, now=T0)
        httpd.shutdown()
        assert r["ok"], r
        assert "hello" in r["summary"] and "bad" not in r["summary"], "script 应被剥离"
    finally:
        rmtree(tmp)


def test_perception_proposes_and_executes():
    tmp = make_tmpdir()
    try:
        from tests.test_perception import FakeAnalysisClient
        raw = json.dumps({"sentiment": 0.1, "emotion": "平静", "intent_type": "闲聊",
                          "topics": ["提醒"], "urgency": 0.4, "cancel_topics": [],
                          "defer_topics": [], "done_topics": [],
                          "hand_action": {"name": "reminder.set",
                                          "params": {"text": "喝水", "when": "30分钟后"}}})
        agent = Agent(config_dir=os.path.join(ROOT, "config"),
                      db_path=os.path.join(tmp, "agent.db"),
                      llm_client=FakeAnalysisClient(raw))
        result = agent.on_user_message("半小时后提醒我喝水", now=T0)
        # 感知提议被引擎执行：提醒入库、结果进 meta 审计
        assert result["meta"]["hand"] and result["meta"]["hand"]["ok"]
        rows = agent.conn.execute("SELECT * FROM reminders").fetchall()
        assert len(rows) == 1 and rows[0]["text"] == "喝水"
        assert agent.conn.execute("SELECT COUNT(*) c FROM hands_log"
                                  ).fetchone()["c"] >= 1
    finally:
        rmtree(tmp)


def test_due_reminder_fires():
    tmp = make_tmpdir()
    try:
        agent = Agent(config_dir=os.path.join(ROOT, "config"),
                      db_path=os.path.join(tmp, "agent.db"))
        agent.hands.notifier = lambda t, b: True  # 测试不弹真通知
        agent.hands.execute("reminder.set", {"text": "收盘看一眼", "when": "1分钟后"},
                            now=T0 - 120)  # 已到点
        out = agent.evaluate_proactive(now=T0)
        assert out.get("sent") and out["sent"]["intent"] == "reminder_due", out
        assert "收盘看一眼" in out["sent"]["text"]
        # 留痕：active_logs（含 user_requested 标记）+ agent 消息（/pending 可见）
        log = agent.conn.execute(
            "SELECT score_detail, source_type FROM active_logs"
            " ORDER BY id DESC LIMIT 1").fetchone()
        assert log["source_type"] == "reminder"
        assert "user_requested" in log["score_detail"]
        assert agent.messages_after(0)[-1]["meta"]["proactive"]["intent"] == "reminder_due"
        # 已消费，不再重复发
        assert agent.evaluate_proactive(now=T0 + 60).get("fired_reminders") is None
    finally:
        rmtree(tmp)




def test_web_search_hand_with_fake_fetch():
    """web.search：结构化结果拼摘要；发布时间后缀解析进 date 字段；
    query 附当天日期锚点；HTTP 错误温和降级；query 缺失明确报。"""
    import json as _json
    import os as _os
    from src.websearch import SearchError, search

    cfg = load_config(os.path.join(repo_config_dir(), "config.example.yaml"))
    _os.environ[cfg.web_search.api_key_env] = "test-key"   # fetch 是假件，key 只需存在

    class FakeResp:
        def read(self):
            return _json.dumps({"choices": [{"message": {"tool_calls": [
                {"search_result": [
                    {"title": "TI2026 落地上海（发布时间：2026-10-23 20:08:27）",
                     "content": "Valve 宣布 2026 国际邀请赛在中国上海举办",
                     "link": "https://example.com/1"},
                    {"title": "第二条", "content": "内容二号", "link": "https://example.com/2"},
]}]}}]}).encode()

    seen = {}

    def fake_fetch(req, timeout):
        seen["body"] = req.data.decode("utf-8")
        return FakeResp().read()

    out = search("上海 major", cfg.web_search, fetch=fake_fetch)
    assert out and out[0]["title"] == "TI2026 落地上海"
    assert out[0]["date"] == "2026-10-23", "发布时间后缀必须解析成 date 字段"
    assert out[1]["date"] == "" and out[1]["title"] == "第二条"
    assert "今天是" in _json.loads(seen["body"])["messages"][0]["content"], \
        "发给引擎的 query 必须带当天日期锚点"

    def boom(req, timeout):
        raise SearchError("HTTP 429: 限流")

    try:
        search("x", cfg.web_search, fetch=boom)
        raise AssertionError("应抛 SearchError")
    except SearchError:
        pass

    _os.environ.pop(cfg.web_search.api_key_env, None)
    try:
        search("无 key", cfg.web_search, fetch=lambda r, t: b"{}")
        raise AssertionError("缺 key 应报错")
    except SearchError as e:
        assert cfg.web_search.api_key_env in str(e)


def test_echo_redraw_guard():
    """复读重画守卫：无关纠正+近似 prompt 跳过；带画图意图词或上次失败则放行。"""
    import json as _json
    import shutil
    from common import make_tmpdir
    from src.main import Agent
    tmp = make_tmpdir()
    try:
        agent = Agent(db_path=os.path.join(tmp, "a.db"),
                      llm_client=MockLLMClient())
        agent.conn.execute(
            "INSERT INTO hands_log(role_id, action, params, result, created_at) VALUES(?,?,?,?,?)",
            ("xiaoman", "image.draw",
             _json.dumps({"prompt": "写实摄影风格：清晨薄雾中的漓江，喀斯特峰林"}, ensure_ascii=False),
             _json.dumps({"ok": True}, ensure_ascii=False), 1000.0))
        agent.conn.commit()
        near = {"name": "image.draw",
                "params": {"prompt": "写实摄影风格：清晨薄雾中的漓江，喀斯特峰林倒影"}}
        assert agent._echo_redraw(near, "辈分错了"), "无关纠正+近似prompt 应拦截"
        assert not agent._echo_redraw(near, "再画一张，雾再重点"), "带画图意图词应放行"
        agent.conn.execute("UPDATE hands_log SET result=?", (_json.dumps({"ok": False}),))
        assert not agent._echo_redraw(near, "辈分错了"), "上次失败应放行重试"
        assert not agent._echo_redraw({"name": "web.search", "params": {"query": "x"}}, "abc"),             "非画图动作不管"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_publish_date_parsing_edges():
    """发布时间后缀：全/半角括号与冒号、无时刻、不在尾部不误剥。"""
    from src.websearch import split_publish_date
    t, d = split_publish_date("标题（发布时间：2026-10-23 20:08:27）")
    assert (t, d) == ("标题", "2026-10-23")
    t, d = split_publish_date("标题(发布时间:2026-10-23)")
    assert (t, d) == ("标题", "2026-10-23")
    t, d = split_publish_date("（发布时间：2026-10-23 08:00）")
    assert (t, d) == ("", "2026-10-23")
    t, d = split_publish_date("历史回顾（发布时间是2026年）十月见闻")
    assert d == "" and "十月见闻" in t, "非尾部/非标准格式不能误剥"
    assert split_publish_date("") == ("", "")


def test_web_search_summary_carries_dates():
    """手部摘要逐条带发布日期，模型才能对照当前时间判新旧。"""
    import sqlite3
    import src.websearch as ws_mod
    cfg = load_config(os.path.join(repo_config_dir(), "config.example.yaml"))
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE hands_log(id INTEGER PRIMARY KEY, role_id TEXT, action TEXT,"
                 " params TEXT, result TEXT, created_at REAL)")
    h = Hands(cfg, conn, "xiaoman")
    orig = ws_mod.search
    ws_mod.search = lambda q, c, fetch=None: [
        {"title": "TI2026 落地上海", "date": "2026-10-23", "content": "Valve 官宣",
         "link": "https://x/1"},
        {"title": "旧闻", "date": "", "content": "早年间的事", "link": "https://x/2"}]
    try:
        out = h.execute("web.search", {"query": "TI2026"}, now=1.0)
    finally:
        ws_mod.search = orig
    assert out["ok"], out["summary"]
    assert "（2026-10-23）" in out["summary"], "有日期的条目要带日期"
    assert "《旧闻》早年间的事" in out["summary"], "无日期条目不硬加空括号"



def test_web_search_chain_fallback():
    """双引擎链：bigmodel 空转 cush；cush 命中后摘要标注 provider。"""
    import sqlite3
    import src.websearch as ws_mod
    import src.cush_search as cush_mod
    cfg = load_config(os.path.join(repo_config_dir(), "config.example.yaml"))
    cfg.web_search.chain = ["bigmodel", "cush"]   # 显式声明双引擎，不依赖配置文件现值
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE hands_log(id INTEGER PRIMARY KEY, role_id TEXT, action TEXT,"
                 " params TEXT, result TEXT, created_at REAL)")
    h = Hands(cfg, conn, "xiaoman")
    calls = []

    def fake_bigmodel(q, c, fetch=None):
        calls.append("bigmodel")
        return []

    def fake_cush(q, exe, limit=4, timeout_s=50.0):
        calls.append("cush")
        return [{"title": "论文标题", "content": "摘要内容", "link": "https://x/1"}]

    orig_bm, orig_cush = ws_mod.search, cush_mod.search
    ws_mod.search, cush_mod.search = fake_bigmodel, fake_cush
    try:
        out = h.execute("web.search", {"query": "attention is all you need"}, now=1.0)
    finally:
        ws_mod.search, cush_mod.search = orig_bm, orig_cush
    assert out["ok"], out["summary"]
    assert calls == ["bigmodel", "cush"], "bigmodel 空必须落到 cush"
    assert "[cush]" in out["summary"] and "论文标题" in out["summary"]



def test_meme_hand_match_and_errors():
    """表情包手部：模板模糊匹配（摸头->petpet）、错误对象不炸（此 rust 绑定
    把错误当返回值而不是抛异常）。生成依赖已下载资源，CI 只测无资源路径。"""
    try:
        import meme_generator  # noqa: F401  可选依赖：未装环境（如 CI）跳过
    except ImportError:
        print("  [skip] meme_generator 未安装（可选依赖），跳过表情包测试")
        return
    import sqlite3
    cfg = load_config(os.path.join(repo_config_dir(), "config.example.yaml"))
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE hands_log(id INTEGER PRIMARY KEY, role_id TEXT, action TEXT,"
                 " params TEXT, result TEXT, created_at REAL)")
    h = Hands(cfg, conn, "xiaoman")
    assert h._meme_match("摸头") == "petpet"
    assert h._meme_match("petpet") == "petpet"
    assert h._meme_match("不存在xyz") == ""
    out = h.execute("meme.generate", {"template": "不存在xyz"}, now=1.0)
    assert not out["ok"] and "没有" in out["summary"]
    # 错误对象当返回值：不抛异常、分类成可读消息
    import meme_generator as mg
    class FakeErr:
        pass
    FakeErr.__name__ = "ImageAssetMissing"
    orig = mg.get_meme
    mg.get_meme = lambda k: type("M", (), {
        "generate": staticmethod(lambda **kw: FakeErr())})()
    try:
        out2 = h.execute("meme.generate", {"template": "摸头"}, now=1.0)
        assert not out2["ok"] and "下载" in out2["summary"], out2["summary"]
    finally:
        mg.get_meme = orig

if __name__ == "__main__":
    sys.exit(run_tests(globals(), "test_hands"))
