"""插件系统验收：发现/装载/执行/隔离/容错/禁用，全部离线。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import make_tmpdir, rmtree, run_tests

from src.config import load_config
from src.hands import Hands
from src.plugins import load_plugins
from src.storage import get_setting, init_db

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GOOD = '''
DESC = "测试插件"
ACTIONS = [{
    "name": "demo.ping",
    "desc": "测试用",
    "params": {"msg": "消息"},
    "handler": lambda params, ctx: (
        ctx.kv_set("last", params.get("msg", "")),
        {"ok": True, "summary": "pong:" + str(params.get("msg", ""))})[1],
}]
'''

BROKEN = "this is not ( valid python"
NO_ACTIONS = "X = 1\n"
BAD_NAME = '''
ACTIONS = [{"name": "BadName", "desc": "x", "params": {},
            "handler": lambda p, c: {"ok": True, "summary": "x"}}]
'''


def _hands(tmp):
    conn = init_db(os.path.join(tmp, "agent.db"))
    cfg = load_config(os.path.join(ROOT, "config", "config.example.yaml"))
    cfg.hands.notify_desktop = False
    return Hands(cfg, conn, "xiaoman", notifier=lambda t, b: True), conn


def _pdir(tmp, files: dict) -> str:
    d = os.path.join(tmp, "plugins")
    os.makedirs(d, exist_ok=True)
    for name, content in files.items():
        with open(os.path.join(d, name), "w", encoding="utf-8") as f:
            f.write(content)
    return d


def test_plugin_load_and_execute():
    tmp = make_tmpdir()
    try:
        hands, conn = _hands(tmp)
        info = load_plugins(hands, _pdir(tmp, {"good.py": GOOD, "_skipme.py": GOOD}))
        assert any(p["plugin"] == "good" and p["status"] == "ok" and p["actions"] == 1
                   for p in info)
        assert not any(p["plugin"] == "_skipme" for p in info), "_ 开头文件不加载"
        # 进清单，感知层可见
        names = [m["name"] for m in hands.manifest()]
        assert "demo.ping" in names and "reminder.set" in names
        # 执行 + ctx KV 持久化 + 审计
        r = hands.execute("demo.ping", {"msg": "哈喽"}, now=1000.0)
        assert r["ok"] and r["summary"] == "pong:哈喽"
        assert get_setting(conn, "plugin:good:last") == "哈喽"
        assert json.loads(conn.execute(
            "SELECT params FROM hands_log ORDER BY id DESC LIMIT 1"
        ).fetchone()["params"])["msg"] == "哈喽"
        # 插件来源可查
        assert hands._actions["demo.ping"]["source"] == "plugin:good"
    finally:
        rmtree(tmp)


def test_broken_plugins_do_not_crash():
    tmp = make_tmpdir()
    try:
        hands, conn = _hands(tmp)
        info = load_plugins(hands, _pdir(tmp, {
            "broken.py": BROKEN, "noactions.py": NO_ACTIONS, "badname.py": BAD_NAME}))
        by = {p["plugin"]: p["status"] for p in info}
        assert by["broken"] == "error" and by["noactions"] == "error"
        assert by["badname"] == "error", "非法动作名应整体拒绝"
        # 内置动作不受影响
        r = hands.execute("reminder.set", {"text": "x", "when": "1小时后"}, now=1000.0)
        assert r["ok"]
    finally:
        rmtree(tmp)


def test_disabled_plugin_skipped():
    tmp = make_tmpdir()
    try:
        hands, conn = _hands(tmp)
        d = _pdir(tmp, {"good.py": GOOD})
        info = load_plugins(hands, d, disabled=["good"])
        assert info[0]["status"] == "disabled"
        assert "demo.ping" not in hands.manifest()
    finally:
        rmtree(tmp)


def test_plugin_summary_and_settings_view():
    tmp = make_tmpdir()
    try:
        hands, conn = _hands(tmp)
        load_plugins(hands, _pdir(tmp, {"good.py": GOOD}))
        summary = hands.plugin_summary()
        entry = next(s for s in summary if s["name"] == "demo.ping")
        assert entry["source"] == "plugin:good"
    finally:
        rmtree(tmp)


def test_builtin_actions_still_registered():
    tmp = make_tmpdir()
    try:
        hands, conn = _hands(tmp)
        load_plugins(hands, _pdir(tmp, {"good.py": GOOD}))
        for name in ("reminder.set", "note.add", "web.fetch",
                     "bilibili.watch", "bilibili.comments", "desktop.notify"):
            assert name in hands._actions, name
            assert hands._actions[name]["source"] == "builtin"
        # shell.run 默认不在对外清单，但注册在表中（开关由 allow_shell 控制）
        assert "shell.run" in hands._actions
        assert "shell.run" not in [m["name"] for m in hands.manifest()]
    finally:
        rmtree(tmp)


if __name__ == "__main__":
    sys.exit(run_tests(globals(), "test_plugins"))
