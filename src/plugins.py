"""插件系统：社区不碰核心代码，往 plugins/ 目录丢一个 .py 就能给角色加"手"。

插件约定（见 plugins/README.md）：
    DESC = "一句话描述"
    ACTIONS = [{
        "name": "weather.get",          # 必须 前缀.动作，小写字母数字下划线
        "desc": "什么时候用这个动作",
        "params": {"参数名": "参数说明"},  # 会展示给感知层
        "handler": callable(params, ctx) -> {"ok": bool, "summary": str}
    }, ...]

安全模型（诚实声明）：插件与主程序同进程运行，拥有完整 Python 能力——
安装第三方插件 = 信任其作者。加载失败/单个动作执行失败都不影响主链路；
可通过 settings 的 plugins_disabled 列表禁用（重启生效）。
"""
import importlib.util
import json
import os
import re
import time
import urllib.request

from .storage import get_setting, set_setting

ACTION_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$")


class PluginContext:
    """给插件的受限上下文：桌面通知、带 UA 的网络抓取、按插件隔离的 KV 存储。
    不提供数据库句柄与 Agent 内部对象。"""

    def __init__(self, conn, role_id: str, plugin_name: str, notify, user_agent: str):
        self._conn = conn
        self._role_id = role_id
        self._plugin = plugin_name
        self._notify = notify
        self._ua = user_agent

    def notify(self, title: str, text: str):
        """弹桌面通知（失败自动降级为日志）。"""
        try:
            return bool(self._notify(title, text))
        except Exception:  # noqa: BLE001
            return False

    def fetch(self, url: str, timeout: int = 15) -> bytes:
        req = urllib.request.Request(url, headers={"User-Agent": self._ua})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()

    def fetch_text(self, url: str, timeout: int = 15) -> str:
        return self.fetch(url, timeout).decode("utf-8", "replace")

    def fetch_json(self, url: str, timeout: int = 15):
        return json.loads(self.fetch_text(url, timeout))

    def kv_set(self, key: str, value):
        set_setting(self._conn, f"plugin:{self._plugin}:{key}", value)

    def kv_get(self, key: str, default=None):
        return get_setting(self._conn, f"plugin:{self._plugin}:{key}", default)

    def log(self, *msg):
        print(f"[plugin:{self._plugin}]", *msg)


def load_plugins(hands, plugin_dir: str, disabled=()) -> list:
    """扫描并加载插件目录。单个插件失败只记日志，不影响其他插件与主程序。
    返回加载摘要（供 CLI/status 展示）。"""
    loaded = []
    if not os.path.isdir(plugin_dir):
        return loaded
    for fname in sorted(os.listdir(plugin_dir)):
        if not fname.endswith(".py") or fname.startswith("_"):
            continue
        plugin_name = fname[:-3]
        if plugin_name in disabled:
            loaded.append({"plugin": plugin_name, "status": "disabled"})
            continue
        path = os.path.join(plugin_dir, fname)
        try:
            spec = importlib.util.spec_from_file_location(f"withme_plugin_{plugin_name}", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            actions = getattr(mod, "ACTIONS", None)
            if not isinstance(actions, list) or not actions:
                raise ValueError("缺少非空 ACTIONS 列表")
            ctx = PluginContext(hands.conn, hands.role_id, plugin_name,
                                hands.notifier,
                                "Mozilla/5.0 WithMeAgent/1.0 (+plugin)")
            count = 0
            for act in actions:
                name = str(act.get("name", ""))
                handler = act.get("handler")
                if not ACTION_NAME_RE.match(name):
                    raise ValueError(f"动作名 {name!r} 不合法（须为 前缀.动作）")
                if not callable(handler):
                    raise ValueError(f"{name} 的 handler 不可调用")
                hands.register_action(
                    name, act.get("params") or {}, act.get("desc", ""),
                    _plugin_wrapper(plugin_name, name, handler, ctx),
                    source=f"plugin:{plugin_name}")
                count += 1
            loaded.append({"plugin": plugin_name, "status": "ok",
                           "desc": str(getattr(mod, "DESC", ""))[:60], "actions": count})
        except Exception as e:  # noqa: BLE001  坏插件不拖垮主程序
            print(f"[plugins] 加载失败 {fname}: {e!r}")
            loaded.append({"plugin": plugin_name, "status": "error", "error": repr(e)[:120]})
    return loaded


def _plugin_wrapper(plugin_name, action_name, handler, ctx):
    def run(params: dict, now: float) -> dict:
        result = handler(params, ctx)
        if not isinstance(result, dict) or "ok" not in result or "summary" not in result:
            return {"ok": False, "summary": f"插件返回格式不对（需要 ok/summary）"}
        result.setdefault("ok", False)
        result["summary"] = str(result["summary"])[:900]
        result["action"] = action_name
        return result
    run.__name__ = f"{plugin_name}:{action_name}"
    return run
