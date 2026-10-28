"""HTTP I/O 面：聊天 API + 主动消息增量轮询 + 内置调度线程 + 自带网页客户端。

端点（除 /health 与两个网页外，设置 AGENT_API_TOKEN 后需带
Authorization: Bearer <token> 或 X-API-Token 头）：

网页：
  GET  /                     网页聊天客户端（零构建单文件）
  GET  /console              后台控制台（人物卡/记忆/轨迹/设置）

基础与观测：
  GET  /health               存活检查（免鉴权）
  GET  /status               状态向量 / 关系 / 今日主动 / 开关
  GET  /console-data         控制台聚合快照（人物卡、记忆、日志）
  GET  /trajectory?limit=80  状态轨迹
  GET  /messages?limit=50    最近消息（双向，含主动）
  GET  /pending?after_id=N   id > N 的 agent 消息增量轮询（客户端拉主动消息用）
  GET  /reminders            未完成提醒
  GET  /settings             当前可调设置视图

动作与设置：
  POST /chat   {"text": …, "images": [{"b64":…,"mime":…}]}   发送用户消息，返回 {reply, addon, id}
  POST /tick                 手动触发一次调度评估（内置线程之外的可选入口）
  POST /reflect              立即执行一次反思（force）
  POST /tts   {"text": …}    edge-tts 语音合成（未装依赖回 503）
  POST /presence {"typing": bool}    对端正在输入心跳
  POST /settings {…}         应用设置（非法项回 400）
  POST /config  {…}          运行时改配置（非法项回 400）
  POST /proactive {"enabled": bool}     总开关
  POST /level   {"level": "L2", "enabled": bool}  分级开关

约定：所有响应 JSON；错误统一 {"error": …}（400 参数问题 / 401 未授权 /
404 不存在 / 500 内部错误）。请求 body：空 = {}（等价无参调用）；
非法 JSON 或非 dict 回 400 invalid json body；查询参数需要整数时（limit/
after_id）非法值回 400。PUT 按 POST 处理。

版本与兼容：`/api/v1/...` 与裸路径完全等价（`/api/v1/chat` = `/chat`），
`/api` 同理。裸路径是历史形态、长期保留（网页客户端与微信桥都在用），
**新对接一律用 `/api/v1` 前缀**；`/health` 响应带 `"api": "v1"` 供握手探测。

线程模型：ThreadingHTTPServer 多线程收请求，Agent 内部 RLock 串行化全部
状态读写；内置 daemon 线程按 interval 秒调用 evaluate_proactive——
因此服务器模式下主动行为无需外部 cron。
"""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .main import Agent
from .storage import set_setting

WEB_INDEX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "index.html")
WEB_CONSOLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "console.html")


class AgentHandler(BaseHTTPRequestHandler):
    server_version = "WithMeAgent/1.0"

    # ---------------- 基础工具 ----------------

    def _reply_json(self, code: int, obj: dict):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        """解析请求 body：空 = {}；合法 JSON dict = dict；坏 JSON / 非 dict = None。"""
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return None
        raw = self.rfile.read(length) if length > 0 else b""
        if not raw:
            return {}
        try:
            data = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def _json_body(self):
        """_body 的带响应版：非法 JSON 直接回 400，调用方判 None 即返回。"""
        body = self._body()
        if body is None:
            self._reply_json(400, {"error": "invalid json body"})
        return body

    def _query_int(self, u, name: str, default: int):
        """查询参数取整数；非法值回 400 并返回 None（不再漏进 500）。"""
        try:
            return int(parse_qs(u.query).get(name, [str(default)])[0])
        except ValueError:
            self._reply_json(400, {"error": f"{name} 必须是整数"})
            return None

    @staticmethod
    def _api_path(raw_path: str) -> str:
        """标准化路由：/api 与 /api/v1 前缀与裸路径等价（加法式，不动历史路径）。"""
        for prefix in ("/api/v1", "/api"):
            if raw_path == prefix or raw_path.startswith(prefix + "/"):
                return raw_path[len(prefix):] or "/"
        return raw_path

    def _error_body(self, e: Exception) -> dict:
        """500 响应体：设了 AGENT_API_TOKEN（可能公网）只回通用信息防泄露，
        完整错误进服务日志；未设（本机调试）回 repr 便于排障。"""
        self.log_message("500 %s: %r", urlparse(self.path).path, e)
        if os.environ.get("AGENT_API_TOKEN"):
            return {"error": "internal error"}
        return {"error": repr(e)}

    def _authorized(self) -> bool:
        token = os.environ.get("AGENT_API_TOKEN", "")
        if not token:
            return True
        auth = self.headers.get("Authorization", "")
        return auth == f"Bearer {token}" or self.headers.get("X-API-Token") == token

    @property
    def agent(self) -> Agent:
        return self.server.agent

    def log_message(self, fmt, *args):
        print(f"[http {time.strftime('%H:%M:%S')}] {self.address_string()} {fmt % args}")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, X-API-Token")
        self.end_headers()

    # ---------------- 路由 ----------------

    def do_GET(self):
        u = urlparse(self.path)
        path = self._api_path(u.path)
        try:
            if path in ("/", "/index.html"):
                self._serve_page(WEB_INDEX)
            elif path == "/console":
                self._serve_page(WEB_CONSOLE)
            elif path == "/health":
                self._reply_json(200, {"ok": True, "api": "v1", "role_id": self.agent.role_id,
                                       "llm_provider": self.agent.client.name})
            elif not self._authorized():
                self._reply_json(401, {"error": "unauthorized"})
            elif path == "/status":
                self._reply_json(200, self.agent.status())
            elif path == "/console-data":
                self._reply_json(200, self.agent.console_snapshot())
            elif path == "/trajectory":
                limit = self._query_int(u, "limit", 80)
                if limit is None:
                    return
                self._reply_json(200, self.agent.trajectory(limit))
            elif path == "/settings":
                self._reply_json(200, self.agent.get_settings_view())
            elif path == "/reminders":
                items = [dict(r) for r in self.agent.conn.execute(
                    "SELECT id, text, due_at, status FROM reminders"
                    " WHERE role_id=? AND status='open' ORDER BY due_at",
                    (self.agent.role_id,)).fetchall()]
                self._reply_json(200, {"items": items})
            elif path == "/messages":
                limit = self._query_int(u, "limit", 50)
                if limit is None:
                    return
                self._reply_json(200, {"items": self.agent.recent_messages(limit)})
            elif path == "/pending":
                after_id = self._query_int(u, "after_id", 0)
                if after_id is None:
                    return
                items = self.agent.messages_after(after_id)
                self._reply_json(200, {"items": items,
                                       "last_id": items[-1]["id"] if items else after_id})
            else:
                self._reply_json(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            self._reply_json(500, self._error_body(e))

    def do_POST(self):
        u = urlparse(self.path)
        path = self._api_path(u.path)
        try:
            if not self._authorized():
                self._reply_json(401, {"error": "unauthorized"})
            elif path == "/chat":
                body = self._json_body()
                if body is None:
                    return
                text = (body.get("text") or "").strip()
                images = _parse_images(body.get("images"))
                if not text and not images:
                    self._reply_json(400, {"error": "text required"})
                    return
                result = self.agent.on_user_message(text, images=images)
                self._reply_json(200, result)
            elif path == "/tick":
                self._reply_json(200, self.agent.evaluate_proactive())
            elif path == "/tts":
                body = self._json_body()
                if body is None:
                    return
                text = (body.get("text") or "").strip()[:300]
                if not text:
                    self._reply_json(400, {"error": "text required"})
                    return
                try:
                    import asyncio
                    import edge_tts
                    voice = os.environ.get("AGENT_TTS_VOICE", "zh-CN-XiaoyiNeural")

                    async def _gen():
                        com = edge_tts.Communicate(text, voice)
                        chunks = []
                        async for c in com.stream():
                            if c["type"] == "audio":
                                chunks.append(c["data"])
                        return b"".join(chunks)

                    audio = asyncio.run(_gen())
                    self.send_response(200)
                    self.send_header("Content-Type", "audio/mpeg")
                    self.send_header("Content-Length", str(len(audio)))
                    self.end_headers()
                    self.wfile.write(audio)
                except ImportError:
                    self._reply_json(503, {"error": "pip install edge-tts 后可用"})
                except Exception as e:
                    self._reply_json(502, {"error": repr(e)})
            elif path == "/presence":
                body = self._json_body()
                if body is None:
                    return
                if body.get("typing") is True:
                    set_setting(self.server.agent.conn, "last_typing_ts", time.time())
                self._reply_json(200, {"ok": True})
            elif path == "/reflect":
                self._reply_json(200, self.agent.maybe_reflect(force=True))
            elif path == "/settings":
                body = self._json_body()
                if body is None:
                    return
                try:
                    self._reply_json(200, self.agent.apply_settings(body))
                except ValueError as e:
                    self._reply_json(400, {"error": str(e)})
            elif path == "/config":
                body = self._json_body()
                if body is None:
                    return
                try:
                    self._reply_json(200, self.agent.apply_config(body))
                except ValueError as e:
                    self._reply_json(400, {"error": str(e)})
            elif path == "/proactive":
                body = self._json_body()
                if body is None:
                    return
                if not isinstance(body.get("enabled"), bool):
                    self._reply_json(400, {"error": "enabled (bool) required"})
                    return
                self.agent.set_proactive(body["enabled"])
                self._reply_json(200, {"proactive_enabled": body["enabled"]})
            elif path == "/level":
                body = self._json_body()
                if body is None:
                    return
                level, enabled = body.get("level"), body.get("enabled")
                if level not in ("L1", "L2", "L3") or not isinstance(enabled, bool):
                    self._reply_json(400, {"error": "level in L1-L3 and enabled (bool) required"})
                    return
                self.agent.set_level_enabled(level, enabled)
                self._reply_json(200, {"level": level, "enabled": enabled})
            else:
                self._reply_json(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            self._reply_json(500, self._error_body(e))

    def do_PUT(self):
        self.do_POST()

    def _serve_page(self, path: str = WEB_INDEX):
        try:
            with open(path, "rb") as f:
                body = f.read()
        except OSError:
            # 基础 IO 发行版不含网页客户端：给 API 导览而不是报错
            info = ("WithMe agent（基础 IO 版，未捆绑网页客户端）\\n"
                    "GET /health /status /settings /messages?limit=N /pending?after_id=N\\n"
                    "POST /chat {text} · /tick · /reflect · /settings · /proactive · /level\\n"
                    "认证：设置 AGENT_API_TOKEN 后带 Authorization: Bearer <token>")
            body = ("<html><body style='font-family:monospace;background:#101014;color:#d8d8e0;"
                    "padding:24px;white-space:pre-wrap'>" + info + "</body></html>").encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def make_server(agent: Agent, host: str, port: int) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), AgentHandler)
    httpd.agent = agent
    return httpd


def _parse_images(raw) -> list:
    """解析 /chat 的 images 字段：[{"b64": …, "mime": …}] -> [{"bytes": …, "mime": …}]。"""
    if not isinstance(raw, list):
        return None
    import base64
    images = []
    for x in raw[:3]:
        if not (isinstance(x, dict) and x.get("b64")):
            continue
        try:
            raw_bytes = base64.b64decode(x["b64"])
        except (ValueError, TypeError):
            continue
        images.append({"bytes": raw_bytes, "mime": x.get("mime") or "image/jpeg"})
    return images or None


def serve(host: str = "127.0.0.1", port: int = 8787, db_path: str = None,
          interval: float = 60.0, config_dir: str = None):
    # memory_provider=sqlite：mem0 的本地 qdrant 有跨进程文件锁，微信桥是主使用方，
    # serve 不抢锁（否则启动竞态输的一方会话构造全炸——控制台聊天记忆退化为
    # SQLite 口径，事实仍进同一个库，只是不做向量语义检索）
    agent = Agent(config_dir=config_dir, db_path=db_path, threaded=True,
                  memory_provider="sqlite")
    httpd = make_server(agent, host, port)
    stop = threading.Event()

    def ticker():
        while not stop.wait(interval):
            try:
                result = agent.evaluate_proactive()
                sent = result.get("sent")
                if sent:
                    print(f"[scheduler {time.strftime('%H:%M:%S')}] "
                          f"主动 L{sent['level']} {sent['intent']}：{sent['text']}")
            except Exception as e:  # noqa: BLE001
                print(f"[scheduler] tick 失败：{e!r}")
            try:
                grown = agent.maybe_reflect()  # 证据足够时才真正执行（内部有阈值）
                if grown and grown.get("ran"):
                    print(f"[reflect {time.strftime('%H:%M:%S')}] 成长：{grown['applied']}")
            except Exception as e:  # noqa: BLE001
                print(f"[reflect] 失败：{e!r}")

    threading.Thread(target=ticker, daemon=True, name="scheduler").start()
    token_on = bool(os.environ.get("AGENT_API_TOKEN"))
    print(f"WithMe agent（{agent.persona.role_id} / 渲染端 {agent.client.name}）")
    print(f"  网页客户端: http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{httpd.server_address[1]}/")
    print(f"  API 基址  : http://{host}:{httpd.server_address[1]}  "
          f"认证：{'Bearer 令牌已启用' if token_on else '未设置 AGENT_API_TOKEN（仅建议本机绑定）'}")
    print(f"  内置调度  : 每 {agent.config.tick.fast_seconds:.0f}s 评估一次主动（控制台可热改）")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n收到中断，关闭服务…")
    finally:
        stop.set()
        httpd.server_close()
