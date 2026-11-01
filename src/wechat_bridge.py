"""微信桥：把 WithMe REST API 接到个人微信。双通道可选（config wechat.channel）：

- wcf   —— wcferry hook（PC 微信 **3.9.12.51**，pip 包）。语音回复 = TTS mp3 文件卡片
           （wcferry 无发送语音接口，RPC 协议层面缺失）。
- wxhook —— github.com/mengyunqwq/wechat-hook（PC 微信 **4.1.10.27**，version.dll 代理，
           HTTP :30001）。语音回复 = **真语音气泡**（type 34）：TTS mp3 -> ffmpeg 24kHz
           PCM -> pilk SILK -> /SendVoiceMsg(bytes 模式，调微信自己的语音工厂）。

架构（README「消息通道」：REST API 是边界）——本模块是一个独立客户端进程：

    PC 微信(常驻登录, 按通道锁版本) <--hook--> wcferry / wechat-hook <-- 本桥 --> WithMe serve

两条线程：
- 接收线程：通道入站消息 -> 路由（文本/图片/语音/链接）-> POST /chat（或 ASR/解码后转发）
- 投递线程：GET /pending?after_id=N 增量轮询 -> 微信发送。
  回复与主动消息走同一条投递路，靠水位线（watermark）天然去重——桥重启不重播历史。

依赖：
- wcf 通道：pip install wcferry；微信 3.9.12.51
- wxhook 通道：编译 version.dll（见 docs/wechat_setup.md）；微信 4.1.10.27；
               pip install pilk imageio-ffmpeg zstandard（语音气泡/收消息解码）
- 公共：pip install edge-tts（TTS，服务端 /tts 用）
"""
import hashlib
import json
import os
import queue
import re
import threading
import time
import urllib.error
import urllib.request

from .config import load_config
from .prompt_builder import strip_media_markers, strip_role_prefix
from .storage import project_root


def _image_mime(raw: bytes) -> str:
    """按魔数嗅探图片 MIME（桥打包视觉请求用；未知格式回落 jpeg）。
    放这里而不是 wximg：wximg 顶层就访问 ctypes.windll，Linux 上 import 即炸。
    之前桥把所有图硬编码成 image/jpeg，GIF/PNG 的真实格式信息丢在解密层。"""
    if raw[:3] == b"GIF":
        return "image/gif"
    if raw[:4] == b"\x89PNG":
        return "image/png"
    if raw[:4] == b"RIFF":
        return "image/webp"
    return "image/jpeg"


# 消息类型（两个通道一致：1 文本 / 3 图片 / 34 语音 / 49 appmsg 链接）
MSG_TEXT = 1
MSG_IMAGE = 3
MSG_VOICE = 34
MSG_APPMSG = 49


def load_env_dotfile():
    """读仓库根 .env（setdefault，不覆盖真实环境变量；与 tools 的约定一致）。"""
    path = os.path.join(project_root(), ".env")
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class BridgeError(Exception):
    pass


# ---------------- wcferry 通道（包装成统一鸭子接口） ----------------

class WcfChannel:
    name = "wcf"
    voice_prefers = "mp3"        # 该通道语音只能以 mp3 文件卡片发出

    def __init__(self, wcf):
        self.wcf = wcf

    def is_login(self):
        return self.wcf.is_login()

    def get_self_wxid(self):
        return self.wcf.get_self_wxid()

    def enable_receiving(self):
        return self.wcf.enable_receiving_msg()

    def disable_receiving(self):
        try:
            return self.wcf.disable_receiving_msg()
        except Exception:  # noqa: BLE001
            return None

    def get_msg(self, block=True):
        return self.wcf.get_msg(block=block)

    def send_text(self, msg, receiver, aters=""):
        return self.wcf.send_text(msg, receiver, aters)

    def send_image(self, path, receiver):
        return self.wcf.send_image(path, receiver)

    def send_file(self, path, receiver):
        return self.wcf.send_file(path, receiver)

    def send_voice_silk(self, silk_path, receiver, duration_ms=0):
        return -1  # wcferry 无发送语音接口（协议缺失），桥不会走到这里

    def fetch_voice(self, msg):
        """语音 -> mp3（wcf 内置 silk->mp3 转换）。"""
        mp3 = self.wcf.get_audio_msg(int(msg.id), self._media_dir, timeout=5)
        if not mp3 or not os.path.exists(mp3):
            return None
        return {"path": mp3, "mime": "audio/mpeg"}

    def fetch_image(self, msg):
        return self.wcf.download_image(int(msg.id), msg.extra or "", self._media_dir,
                                       timeout=30) or None

    def bind_media_dir(self, media_dir: str):
        self._media_dir = media_dir
        return self


# ---------------- WithMe REST 客户端 ----------------

class AgentAPI:
    """极薄 REST 客户端：chat / pending / tts / health / messages。"""

    def __init__(self, base_url: str, token: str = "", timeout: int = 180):
        self.base = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _request(self, path: str, body: dict = None):
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method="POST" if data else "GET",
                                     headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read()
        ctype = resp.headers.get("Content-Type", "")
        if "json" in ctype:
            return json.loads(raw.decode("utf-8"))
        return raw  # /tts 返回 audio/mpeg 字节

    def chat(self, text: str, images: list = None) -> dict:
        body = {"text": text}
        if images:
            body["images"] = images
        return self._request("/chat", body)

    def pending(self, after_id: int) -> list:
        return (self._request(f"/pending?after_id={int(after_id)}") or {}).get("items") or []

    def latest_message_id(self) -> int:
        items = (self._request("/messages?limit=1") or {}).get("items") or []
        return items[-1]["id"] if items else 0

    def tts(self, text: str):
        """返回 mp3 bytes；不可用抛 BridgeError。"""
        try:
            return self._request("/tts", {"text": text})
        except urllib.error.HTTPError as e:
            raise BridgeError(f"/tts HTTP {e.code}") from None

    def health(self) -> dict:
        return self._request("/health")


# ---------------- 桥状态（主人绑定 + 投递水位线，跨重启持久化） ----------------

class BridgeState:

    def __init__(self, path: str):
        self.path = path
        self.owner_wxid = ""
        self.watermark = 0
        self.self_name = ""     # 我自己的群昵称：contact.db 抖动/重启时兜底，防"点名认不出"
        self.overrides = {}     # 聊天命令改过的桥级参数（重启后覆盖 config.yaml）
        self._lock = threading.Lock()   # 水位线/命令/主人绑定都可能并发写状态文件
        if os.path.exists(path):
            try:
                data = json.load(open(path, encoding="utf-8"))
                self.owner_wxid = data.get("owner_wxid") or ""
                self.watermark = int(data.get("watermark") or 0)
                self.self_name = data.get("self_name") or ""
                self.overrides = dict(data.get("overrides") or {})
            except (json.JSONDecodeError, OSError, ValueError):
                pass

    def save(self):
        # 临时文件 + 原子替换：并发写/进程被杀都不会留下半截 JSON
        #（状态文件里有 owner_wxid，损坏会导致主人重新绑定，后果比丢水位线严重）
        with self._lock:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"owner_wxid": self.owner_wxid, "watermark": self.watermark,
                           "self_name": self.self_name, "overrides": self.overrides}, f,
                          ensure_ascii=False)
            os.replace(tmp, self.path)


# ---------------- 多用户池（wechat.multi_user=true：每个私聊用户一份独立小满） ----------------

class UserPool:
    """wxid -> 独立 Agent 实例（独立 SQLite 库）。主人沿用默认库（记忆连续），其他人
    各建 data/wechat_users/<wxid>/agent.db。上限 max_users 防陌生人刷爆。"""

    def __init__(self, cfg, log, owner_wxid_getter, base_dir: str = None, channel=None):
        self.cfg = cfg
        self.log = log
        self._owner_getter = owner_wxid_getter   # () -> 当前主人 wxid（可为空）
        self._base = base_dir or os.path.join(project_root(), "data")
        self.agents = {}
        self.lock = threading.Lock()
        self.channel = channel    # 取群名/头像/昵称用（可空，鸭子接口）

    def _db_path_for(self, wxid: str):
        if wxid and wxid == self._owner_getter():
            return None  # 主人走默认库（与网页端/历史记忆连续）
        safe = "".join(c for c in wxid if c.isalnum() or c in "_-") or "unknown"
        d = os.path.join(self._base, "wechat_users", safe)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, "agent.db")

    def warm_sessions(self):
        """启动时预建已存在的会话 Agent（每个群 / 每个其他私聊用户一份）：
        把群名、昵称写进库，后台控制台的会话轴一上来就有可读的名字
        （否则只能看到 roomid）。失败不影响收发。"""
        base = os.path.join(self._base, "wechat_users")
        if not os.path.isdir(base):
            return
        for name in sorted(os.listdir(base)):
            if not os.path.exists(os.path.join(base, name, "agent.db")):
                continue
            sid = name.replace("chatroom", "@chatroom") if name.endswith("chatroom") else name
            try:
                self.get(sid)
            except Exception as e:  # noqa: BLE001  预热失败不挡启动
                self.log(f"预热会话 Agent 失败 {sid}：{e!r}")

    def _display_name(self, wxid: str) -> str:
        """群/用户显示名（contact.db 的 remark/nick_name）。取不到就空串，不影响收发。"""
        getter = getattr(self.channel, "nickname_of", None)
        if getter is None:
            return ""
        try:
            return str(getter(wxid) or "")
        except Exception:  # noqa: BLE001  昵称只是展示层
            return ""

    def _wire_avatar(self, agent, wxid: str):
        """给 hands 注入头像获取器（表情包头像梗图：'对方'=该用户，'我'=bot）。"""
        ch = getattr(self, "channel", None)
        if ch is None or not hasattr(ch, "avatar_bytes"):
            return
        self_wxid = ch.get_self_wxid()

        def getter(who: str) -> bytes:
            return ch.avatar_bytes(self_wxid if who == "我" else wxid)

        agent.hands.avatar_getter = getter

    def get(self, wxid: str):
        """返回该用户的 Agent；超上限返回 None（调用方忽略并记日志）。"""
        with self.lock:
            agent = self.agents.get(wxid)
            if agent is not None:
                agent.loose_mode = bool(wxid in (self.cfg.wechat.loose_chats or []))
                return agent
            if len(self.agents) >= self.cfg.wechat.max_users:
                return None
            from .main import Agent
            db = self._db_path_for(wxid)
            is_group = bool(wxid and wxid.endswith("@chatroom"))
            agent = Agent(db_path=db, threaded=True,
                          group_room=wxid if is_group else "",
                          group_name=self._display_name(wxid),
                          task_owner="bridge")
            agent.is_group = is_group
            agent.loose_mode = bool(wxid in (self.cfg.wechat.loose_chats or []))
            self._wire_avatar(agent, wxid)
            self.agents[wxid] = agent
            self.log(f"多用户：为 {wxid} 建独立小满（db={db or '默认库'}，"
                     f"当前 {len(self.agents)} 人）")
            return agent


# ---------------- 拟人发送（拆条 + 打字延迟） ----------------

import random
import re as _re


VOICE_WANT_RE = _re.compile(
    r"用语音|语音(?:回答|说|回复|发|讲)|发(?:一|条)?个?语音|说给我听|语音发给我"
    # 唱歌是口头发声请求，文字念歌词不算数——代码级识别，不依赖模型自觉加标记
    r"|唱歌|唱吧|唱一?(?:个|首|支)|来一?首歌?|点一?首歌?")


def split_bubbles(text: str, max_bubbles: int = 3) -> list:
    """长回复拆成多条短消息（像人连发）。短文本原样单条。

    拆分优先级：
    1. 含换行 = 模型显式分段，按行拆成气泡（写 N 行就发 N 条；超出的行并入
       最后一条防刷屏）——实测病灶：43 字双行文本从快速路径原样单发，
       微信里呈现为"一条消息中间空一行"，像消息被截断；
    2. 单行长文本（>48 字）：按句末标点切，碎片合并成 2~max 条。"""
    text = (text or "").strip()
    if not text:
        return []
    if max_bubbles <= 1:
        return [text]
    if chr(10) in text:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(lines) <= 1:
            return [text]
        if len(lines) > max_bubbles:
            lines = lines[:max_bubbles - 1] + ["".join(lines[max_bubbles - 1:])]
        return lines
    if len(text) <= 48:
        return [text]
    parts = [p.strip() for p in _re.split(r"(?<=[。！？!?…])+", text) if p.strip()]
    if len(parts) <= 1:
        return [text]
    # 贪心合并成 2~max 条、每条尽量 ≥10 字
    target = min(len(parts), max_bubbles)
    out, cur = [], ""
    for p in parts:
        if cur and len(out) < target - 1 and len(cur) + len(p) > max(18, len(text) // target):
            out.append(cur)
            cur = p
        else:
            cur = cur + p if cur else p
    if cur:
        out.append(cur)
    return [b for b in out if b] or [text]


# ---------------- 桥本体 ----------------

class WeChatBridge:

    def __init__(self, cfg, channel, api: AgentAPI, state: BridgeState,
                 asr=None, media_dir: str = None, log=None):
        """cfg: AppConfig。channel/asr 可注入假件供测试（鸭子接口见 WcfChannel/WxHookChannel）。"""
        self.cfg = cfg
        self.channel = channel
        self.api = api
        self.state = state
        self.asr = asr
        self.log = log or (lambda msg: print(f"[wechat {time.strftime('%H:%M:%S')}] {msg}"))
        self.media_dir = media_dir or os.path.join(project_root(), cfg.wechat.media_dir)
        os.makedirs(self.media_dir, exist_ok=True)
        if hasattr(channel, "bind_media_dir"):
            channel.bind_media_dir(self.media_dir)
        self.stop = threading.Event()
        self._voice_pending = 0      # 镜像策略：待语音回复的计数
        self._inflight = 0           # 正在生成回复的条数（重启/排障看，不杀在途）
        self._inflight_lock = threading.Lock()
        self._last_err_note = 0.0    # 后台故障提示的限频
        self.multi = bool(cfg.wechat.multi_user)
        # 每用户库放在 <data>/wechat_users/（与 media_dir 同级；测试注入 tmp 即隔离）
        self.pool = (UserPool(cfg, self.log, lambda: self.state.owner_wxid,
                              base_dir=os.path.dirname(self.media_dir),
                              channel=channel)
                     if self.multi else None)
        # 每用户独立工作线程：同一用户严格保序，不同用户互不阻塞（LLM 慢不再拖全场）
        self._peer_queues = {}
        self._peer_lock = threading.Lock()
        self.async_dispatch = True   # 测试置 False 走同步直调
        # 群 -> (最近说话人 wxid, 昵称)：群主动时 @ 谁
        self._group_last = {}
        # 发送节流：peer -> 最近发送时间戳；任何路径（回复/插嘴/主动）都不能把同一个
        # 会话刷爆——"疯狂发消息"就是这里没有硬闸（2026-10-28 实测）
        self._last_send = {}
        self._send_log = {}
        # 启动时刻：刚起来的几分钟不主动开口（重启后立刻说话最像"抽风"）
        self._started_at = time.time()
        # 群智能插嘴的频率状态：room -> {msgs, last_ts, last_chime, day, chimes}
        self._group_watch = {}
        # peer -> 自己最后一条回复的时间：刚说完话不主动（真人不会自言自语）
        self._last_reply_at = {}

    # ---------------- TTS（本地 edge-tts 优先，REST 兜底） ----------------

    def _tts_bytes(self, text: str):
        """返回 mp3 bytes；失败抛 BridgeError。多用户模式下桥可脱离服务端独立跑。"""
        try:
            import asyncio
            import edge_tts
            voice = os.environ.get("AGENT_TTS_VOICE", "zh-CN-XiaoyiNeural")

            def _gen():
                com = edge_tts.Communicate(text, voice)
                chunks = []
                loop = asyncio.new_event_loop()
                try:
                    async def _run():
                        async for c in com.stream():
                            if c["type"] == "audio":
                                chunks.append(c["data"])
                    loop.run_until_complete(_run())
                finally:
                    loop.close()
                return b"".join(chunks)

            audio = _gen()
            if audio:
                return audio
        except Exception as e:  # noqa: BLE001  本地失败回落 REST
            self.log(f"本地 TTS 失败，回落服务端：{e!r}")
        return self.api.tts(text)

    # ---------------- 接收路径 ----------------

    @staticmethod
    def _msg_age(msg) -> float:
        """消息在微信里的创建时间距现在多少秒（拿不到返回 0=不判）。"""
        ct = float(getattr(msg, "create_time", 0) or getattr(msg, "ts", 0) or 0)
        if ct > 1e12:      # 毫秒时间戳
            ct = ct / 1000.0
        return max(0.0, time.time() - ct) if ct else 0.0

    def _chat_enabled(self, peer: str) -> bool:
        """启用范围（wechat.enabled_chats）：空=全部；非空=只处理列出的会话
        （群房间号或私聊 wxid）。停用名单 disabled_chats 优先（/chat off 写这里）。
        范围外的消息不回复、不记记忆、不跑主动，但主人的斜杠命令例外。"""
        who = str(peer or "")
        disabled = [str(x).strip() for x in
                    (getattr(self.cfg.wechat, "disabled_chats", None) or [])
                    if str(x).strip()]
        if who in disabled:
            return False
        allow = [str(x).strip() for x in
                 (getattr(self.cfg.wechat, "enabled_chats", None) or [])
                 if str(x).strip()]
        return not allow or who in allow

    def _observe_chat(self, peer: str) -> bool:
        """观察模式：主动/插嘴只记录不发送，用于上线前看“本来会说什么”。"""
        return str(peer or "") in [
            str(x).strip() for x in
            (getattr(self.cfg.wechat, "observe_chats", None) or [])
            if str(x).strip()]

    def chat_label(self, peer: str) -> str:
        """会话显示名（群名/昵称/房间号），命令回执与 /chat list 用。"""
        peer = str(peer or "")
        ag = self.pool.agents.get(peer) if self.pool is not None else None
        name = str(getattr(ag, "group_name", "") or "") if ag is not None else ""
        if not name:
            try:
                name = self._display_name(peer) or ""
            except Exception:  # noqa: BLE001  昵称只是展示层
                name = ""
        return name or peer

    def resolve_chat(self, token: str):
        """"AI技术冲了" / wxid / 房间号 -> room id。

        只查已经建过 Agent 的会话（启动预热/收过消息的群都在里面）；
        名字先精确匹配，再子串匹配，避免"家人"命中"家人群"这类歧义时乱指。
        """
        t = str(token or "").strip()
        if not t:
            return None
        if t.endswith("@chatroom") or t.startswith("wxid_"):
            return t
        if self.pool is None:
            return None
        cands = []
        for wid, ag in list(self.pool.agents.items()):
            name = str(getattr(ag, "group_name", "") or "")
            if not name:
                try:
                    name = self._display_name(wid) or ""
                except Exception:  # noqa: BLE001
                    name = ""
            if not name:
                continue
            if t == name:
                return wid
            if t in name:
                cands.append(wid)
        return cands[0] if cands else None

    def _maybe_command(self, msg) -> bool:
        """主人斜杠命令：在启用范围/群路由之前截住。

        返回 True = 已消化（命令/非主人命令），调用方直接 return；
        返回 False = 不是命令，继续走正常聊天。命令不进对话历史，也不进记忆。

        群里不强制 @：微信的 @ 元数据不可靠（兜底轮询补收的消息 source 里
        可能没有 atuserlist，2026-10-30 实测：连着两条 @小满 /xxx，第二条
        被当成"没@我"）。真正的守卫是"只认主人 + 只认白名单命令名"，
        普通文本里的 "/" 不会被误吃。
        """
        if msg.type != MSG_TEXT:
            return False
        from . import chat_commands
        text = self._strip_group_noise(msg.content or "")
        if chat_commands.parse(text) is None:
            return False
        sender = msg.sender or msg.roomid
        is_group = bool(msg.from_group())
        peer = msg.roomid if is_group else sender
        owner = bool(self.state.owner_wxid) and sender == self.state.owner_wxid
        if not owner:
            if is_group:
                self.log(f"群 {peer}：非主人的命令，忽略")
                return True
            return False   # 私聊非主人：当普通聊天，继续往下走
        self.log(f"命令：{peer} {text[:40]}")
        self._dispatch_command(text, peer, None, is_group)
        return True

    def _dispatch_command(self, text: str, peer: str, agent, is_group: bool):
        """执行命令并回执。放在线程里：Agent 的锁可能被进行中的对话占着
        （一轮最多几十秒），建库也可能要等内存模型，命令不能把接收线程堵住。"""
        from . import chat_commands

        def run():
            nonlocal agent
            if agent is None and self.multi and self.pool is not None:
                try:
                    agent = self.pool.get(peer)   # 懒建库也放到线程里
                except Exception as e:  # noqa: BLE001
                    self.log(f"命令建库失败 {peer}：{e!r}")
            ctx = chat_commands.Ctx(agent=agent, bridge=self, peer=peer,
                                    is_group=is_group, is_owner=True, now=time.time())
            try:
                reply = chat_commands.handle(text, ctx)
            except Exception as e:  # noqa: BLE001  命令失败不能拖垮桥
                reply = f"命令出错：{type(e).__name__}: {e}"
            if reply:
                self._send_text(reply, peer)

        if self.async_dispatch:
            threading.Thread(target=run, daemon=True, name="wx-cmd").start()
        else:
            run()

    def handle_message(self, msg) -> None:
        """一条入站消息 -> 路由。通道注入假件时本函数即测试面。"""
        try:
            if msg.from_self():
                return
            # 历史回放守卫：重启、兜底补收、库重建都可能把"老消息"重新送进来，
            # 一条条回过去就是把过去全答一遍（2026-10-28 实测事故：几分钟内把
            # 两天前的对话逐条重答）。超过阈值的直接丢，不回、不入库。
            limit = float(getattr(self.cfg.wechat, "max_message_age_seconds", 180.0) or 0)
            age = self._msg_age(msg)
            if limit > 0 and age > limit:
                peer0 = getattr(msg, "roomid", "") or getattr(msg, "sender", "")
                self.log(f"跳过历史消息（{int(age)}s 前 · {peer0}）："
                         f"{(getattr(msg, 'content', '') or '')[:30]}")
                return
            peer0 = getattr(msg, "roomid", "") or getattr(msg, "sender", "")
            # 主人命令先于启用范围：这样在已被 /chat off 停用的会话里也能 /chat on
            if self._maybe_command(msg):
                return
            if not self._chat_enabled(peer0):
                return   # 启用范围外：静默忽略（不回复/不入库/不触发记忆）
            if msg.from_group():
                if self.multi and getattr(self.cfg.wechat, "groups_enabled", False):
                    self._group_message(msg)
                return
            peer = msg.sender or msg.roomid
            if not peer:
                return
            if self.multi:
                if self._multi_ok(peer):
                    self._multi_message(msg, peer)
                return
            if not self._owner_ok(peer):
                return
            if msg.type == MSG_TEXT:
                self._on_text(msg, peer)
            elif msg.type == MSG_IMAGE:
                self._on_image(msg, peer)
            elif msg.type == MSG_VOICE:
                self._on_voice(msg, peer)
            elif msg.type == MSG_APPMSG:
                self._on_link(msg, peer)
            # 视频/表情/名片等暂不处理（记录在案，不回复）
        except Exception as e:  # noqa: BLE001  单条消息失败不拖垮桥
            self.log(f"消息处理失败 {getattr(msg, 'type', '?')}: {e!r}")

    # ---------------- 媒体获取（单/多用户共用） ----------------

    def _fetch_image_b64(self, msg, peer):
        """取图解密 -> b64 列表；拿不到时给对方温和提示并返回 None。"""
        import base64
        path = None
        try:
            path = self.channel.fetch_image(msg)
        except Exception as e:  # noqa: BLE001
            self.log(f"图片获取失败：{e!r}")
        if path and os.path.exists(path):
            # 魔数校验：没打开过的群只有缩略图 _t.dat（密钥与全图不同，解出
            # 垃圾字节，发给视觉端必被 11135 拒收）——垃圾不当图用
            with open(path, "rb") as f:
                magic = f.read(4)
            if magic[:3] != bytes([0xFF, 0xD8, 0xFF]) and magic[:4] not in (
                    bytes([0x89, 0x50, 0x4E, 0x47]), b"RIFF", b"GIF8"):
                self.log(f"图片解密产物非有效图片（疑似仅缩略图）msg.id={msg.id}")
                self._send_text("这张我只拿到缩略图，群里打开一下这张图我就能看到原图", peer)
                return None
        if not path or not os.path.exists(path):
            self.log(f"图片拿不到 msg.id={msg.id}")
            self._send_text("（这张图我没能打开，再发一次试试？）", peer)
            return None
        with open(path, "rb") as f:
            raw = f.read()
        self.log(f"收图 {len(raw) // 1024}KB -> 视觉理解")
        return [{"b64": base64.b64encode(raw).decode(), "mime": _image_mime(raw)}]

    def _fetch_voice_text(self, msg, peer):
        """语音 -> ASR 文本；失败给提示并返回 None。"""
        if self.asr is None:
            self.log("语音识别未配置，跳过")
            return None
        audio = None
        try:
            audio = self.channel.fetch_voice(msg)
        except Exception as e:  # noqa: BLE001
            self.log(f"语音获取失败：{e!r}")
        if not audio:
            self.log(f"语音拿不到 msg.id={msg.id}")
            self._send_text("（这条语音我没能收到，再发一次试试？）", peer)
            return None
        try:
            with open(audio["path"], "rb") as f:
                text = self.asr.transcribe(f.read(),
                                           filename=os.path.basename(audio["path"]),
                                           mime=audio.get("mime", "audio/mpeg"))
        except Exception as e:  # noqa: BLE001
            self.log(f"ASR 失败：{e!r}")
            self._send_text("（语音我没听清，打字跟我说吧）", peer)
            return None
        self.log(f"语音听写：{text[:50]}")
        return text

    @staticmethod
    def _parse_quote(msg, self_wxid=""):
        """引用回复（appmsg type=57）-> '[引用 X：「…」] 新回复'。

        引用消息此前整条隐形（appmsg 分支只认链接，refermsg 没人解析）——
        用户引用小满的话提问，小满完全看不到被引内容，答非所问。
        被引内容是 XML（表情包/图片/语音）时按类型概括。"""
        import html as _html
        from .wxhook import payload_xml
        xml = payload_xml(msg, ("<refermsg>",))
        i = xml.find("<refermsg>")
        if i < 0:
            return None
        seg = xml[i:]

        def _tag(name, src):
            m = _re.search(r"<{0}>(.*?)</{0}>".format(name), src, _re.S)
            return (m.group(1).strip() if m else "")

        title = _html.unescape(_tag("title", xml)).strip()
        ref = _html.unescape(_tag("content", seg)).strip()
        name = _html.unescape(_tag("displayname", seg)).strip() or "对方"
        chatusr = _html.unescape(_tag("chatusr", seg)).strip()
        if chatusr and self_wxid and chatusr == self_wxid:
            name = "你自己"   # 引用的是我自己的话：别再说"那句话不是我说的"
        if ref.startswith("<"):
            # 引用的是另一条引用/链接/文件：优先取它自己的 title，别一律说“一条消息”
            nested_title = _html.unescape(_tag("title", ref)).strip()
            if nested_title:
                ref = nested_title
            elif "<emoji" in ref[:120]:
                ref = "一个表情包"
            elif "<img" in ref[:120]:
                ref = "一张图片"
            elif "voicemsg" in ref[:160]:
                ref = "一条语音"
            else:
                ref = "一条消息"
        ref = ref[:60] + ("…" if len(ref) > 60 else "")
        out = f"[引用 {name}：「{ref}」]"
        return (out + " " + title).strip() if title else out

    def _quote_image(self, msg):
        """引用的是图片时，把被引原图解出来一并给视觉。

        引用图片消息（refermsg type=3）的 content 自带原图的 aeskey/md5，
        之前只概括成「一张图片」，问"这图是什么"必然答不上。解密走 wximg
        的三档+多密钥路径（原图已落盘走本地，只剩缩略图走账号派生密钥）。
        成功返回 b64 图片列表；失败返回 None（引用文本里仍是"一张图片"）。"""
        import base64
        import html as _html
        from urllib.parse import urlparse
        from .wxhook import payload_xml
        xml = payload_xml(msg, ("<refermsg>",))
        i = xml.find("<refermsg>")
        if i < 0:
            return None
        seg = xml[i:]
        m_type = _re.search(r"<type>(\d+)</type>", seg)
        if not m_type or m_type.group(1) != "3":
            return None
        m_ref = _re.search(r"<content>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</content>",
                           seg, _re.S)
        ref = _html.unescape(m_ref.group(1)) if m_ref else ""
        m_md = _re.search(r'\bmd5="([0-9a-fA-F]{32})"', ref)
        if not m_md:
            return None
        resolve = getattr(self.channel, "_resolve_data_root", None)
        root = resolve() if resolve else None
        if not root:
            return None
        attach = os.path.join(root, "msg", "attach")
        synth = '<img md5="%s"' % m_md.group(1)
        m_ak = _re.search(r'aeskey="([0-9a-fA-F]{32})"', ref)
        if m_ak:
            synth += ' aeskey="%s"' % m_ak.group(1)
        synth += "/>"
        from .wximg import decode_img_xml
        port = urlparse(getattr(self.channel, "base", "") or "").port or 30001
        dst = os.path.join(self.media_dir, "qimg_%s.jpg" % msg.id)
        try:
            fmt = decode_img_xml(dst, synth, attach, log=self.log,
                                 hook_port=port,
                                 cache_path=os.path.join(
                                     os.path.dirname(self.media_dir),
                                     "wechat_image_key.json"))
        except Exception as e:  # noqa: BLE001  引用图解不出就走文字概括
            self.log(f"引用图片解密失败：{e!r}")
            return None
        if not fmt or fmt == "WXGF" or not os.path.exists(dst) \
                or os.path.getsize(dst) < 100:
            self.log(f"引用图片不可用（fmt={fmt}）")
            return None
        with open(dst, "rb") as f:
            raw = f.read()
            self.log(f"引用图片已解出（{fmt}, {len(raw) // 1024}KB）")
            return [{"b64": base64.b64encode(raw).decode(),
                     "mime": _image_mime(raw)}]

    @staticmethod
    def _parse_link(msg):
        """appmsg 里的分享链接 -> "[链接] 标题 URL"；非链接返回 None。"""
        import re
        from .wxhook import payload_xml
        # wxhook 通道载荷在 content（xml 是 <msgsource> 元数据），wcf 在 xml
        xml = payload_xml(msg, ("<url>", "<appmsg"))
        m_url = re.search(r"<url>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</url>", xml)
        if not m_url:
            return None
        url = m_url.group(1).strip()
        m_title = re.search(r"<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>", xml)
        title = (m_title.group(1).strip() if m_title else "")[:60]
        return f"[链接] {title} {url}"

    # ---------------- 多用户路径（wechat.multi_user=true） ----------------

    def _multi_ok(self, peer: str) -> bool:
        allowed = self.cfg.wechat.allowed_wxids
        if allowed and peer not in allowed:
            self.log(f"多用户：{peer} 不在白名单，忽略")
            return False
        if peer.endswith("@chatroom"):
            return True   # 群绝不参与主人绑定
        if not self.state.owner_wxid:
            # 第一个说话的人成为主人：沿用默认库，与网页端/既有记忆连续
            self.state.owner_wxid = peer
            self.state.save()
            self.log(f"多用户：绑定主人 {peer}")
        return True

    # ---------------- 群聊：谁在跟"我"说话 ----------------

    def _group_self_names(self, agent=None) -> list:
        """群里可能指代"我"的名字：群昵称（contact.db）+ 人设名 + 配置别名。

        群昵称是权威（别人看到的名字）；人设名与 `wechat.group_self_aliases`
        兜底（如"小满""机器人"）。任何一个被点名都按直接对话处理。"""
        names = []
        nick_getter = getattr(self.channel, "nickname_of", None)
        if nick_getter:
            try:
                nick = nick_getter(self.channel.get_self_wxid() or "") or ""
            except Exception:  # noqa: BLE001  昵称是增强，失败不影响路由
                nick = ""
            if nick:
                names.append(nick)
                if getattr(self.state, "self_name", "") != nick:
                    self.state.self_name = nick
                    try:
                        self.state.save()
                    except Exception:  # noqa: BLE001  缓存名字失败不影响路由
                        pass
            elif getattr(self.state, "self_name", ""):
                # contact.db 抖动/没起来：用上次记住的群昵称，别把自己当陌生人
                names.append(self.state.self_name)
        persona = getattr(agent, "persona", None)
        ident = getattr(persona, "identity_summary", "") if persona is not None else ""
        m = _re.match(r"\s*我是([^，,。；;\s]+)", ident or "")
        if m:
            names.append(m.group(1))
        for extra in getattr(self.cfg.wechat, "group_self_aliases", []) or []:
            if str(extra).strip():
                names.append(str(extra).strip())
        return list(dict.fromkeys(names))

    def _group_owner_name(self) -> str:
        """创造者的群昵称：给插嘴提示词标注[创造者]；只表达亲近，不表达权限。"""
        owner = self.state.owner_wxid or ""
        getter = getattr(self.channel, "nickname_of", None)
        if not owner or not getter:
            return ""
        try:
            return getter(owner) or ""
        except Exception:  # noqa: BLE001
            return ""

    @staticmethod
    def _text_mentions_names(text: str, names: list) -> bool:
        """正文是否点名了 names 之一；英文名按词边界，避免 AI/AW 误伤。"""
        text = str(text or "")
        for name in names or []:
            if not name:
                continue
            if str(name).isascii():
                if _re.search(r"(?<![A-Za-z0-9_])" + _re.escape(str(name))
                              + r"(?![A-Za-z0-9_])", text, _re.I):
                    return True
            elif str(name) in text:
                return True
        return False

    def _group_mentions_me(self, msg, agent=None) -> bool:
        """正文是否提到我的名字（包括别人在讨论我/让人骂我）。"""
        if msg.type not in (MSG_TEXT, MSG_APPMSG):
            return False
        return self._text_mentions_names(msg.content or "",
                                         self._group_self_names(agent))

    def _group_addressed_to_me(self, msg, agent=None) -> bool:
        """正文是否在"叫我"（不是提到我，也不是让别人骂我）。

        - "阿满，..." / "@阿满 ..." -> 直接叫我；
        - "帮我骂 阿满" / "骂 阿满" -> 阿满 是宾语，不是收件人；
        - "阿满 是..." -> 在说我，不是叫我。
        """
        if msg.type not in (MSG_TEXT, MSG_APPMSG):
            return False
        text = self._strip_group_noise(msg.content or "")
        for name in self._group_self_names(agent):
            if not name:
                continue
            if text.strip() == name:
                return False   # 单叫一个名字很可能是在回答"骂谁"，不是喊我
            if "@" + name in text:
                return True
            if _re.match(r"^(?:帮我|帮忙)?(?:骂|怼|打|说|夸|找|叫|问)\s*" + _re.escape(name), text):
                return False
            if text.startswith(name):
                return True
        return False

    @staticmethod
    def _at_ids(msg) -> list:
        """这条群消息 @ 了谁（atuserlist 原始 wxid 列表，CDATA 可选）。"""
        xml = getattr(msg, "xml", "") or ""
        m = _re.search(r"atuserlist>(?:<!\[CDATA\[)?([^<]+?)(?:\]\]>)?</atuserlist", xml)
        if not m:
            return []
        return [u.strip() for u in m.group(1).split(",") if u.strip()]

    def _group_message(self, msg):
        """群消息路由：只回被@（atuserlist 硬信号）、主人说的、或正文点名我的；
        其他人静默（免刷屏）。文本去掉「wxid: 」前缀与@称呼，换成「[昵称] 」
        标注谁在说。群内一律文字回复（不镜像语音）；群主动由引擎参数把关。"""
        mention_only = bool(getattr(self.cfg.wechat, "group_mention_only", True))
        sender = msg.sender or ""
        agent = self.pool.agents.get(msg.roomid)
        is_owner = bool(self.state.owner_wxid) and sender == self.state.owner_wxid
        # @ 懒重查：轮询瞬间 source 常还没写入（atuserlist 竞争，实测群友 @阿满
        # 三连全被当旁听），到处理时早已写好——带 @ 的消息回读 source 重判一次
        if not getattr(msg, "at_me", False) and "@" in (msg.content or ""):
            reread = getattr(self.channel, "_reread_col", None)
            at_check = getattr(self.channel, "_at_me", None)
            if reread and at_check:
                try:
                    src = reread(msg, "source")
                    if src and at_check(src, self.channel.get_self_wxid() or ""):
                        msg.at_me = True
                except Exception:  # noqa: BLE001
                    pass
        # 正文点名不在这里判：交给 _group_overhear 记 mention 硬信号，
        # 由 chime_check 绕开冷却/攒批——被点名必须接（见 chime_check 注释）
        # 正文点名（"阿满，…" / "阿满 这波…"）= 软点名：走正常回复路径（攒批 + 静默窗），
        # 不再让插嘴路径单独再接一遍——两条路都接就是"疯狂发消息"（2026-10-28 实测：
        # 群里每句话都以 阿满 开头，插嘴对每条消息各接一次，几分钟刷出十几条）。
        self_wxid = self.channel.get_self_wxid() or ""
        at_ids = self._at_ids(msg)
        at_me_hard = bool(getattr(msg, "at_me", False)) or (bool(self_wxid) and self_wxid in at_ids)
        at_others = [i for i in at_ids if i and i != self_wxid]
        at_names = []
        for _id in at_others:
            _nk = (self.channel.nickname_of(_id) if hasattr(self.channel, "nickname_of") else "") or ""
            at_names.append(_nk or _id[-6:])
        msg.at_me = at_me_hard
        msg._at_names = at_names
        addressed = self._group_addressed_to_me(msg, agent)
        # 主人在 @ 别人时不是在跟 AI 说话：不命中、不执行（2026-10-28 实测：
        # "你是大肥鱼，你不要说话" 被当成对 AI 的指令，AI 自己闭嘴了）
        # 主人在"引用某个群友"的回复里，通常是在跟那个人说话（2026-10-29 实测：
        # 引用大肥鱼的话去回他，被当成"主人在跟我说"，AI 抢答了）
        _txt = msg.content or ""
        quoted_other = ("[引用" in _txt and "引用 你自己" not in _txt
                        and "引用 我：" not in _txt)
        # 不引用也能看出来：看这个群里上一条是谁说的。上一条是群友 -> 主人多半在回他；
        # 上一条是 AI -> 主人在接着跟 AI 说（2026-10-29 反馈：跳过 AI 的消息直接回群友）
        prev_sender, prev_speaker = "", ""
        if agent is not None and hasattr(agent, "conn"):
            try:
                _row = agent.conn.execute(
                    "SELECT sender, speaker FROM messages WHERE role_id=?"
                    " ORDER BY id DESC LIMIT 1", (agent.role_id,)).fetchone()
                if _row:
                    prev_sender = _row["sender"] or ""
                    prev_speaker = _row["speaker"] or ""
            except Exception:  # noqa: BLE001  查不到就按老逻辑
                pass
        reply_to_member = (is_owner and not at_me_hard and not addressed
                           and prev_sender == "user" and prev_speaker
                           and prev_speaker != self.state.owner_wxid)
        # 主人发言不再"必回"：主人跟群友聊天时，AI 每条都插一句自说自话
        # （2026-10-29 AI技术群实测：主人和 marblue/AW 聊工作，AI 每条都接
        # "活儿全压研发头上""我才是被调用的那个"——观感是"你说你的我说我的"）。
        # 主人消息也要有"是对我说"的信号才回复：
        # @我 / 叫我的名字 / 引用我的话 / 接着我上一句（上条是 AI 且带"你/问号"）/
        # 打招呼（在吗/人呢/哈喽）/ 指令句式（用/发/查/看/试试…开头，如"用mcp"）。
        # 否则旁听，由插嘴闸门按需决定凑不凑热闹。
        quoted_bot = ("[引用" in _txt and ("引用 你自己" in _txt
                                            or "引用 我：" in _txt))
        _norm = self._strip_group_noise(_txt)
        _hail = _re.search(r"在吗|在么|在不在|人呢|干啥呢|干嘛呢|哈喽|你好|出来",
                           _norm)
        # 指令句式：对 AI 的直接指令常没有"你"和问号——"用mcp""发我看看"
        # （2026-10-29 实测：主人回"用mcp"被当旁听，机器人不吭声了）
        _imperative = _re.match(
            r"^(?:用|发|查|搜|看|听|说|讲|写|画|换|改|调|修|搞|做|来|再|别|不要"
            r"|给我|帮我|把|重发|试试|念|唱)", _norm)
        # 接话词：对话中的延续指令（"继续吧""还有呢""分点一口气说完吧"）都是
        # 对我说的，不能靠插嘴兜底——插嘴只出短句、还会被长度兜底砍半（实测）。
        # 用"短消息 + 含接话词"判断，不要求开头命中（续讲的说法太随性）。
        _continuation = (len(_norm) <= 15 and bool(_re.search(
            r"继续|接着|还有|往下|展开|详细|讲完|说完|一口气|分点|再来|多讲|多说"
            r"|从头|捋完|都讲|都写", _norm)))
        owner_to_me = (at_me_hard or addressed or quoted_bot
                       or (prev_sender == "agent"
                           and ("你" in _norm or "？" in _norm or "?" in _norm))
                       or bool(_hail) or bool(_imperative) or bool(_continuation))
        if is_owner and (not owner_to_me
                         or (quoted_other and not at_me_hard and not addressed)):
            hit = False
        else:
            hit = (at_me_hard or (is_owner and not at_others) or not mention_only
                   or addressed)
        # 频率素材对所有群消息记账（插嘴触发只数"没被回复的旁听"，回复不攒门）
        label = ""
        nick_getter = getattr(self.channel, "nickname_of", None)
        if nick_getter and sender:
            label = nick_getter(sender)
        self._group_last[msg.roomid] = (sender, label)
        self._group_note_activity(msg.roomid, overheard=not hit)
        if not hit:
            self.log(f"群 {msg.roomid}：旁听 {sender[-8:]}（未@我，静默等插嘴决策）")
            self._group_overhear(msg, sender, role="owner" if is_owner else "member")
            return
        if msg.type not in (MSG_TEXT, MSG_IMAGE, MSG_VOICE, 47, MSG_APPMSG):
            return   # 10000 系统提示等
        if msg.type == MSG_TEXT:
            # 角色不写进正文（正文会发给用户/被模型抄进回复），只放 meta
            _raw = (msg.content or "").strip()
            msg.content = self._strip_group_noise(msg.content or "")
            if not msg.content.strip():
                # 裸 @（"wxid_xxx:\n@小满\u2005"）：剥完只剩空。
                # 注意原文带 wxid 前缀，判断要先剥前缀再找 @——旧实现直接
                # _raw.startswith("@") 永远不成立，两条 @小满 被静默丢弃，
                # 看起来像机器人没启动（2026-10-29 家人群实测）。
                _head, _sep, _rest = _raw.partition(":")
                _pre = (_rest.lstrip()
                        if (_sep and 6 <= len(_head) <= 64
                            and all(c.isalnum() or c in "_-." for c in _head))
                        else _raw)
                if _pre.startswith("@"):
                    # 只@不说话：保留原样 "@小满" 作为用户消息（别伪造"对方说了
                    # 什么"的文本进历史/记忆），模型看到 @ 会自然应一声
                    msg.content = _pre.strip()
        peer = msg.roomid
        if msg.type == MSG_TEXT and not (msg.content or "").strip():
            return
        if not self._multi_ok(peer):
            return
        why = ("@我" if at_me_hard else
               "叫我" if addressed else
               ("创造者@别人" if (is_owner and at_others) else
                "创造者" if is_owner else "全量"))
        msg._from_owner = is_owner   # 创造者消息标记：用于记忆/关系归属，不做服从模式
        msg._speaker_name = label or (sender[-6:] if sender else "群友")
        msg._speaker_wxid = sender
        msg._speaker_role = "owner" if is_owner else "member"
        if (at_others or (is_owner and (quoted_other or reply_to_member)
                          and not at_me_hard and not addressed)):
            msg._addressee = "群聊"
            if reply_to_member and prev_speaker:
                msg._addressee = f"群聊（像是回 {prev_speaker}）"
        elif at_me_hard or is_owner or addressed:
            msg._addressee = "你"
        else:
            msg._addressee = "群聊"
        # 这条已经在回复路径应答了：把插嘴窗口清空并压住一小会儿，
        # 否则插嘴线程会拿同一段对话再答一遍（2026-10-28 实测：7 秒内两条同义回复）
        _w = self._group_watch.get(msg.roomid)
        if _w is not None:
            _w["oh"] = 0
            _w["suppress_until"] = time.time() + float(
                getattr(self.cfg.wechat, "group_chime_after_reply_seconds", 60))
        self.log(f"群 {peer}：命中（{why} type={msg.type}）")
        self._multi_message(msg, peer)

    def _group_note_activity(self, room: str, overheard: bool = False):
        """记一笔群活动（插嘴门控的频率素材：条数/时间/旁听数）。"""
        w = self._group_watch.get(room)
        if w is None:
            w = {"msgs": 0, "oh": 0, "mention": 0, "about_me": 0,
                 "last_ts": 0.0, "last_chime": 0.0, "day": 0, "chimes": 0}
            self._group_watch[room] = w
        w["msgs"] = int(w.get("msgs", 0)) + 1
        if overheard:
            w["oh"] = int(w.get("oh", 0)) + 1
        w["last_ts"] = time.time()

    def chime_check(self, room: str, now: float = None) -> bool:
        """群消息的"要不要接"判断入口：只要有人说了一句，就让审核过一遍。

        判断分两段（模型第一件事就是答这个）：
        1) 这句话是不是在跟我说话（叫名字/用"你"指我/接着我上一句问/明显问我懂的事）
           → 是：必须回，不受冷却和日限约束（被点名不回是冷暴力）；
        2) 不是对我说话：再判断值不值得插一句 → 受冷却（默认 120s）+ 日限约束。

        门控只负责"省 token"和"防刷屏"，不再替模型做"是不是叫我"的判断：
        旧逻辑用正则判点名，漏一个就整段静音（2026-10-28 实测"只回主人"）。"""
        now = time.time() if now is None else now
        w = self._group_watch.get(room)
        if not w:
            return False
        # 刚在回复路径答过这个群：这段时间不判、不接（防一条消息两个回复）
        if now < float(w.get("suppress_until", 0) or 0):
            return False
        directed = int(w.get("mention", 0)) > 0
        about_me = int(w.get("about_me", 0)) > 0
        # 有一条"没人接的话"就值得判一次（不再要求攒够两条）
        if int(w.get("oh", 0)) < 1:
            return False
        gap = now - float(w.get("last_ts", 0))
        if gap < 4.0 or gap > 240.0:   # 没说完不判；话题冷掉不翻旧账
            return False
        if now - float(w.get("last_eval", 0)) < 3.0:
            return False               # 同一轮不重复评估（省 token）
        hour = int(now // 3600)
        if w.get("eval_hour") != hour:
            w["eval_hour"], w["evals"] = hour, 0
        cap = int(getattr(self.cfg.wechat, "group_chime_max_evals_per_hour", 40) or 0)
        if cap and int(w.get("evals", 0)) >= cap and not directed:
            return False               # 判断配额用完：只留"点名"这条硬路
        # 刚在这个群发过话：别紧接着自己再插一句（只防同一条接两遍，不是封口）
        quiet_after = float(getattr(self.cfg.wechat, "group_chime_after_reply_seconds", 60))
        if now - float(self._last_send.get(room, 0) or 0) < quiet_after:
            return False
        # 注意：这里不再用冷却拦"判断"——先让模型判是不是在跟我说话，
        # 只有判定为"自发插话"时才在发之前按冷却丢弃（见下方发送段）。
        day = time.localtime(now).tm_yday
        if w.get("day") != day:
            w["day"], w["chimes"] = day, 0
        if not directed and int(w.get("chimes", 0)) >= int(
                getattr(self.cfg.wechat, "group_chime_daily_max", 12)):
            w["msgs"] = w["oh"] = 0
            return False
        w["msgs"] = w["oh"] = 0          # 本轮已评估，重开窗口
        w["mention"] = 0
        w["about_me"] = 0
        w["evals"] = int(w.get("evals", 0)) + 1
        w["last_eval"] = now
        agent = self.pool.agents.get(room)
        if agent is None or not hasattr(agent, "chime_in"):
            return True
        if hasattr(agent, "group_self_names"):
            agent.group_self_names = self._group_self_names(agent)
            agent.group_owner_name = self._group_owner_name()
        try:
            rows = agent.conn.execute(
                "SELECT sender, content, meta, speaker, speaker_role, created_at"
                " FROM messages WHERE role_id=? ORDER BY id DESC LIMIT 8",
                (agent.role_id,)).fetchall()
            # 对话对象判断复用（meta 在旁听入库时已算好）：最新消息 @ 了别人
            # = 定向对话，AI 不插嘴。旧逻辑只看"有没有没人接的话"，没消费这个
            # 字段——2026-10-29 实测：主人@marblue"到点走不"，插嘴接了
            # "走啊，说好05就05"，看起来像没@也乱回。
            aimed_at_others = False
            try:
                _latest = json.loads(rows[0]["meta"] or "{}") if rows else {}
            except Exception:  # noqa: BLE001
                _latest = {}
            if _latest.get("at_names"):
                aimed_at_others = True
                # @别人但不一定是两人对话：带"大家/各位/群里…"这类广播信号的
                # 可能本来就是发给大家看的——放行给模型语义判断；其余跳过。
                _ltxt = (rows[0]["content"] or "") if rows else ""
                _broadcast = bool(_re.search(
                    r"大家|各位|你们|咱们|群里|全体|有没有人|谁知道|所有人都", _ltxt))
                if not _broadcast:
                    w["msgs"] = w["oh"] = 0
                    w["mention"] = 0
                    w["about_me"] = 0
                    self.log(f"群 {room}：最新消息@的是别人，跳过插嘴")
                    return True
                self.log(f"群 {room}：@别人的广播型消息，交给插嘴判断")
            # 近 3 条里有人 @ 别人（非广播型、3 分钟内）：这是他们之间的对话，
            # 紧随其后的接话（"坠机了吗"）也属于那场对话，不插嘴。
            # （2026-10-29 23:50 实测：@深情的大肥鱼 "还在吗" 之后的追问被
            # chime 当成问自己，回了"在呢，信号好好的"）
            _skip_at_other = False
            for _r in rows[:3]:
                try:
                    _m = json.loads(_r["meta"] or "{}")
                except Exception:  # noqa: BLE001
                    _m = {}
                if _m.get("at_names"):
                    _t = _r["content"] or ""
                    _broad = bool(_re.search(
                        r"大家|各位|你们|咱们|群里|全体|有没有人|谁知道|所有人都", _t))
                    try:
                        _age = time.time() - float(_r["created_at"] or 0)
                    except (TypeError, ValueError):
                        _age = 0.0
                    if not _broad and _age <= 180:
                        _skip_at_other = True
                    break
            if _skip_at_other:
                w["msgs"] = w["oh"] = 0
                w["mention"] = 0
                w["about_me"] = 0
                self.log(f"群 {room}：近几条在@别人，跳过插嘴")
                return True
            _now_ts = time.time()
            def _line(r):
                c = (r["content"] or "")[:80]
                if len(r["content"] or "") > 80:
                    c += "…"   # 预算内裁剪要带省略号：审核模型别把半句当整句
                # 时间标签：5 分钟前的消息只作背景。实测病灶（2026-10-29）：
                # 20 分钟前的抽奖玩笑被当成当前话题，接在 marblue 的新消息上；
                # 没有时间锚，模型会把旧梗当"刚发生"。
                try:
                    age = _now_ts - float(r["created_at"] or 0)
                except (TypeError, ValueError):
                    age = 0.0
                label = f"[{int(age // 60)}分钟前]" if age >= 300 else ""
                if r["sender"] != "user":
                    return label + "你：" + c
                try:
                    m = json.loads(r["meta"] or "{}")
                except Exception:  # noqa: BLE001
                    m = {}
                role = r["speaker_role"] or m.get("speaker_role") or ""
                name = r["speaker"] or m.get("speaker") or ""
                if role == "owner":
                    return f"{label}[创造者 {name}]：{c}"
                if role == "member":
                    return f"{label}[群友 {name}]：{c}"
                return label + "群友：" + c
            dialogue = [_line(r) for r in reversed(rows)]
            observe = self._observe_chat(room)
            text = agent.chime_in(dialogue, now=now,
                                  self_names=self._group_self_names(agent),
                                  directed=directed, about_me=about_me,
                                  owner_name=self._group_owner_name(),
                                  aimed_at_others=aimed_at_others,
                                  observe=observe)
        except Exception as e:  # noqa: BLE001
            self.log(f"插嘴决策失败 {room}：{e!r}")
            return True
        if text:
            if observe:
                # 观察模式：记录“本来会接什么”，不发送、不写 messages；
                # 仍推进 last_chime，避免同一批消息每 5 秒重复判。
                w["last_chime"] = now
                self.log(f"群 {room}：observe 未发送 -> {text[:40]}")
                return True
            meta = getattr(agent, "last_chime_meta", {}) or {}
            to_me = bool(meta.get("to_me")) or directed
            # 发送前再确认一次：判断耗时期间可能已经答过这条了
            # 判断耗时期间可能已经答过这条了：用墙钟复核（now 是测试可注入的时钟）
            ref = now if abs(time.time() - now) > 5 else time.time()
            sent_at = float(self._last_send.get(room, 0) or 0)
            if sent_at and ref - sent_at < quiet_after:
                self.log(f"群 {room}：刚发过话，丢掉重复的接话")
                return True   # 判过了，只是不发
            # 以库为准再查一次：**最新一条用户消息**在"本条 chime 落库之前"
            # 是否已经被别的 agent 消息回复过——防 respond 与 chime 同句双发。
            # 注意必须排除本条 chime 自己：main 在返回文本前已把它记进库，
            # 旧实现按"最后一条是不是 agent"判断，会把每条 chime 自己拦住
            # （2026-10-29 23:38 实测：chime 全被自己误杀，一句都发不出去）。
            try:
                _u = agent.conn.execute(
                    "SELECT id FROM messages WHERE role_id=? AND sender='user'"
                    " ORDER BY id DESC LIMIT 1", (agent.role_id,)).fetchone()
                _u_id = int(_u["id"]) if _u else 0
                _me = agent.conn.execute(
                    "SELECT id FROM messages WHERE role_id=? AND sender='agent'"
                    " AND content=? ORDER BY id DESC LIMIT 1",
                    (agent.role_id, text)).fetchone()
                _me_id = int(_me["id"]) if _me else 2 ** 62
                _replied = agent.conn.execute(
                    "SELECT COUNT(*) c FROM messages WHERE role_id=? AND sender='agent'"
                    " AND id > ? AND id < ?",
                    (agent.role_id, _u_id, _me_id)).fetchone()["c"]
            except Exception:  # noqa: BLE001
                _replied = 0
            if _replied:
                self.log(f"群 {room}：最新消息已有回复，跳过插嘴")
                return True
            # 冷却只对"硬点名"（directed：名字/别名命中）豁免；模型猜的
            # "在跟我说话"（to_me）同样纳入冷却/日限——否则旁听对话被模型
            # 判成 to_me 就每条都接，1 分钟内连发数条（2026-10-29 22:40 实测）
            if not directed:
                if now - float(w.get("last_chime", 0)) < float(
                        getattr(self.cfg.wechat, "group_chime_cooldown_seconds", 120)):
                    self.log(f"群 {room}：插嘴还在冷却，这条不发")
                    return True   # 判过了，只是不发
                if int(w.get("chimes", 0)) >= int(
                        getattr(self.cfg.wechat, "group_chime_daily_max", 12)):
                    return True   # 判过了，日限用完了
                w["chimes"] = int(w.get("chimes", 0)) + 1
                self.log(f"群 {room}：AI 决定插嘴"
                         f"{'（判为对我说话）' if to_me else ''} -> {text[:40]}")
            else:
                self.log(f"群 {room}：被点名 -> 回 {text[:40]}")
            w["last_chime"] = now
            self._send_reply(text, room, self_initiated=True)
        else:
            self.log(f"群 {room}：AI 决定继续潜水")
        return True

    def _group_watch_loop(self):
        """群活动看门狗：静默窗过后问一次"要不要插嘴"。"""
        while not self.stop.wait(5.0):
            if not getattr(self.cfg.wechat, "group_chime", True):
                continue
            for room in list(self._group_watch):
                if self.stop.is_set():
                    return
                try:
                    self.chime_check(room)
                except Exception as e:  # noqa: BLE001
                    self.log(f"插嘴检查异常 {room}：{e!r}")

    def _group_overhear(self, msg, sender: str, role: str = "member"):
        """旁听：群友没@我的话不回（免刷屏），但记进群 Agent 的消息表——
        被@时接得上前文、主动引擎有新鲜话题可聊。只旁听文字，不调 LLM。"""
        if msg.type != MSG_TEXT:
            return
        text = self._strip_group_noise(msg.content or "")
        if not text:
            return
        label = ""
        nick_getter = getattr(self.channel, "nickname_of", None)
        if nick_getter and sender:
            label = nick_getter(sender)
        label = label or (sender[-6:] if sender else "群友")
        agent = self.pool.get(msg.roomid)
        if agent is None:
            return
        # 每个人都要有卡：旁听消息不经过 on_user_message，也要建/更新人物卡
        if hasattr(agent, "persons"):
            try:
                import time as _pt
                pid = agent.persons.ensure(sender, label, "group_member", _pt.time())
                agent.persons.observe(pid, text, from_owner=False,
                                      source_message_id=getattr(msg, "id", None),
                                      now=_pt.time())
            except Exception as e:  # noqa: BLE001  人物卡失败不影响旁听
                self.log(f"旁听人物卡失败：{e!r}")
        # 直接叫我才算"必须接"；只是提到我/让人骂我只记 about_me。
        direct = self._group_addressed_to_me(msg, agent)
        mentioned = self._group_mentions_me(msg, agent)
        if not direct:
            # 旁听/未点名消息只记 observe；真正回复的 respond 由 _multi_chat 记，
            # 这里再记一次会被 1.5s 去重当成重复投递，反而把回复吞掉。
            try:
                agent.triggers.log(msg.roomid, sender or "", "text", "observe",
                                   "群聊旁听", False, text)
            except Exception as e:  # noqa: BLE001  审计失败不影响旁听
                self.log(f"触发审计写入失败：{e!r}")
        if direct or mentioned:
            w = self._group_watch.setdefault(msg.roomid, {
                "msgs": 0, "oh": 0, "mention": 0, "about_me": 0,
                "last_ts": 0.0, "last_chime": 0.0, "day": 0, "chimes": 0})
            if direct:
                w["mention"] = int(w.get("mention", 0)) + 1
            if mentioned and not direct:
                w["about_me"] = int(w.get("about_me", 0)) + 1
            w["oh"] = max(int(w.get("oh", 0)), 1)
            w["last_ts"] = time.time()
            self._group_watch[msg.roomid] = w
        try:
            import time as _t
            # 防刷屏上限 4000 字（此前 text[:200] 把长消息腰斩入库——之后控制台/
            # 插嘴上下文/滚动摘要读到的全是残句）。真截断必须带显式标记，
            # 模型才不会把半句当完整内容理解。
            stored = (text if len(text) <= 4000
                      else text[:4000] + "…（超长已截断）")
            with agent._lock:
                agent.conn.execute(
                    "INSERT INTO messages(role_id, sender, content, meta, created_at,"
                    " speaker, speaker_role) VALUES(?,?,?,?,?,?,?)",
                    (agent.role_id, "user", stored,
                     json.dumps({"overheard": True, "speaker": label,
                                 "speaker_role": role,
                                 "addressee": ("你" if direct else "群聊"),
                                 "at_names": list(getattr(msg, "_at_names", []) or []),
                                 "about_me": bool(mentioned and not direct)},
                                ensure_ascii=False),
                     _t.time(), label, role))
                agent.conn.commit()
        except Exception as e:  # noqa: BLE001  旁听失败不影响任何事
            self.log(f"旁听写入失败：{e!r}")

    @staticmethod
    def _strip_group_noise(text: str) -> str:
        """群文本 =「wxid_xxx:@阿满 内容」：去掉发送者前缀与前导@称呼。
        前缀分隔符实测有 "": "" 和 "":\n"" 两种，都要剥。
        （微信的@后面跟 U+2005 四分之一空格，不是普通空格。）"""
        head, sep, rest = (text or "").partition(":")
        if sep and 6 <= len(head) <= 64 and head and \
                all(c.isalnum() or c in "_-." for c in head):
            text = rest.lstrip()
        sep5 = chr(0x2005)
        for _ in range(2):
            if text.startswith("@"):
                cut = text.find(sep5)
                if 1 <= cut <= 32:
                    text = text[cut + 1:]
                elif 1 <= text.find(" ") <= 32:
                    text = text[text.find(" ") + 1:]
                else:
                    # 没有任何分隔符：整条就是一个裸 @ 昵称（"@小满"）——
                    # 剥成空串，避免昵称变正文让模型拿旧话题凑答案
                    # （2026-10-29 实测：只@不说话，回了上一个话题的股市分析）
                    if _re.fullmatch(r"@[^\s，。！？!?、,.]+", text):
                        text = ""
                    break
                text = text.lstrip(sep5 + " ")
            else:
                break
        return text.strip()

    def _multi_message(self, msg, peer):
        if not self.async_dispatch:
            self._process_batch([msg], peer)
            return
        with self._peer_lock:
            q = self._peer_queues.get(peer)
            if q is None:
                q = queue.Queue()
                self._peer_queues[peer] = q
                threading.Thread(target=self._peer_worker, args=(q, peer),
                                 daemon=True, name=f"wx-u-{peer[-6:]}").start()
        q.put(msg)

    def _peer_worker(self, q, peer):
        while not self.stop.is_set():
            try:
                first = q.get(timeout=1.0)
            except queue.Empty:
                with self._peer_lock:
                    if q.empty() and self._peer_queues.get(peer) is q:
                        del self._peer_queues[peer]
                        return
                continue
            batch = self._collect_burst(q, first)
            try:
                self._process_batch(batch, peer)
            except Exception as e:  # noqa: BLE001
                self.log(f"{peer} 处理异常：{e!r}")

    def _collect_burst(self, q, first):
        """连发收集：人一句话常拆成几条气泡连发——静默 batch_quiet_seconds 秒
        视为说完（总窗封顶 12s、条数封顶 6），攒成一次理解、一次回复。"""
        batch = [first]
        deadline = time.time() + 12.0
        while len(batch) < 6 and time.time() < deadline:
            try:
                batch.append(q.get(timeout=self.cfg.wechat.batch_quiet_seconds))
            except queue.Empty:
                break
        return batch

    def _process_batch(self, batch, peer):
        """把一次突发的多条消息合成一轮：文字/语音转写/链接拼成一段（
 分隔），
        图片并入同一次视觉理解；was_voice 只要有语音就置位（镜像回语音）。"""
        texts, images, was_voice = [], [], False
        from_owner = ((peer == self.state.owner_wxid) if not str(peer).endswith("@chatroom")
                      else any(getattr(m, "_from_owner", False) for m in batch))
        speaker_role = "owner" if from_owner else "member"
        speaker_name = ""
        speaker_wxid = ""
        addressee = "群聊"
        at_names = []
        for m in batch:
            if getattr(m, "_at_names", None):
                at_names = list(m._at_names)
            if getattr(m, "_speaker_role", "") == speaker_role:
                speaker_name = getattr(m, "_speaker_name", "") or speaker_name
                speaker_wxid = getattr(m, "_speaker_wxid", "") or speaker_wxid
                addressee = getattr(m, "_addressee", addressee) or addressee
        if not str(peer).endswith("@chatroom"):
            speaker_name = speaker_name or "你"
            speaker_wxid = speaker_wxid or peer
        # "再看看/再试一次"：重取该会话最近一张图/表情（媒体落盘慢于消息行的
        # 竞争，当场取失败很常见——用户喊一声就重试，实测重放即可成功）
        combined_probe = chr(10).join((m.content or "")[:60] for m in batch)
        if not any(m.type in (MSG_IMAGE, 47) for m in batch) and _re.search(
                r"再看看|再看一[遍次]|再试一[次下]|重新看|再瞅瞅|再看下|在看下|看一下图|图呢|怎么看不了|为啥看不了", combined_probe):
            lm = getattr(self.channel, "last_media_msg", None)
            if lm is not None:
                try:
                    m2 = lm(peer)
                    if m2 is not None:
                        import base64 as _b64
                        path = (self.channel.fetch_emoji(m2) if m2.type == 47
                                else self.channel.fetch_image(m2))
                        path = path if isinstance(path, (list, tuple)) else ([path] if path else [])
                        for p3 in [x for x in path if x and os.path.exists(x)][:1]:
                            with open(p3, "rb") as f:
                                raw = f.read()
                            images.append({"b64": _b64.b64encode(raw).decode(),
                                           "mime": _image_mime(raw)})
                            self.log(f"重看 {peer} 最近媒体 -> {os.path.basename(p3)}")
                except Exception as e:  # noqa: BLE001  重看失败走原对话
                    self.log(f"重看失败：{e!r}")
        for msg in batch:
            if msg.type == MSG_TEXT:
                t = (msg.content or "").strip()
                if t:
                    texts.append(t)
            elif msg.type == MSG_VOICE:
                t = self._fetch_voice_text(msg, peer)
                if t is not None:
                    texts.append(t)
                    was_voice = True
            elif msg.type == MSG_IMAGE:
                imgs = self._fetch_image_b64(msg, peer)
                if imgs:
                    images.extend(imgs)
            elif msg.type == MSG_APPMSG:
                quote = self._parse_quote(msg, self.channel.get_self_wxid() or "")
                if quote:
                    texts.append(quote)
                    qimg = self._quote_image(msg)   # 引用图片：原图一并给视觉
                    if qimg:
                        images.extend(qimg)
                else:
                    link = self._parse_link(msg)
                    if link:
                        texts.append(link)
            elif msg.type == 47:  # 表情包：视觉理解（CDN 原图）；拿不到至少告知模型
                fetch_emoji = getattr(self.channel, "fetch_emoji", None)
                path = None
                if fetch_emoji is not None:
                    try:
                        path = fetch_emoji(msg)
                    except Exception as e:  # noqa: BLE001
                        self.log(f"表情获取失败：{e!r}")
                paths = path if isinstance(path, (list, tuple)) else ([path] if path else [])
                paths = [p3 for p3 in paths if p3 and os.path.exists(p3)][:3]
                if paths:
                    import base64
                    total = 0
                    frames_b64 = []
                    for p3 in paths:
                        with open(p3, "rb") as f:
                            raw = f.read()
                        total += len(raw)
                        frames_b64.append(base64.b64encode(raw).decode())
                    # 一个动图表情包 = 一条图片消息：首帧为主图、其余帧挂采样组，
                    # 视觉端一次调用合并理解——模型看到的是"一张动图"，
                    # 不再是 N 张独立图片（首/中/尾抽样本就是为了躲空白首帧）
                    images.append({"b64": frames_b64[0], "mime": "image/jpeg",
                                   "animated": True,
                                   "frames": frames_b64[1:]})
                    self.log(f"收表情包 动图采样 {len(frames_b64)} 帧"
                             f" {total // 1024}KB -> 一次视觉理解")
                else:
                    texts.append("[对方发了一个表情包，画面暂时没加载出来]")
        combined = "\n".join(texts).strip()
        if not combined and not images:
            return
        if was_voice:
            combined = f"[语音] {combined}"
        voice_request = bool(VOICE_WANT_RE.search(combined))
        self._multi_chat(combined, images or None, was_voice, peer,
                          voice_request=voice_request, from_owner=from_owner,
                          speaker_name=speaker_name, speaker_role=speaker_role,
                          speaker_wxid=speaker_wxid, addressee=addressee,
                          at_names=at_names)

    def _recover_unanswered(self) -> int:
        """桥重启补处理：上一个进程已收下、但回复还没生成的消息。

        只补最近窗口内（默认 600s，可用 AGENT_RECOVER_SECONDS 调大）且确实
        没有 agent 回复的记录；已经答过的、太老的、空内容的一律跳过。
        """
        if self.pool is None:
            return 0
        try:
            window = float(os.environ.get("AGENT_RECOVER_SECONDS") or 600)
        except ValueError:
            window = 600.0
        cutoff = self._started_at - max(30.0, window)
        recovered = 0
        for wxid, agent in list(self.pool.agents.items()):
            if not self._chat_enabled(wxid):
                continue
            try:
                rows = agent.conn.execute(
                    "SELECT id, content, speaker, speaker_role, meta, created_at"
                    " FROM messages WHERE role_id=? AND sender='user' AND created_at>=?"
                    " ORDER BY id ASC LIMIT 5",
                    (agent.role_id, cutoff)).fetchall()
            except Exception as e:  # noqa: BLE001  单个库读失败不影响其他会话
                self.log(f"补处理查询失败 {wxid}：{e!r}")
                continue
            for row in rows:
                text = str(row["content"] or "").strip()
                if not text:
                    continue
                try:
                    # 只要这条用户消息之后有过任何 agent 消息（含插嘴/主动），
                    # 就认为已经被接住；宁可少补，不能重启后补出第二条。
                    replied = agent.conn.execute(
                        "SELECT 1 FROM messages WHERE role_id=? AND sender='agent'"
                        " AND (reply_to=? OR created_at>=?) LIMIT 1",
                        (agent.role_id, row["id"], row["created_at"])).fetchone()
                except Exception:  # noqa: BLE001  查不到就保守跳过
                    replied = True
                if replied:
                    continue
                try:
                    meta = json.loads(row["meta"] or "{}")
                except Exception:  # noqa: BLE001
                    meta = {}
                role = row["speaker_role"] or meta.get("speaker_role") or "member"
                self.log(f"启动补处理：{wxid} #{row['id']} {text[:30]}")
                try:
                    self._multi_chat(
                        text, None, False, wxid,
                        from_owner=(role == "owner"),
                        speaker_name=row["speaker"] or "",
                        speaker_role=role,
                        speaker_wxid=meta.get("speaker_wxid", "") or "",
                        addressee=meta.get("addressee")
                        or ("群聊" if str(wxid).endswith("@chatroom") else ""),
                        at_names=meta.get("at_names") or [],
                        replay_message_id=int(row["id"]))
                    recovered += 1
                except Exception as e:  # noqa: BLE001  补处理失败只记录
                    self.log(f"启动补处理失败 {wxid} #{row['id']}：{e!r}")
                if recovered >= 3:
                    return recovered
        if recovered:
            self.log(f"启动补处理完成：{recovered} 条")
        return recovered

    def _multi_chat(self, text, images, was_voice, peer, voice_request=False,
                    from_owner=True, speaker_name="", speaker_role="",
                    speaker_wxid="", addressee="群聊", at_names=None,
                    replay_message_id=None):
        """多用户主路径：独立 Agent 处理 -> 即时投递（回复/主动附加/手部图片/语音镜像）。
        voice_request：对方明确要语音（代码级意图识别，模型信不信自己会说话都拦不住）。"""
        t0 = time.time()
        self.log(f"← {peer}{'[语音]' if was_voice else '[图]' if images else ''}: "
                 f"{(text or '')[:30]}")
        agent = self.pool.get(peer)
        if agent is None:
            self.log(f"多用户：已达上限（{self.cfg.wechat.max_users} 人），忽略 {peer}")
            return
        if peer.endswith("@chatroom"):
            # 群聊身份注入：名字和创造者是稳定事实，不能让群友的"你不是阿满"带偏
            agent.group_self_names = self._group_self_names(agent)
            agent.group_owner_name = self._group_owner_name()
        # 唱歌/画图等后台任务好了通过回调直接发到这个会话（不能等本轮回复）
        try:
            agent.hands.song_sender = lambda info, _peer=peer: self._send_song_result(_peer, info)
            agent.hands.async_sender = lambda info, _peer=peer: self._send_async_result(_peer, info)
        except Exception as e:  # noqa: BLE001  老 Agent/无手部时不影响聊天
            self.log(f"后台任务回调注入失败 {peer}：{e!r}")
        self._notify_orphans(peer, agent)
        # 用户纠正“不是让你唱/搞错了”时，取消还在跑的翻唱，避免把错歌发出去
        try:
            if agent.hands.song_running() and re.search(
                    r"不是让你唱|不是让你点|我让你.*(给|让).*唱|理解岔|搞错"
                    r"|别唱了|不用唱了|取消.*(歌|翻唱)|不是这个意思", text or ""):
                n = agent.hands.cancel_song_jobs()
                if n:
                    self.log(f"取消翻唱任务 {n} 个（用户纠正）")
        except Exception as e:  # noqa: BLE001
            self.log(f"取消翻唱失败 {peer}：{e!r}")
        try:
            with self._inflight_lock:
                self._inflight += 1
            # 入站触发账本：重复投递（补收/重启回放）在进 Agent 前就截住。
            try:
                addressed = (not str(peer).endswith("@chatroom")) or bool(from_owner)
                fresh = agent.triggers.log(
                    peer, speaker_wxid or speaker_name or "", 
                    "mixed" if images else "text", "respond", "accepted",
                    addressed, text or "[图片]")
                if not fresh:
                    self.log(f"重复触发已忽略：{peer} {(text or '')[:30]}")
                    return
            except Exception as e:  # noqa: BLE001  账本失败不挡回复
                self.log(f"触发账本写入失败：{e!r}")
            result = agent.on_user_message(text, images=images, from_owner=from_owner,
                                           speaker=speaker_name,
                                           speaker_role=speaker_role,
                                           speaker_wxid=speaker_wxid,
                                           addressee=addressee,
                                           at_names=at_names,
                                           replay_message_id=replay_message_id)
        except Exception as e:  # noqa: BLE001
            self.log(f"多用户：{peer} 处理失败：{e!r}")
            return
        finally:
            with self._inflight_lock:
                self._inflight = max(0, self._inflight - 1)
        hand = ((result.get("meta") or {}).get("hand") or {})
        for img in hand.get("images") or []:
            try:
                code = self.channel.send_image(str(img), peer)
            except Exception as e:  # noqa: BLE001
                code = -1
                self.log(f"发图异常：{e!r}")
            self.log(f"发图 {'ok' if code == 0 else f'fail({code})'}: {img}")
            if code != 0:
                self._enqueue_media(peer, "image", {"path": str(img)},
                                    dedupe_key="image:%s|%s" % (peer, img))
        if result.get("addon"):
            self._send_sticker_reply(result["addon"], peer, agent, self_initiated=True)
        self._last_reply_at[peer] = time.time()
        reply = result.get("reply")
        if not reply:
            self.log(f"→ {peer}: (无回复 {time.time()-t0:.1f}s)")
            return
        self._last_reply_at[peer] = time.time()
        want_voice_override = False
        if reply.startswith("[语音]"):   # 模型按需点语音（对方明确要求时）
            reply = reply[4:].lstrip()
            want_voice_override = True
        if "[语音]" in reply:
            # 模型偶尔在段中插标记（尤其唱歌连发段）；无论走语音还是文字都剥掉，
            # 否则 TTS 会把标记念出来、文字气泡会露出协议符号
            reply = reply.replace("[语音]", "").strip()
        self.log(f"→ {peer} ({time.time()-t0:.1f}s): {reply[:40]}")
        # 语音回复：mirror/always 全场景生效（群聊也支持——对方在群里发语音就
        # 回语音；明确要求/模型标记照旧）。之前群聊仅限明确要求，按反馈放开
        want_voice = ((want_voice_override or voice_request)
                      or (self.cfg.wechat.voice_reply == "always"
                          or (was_voice and self.cfg.wechat.voice_reply == "mirror")))
        if voice_request:
            self.log(f"{peer} 明确要语音 -> 强制语音回复")
        # 表情包：回复里的 [sticker:情绪词] 换成真表情图（先发图，正文去掉标记）
        store = getattr(agent, "stickers", None)
        if store is not None and getattr(store, "enabled", False) and "[sticker" in reply:
            from .stickers import extract_stickers
            clean, paths = extract_stickers(reply, store)
            for pth in paths:
                try:
                    code = self.channel.send_image(pth, peer)
                    if code != 0 and pth.endswith(".gif"):
                        # 大 GIF 发不出去：回落静态缩略图（同一张贴纸的 PNG）
                        alt = pth[:-4] + ".png"
                        if os.path.exists(alt):
                            code = self.channel.send_image(alt, peer)
                            self.log(f"发表情：动图失败，改用静态图 {'ok' if code == 0 else code}")
                    self.log(f"发表情 {'ok' if code == 0 else f'fail({code})'}")
                except Exception as e:  # noqa: BLE001  发图失败不影响正文
                    self.log(f"发表情失败：{e!r}")
            reply = clean or reply
        if want_voice and not self._send_voice_reply(reply, peer):
            self._send_reply(reply, peer)  # 语音链路失败回落文字
        elif not want_voice:
            self._send_reply(reply, peer)

    def _write_health(self):
        """给控制台/健康检查写心跳；桥进程活着但 hook 掉线也能看出来。"""
        try:
            hook = bool(self.channel.is_login())
        except Exception:  # noqa: BLE001  登录态读不到不算桥死
            hook = None
        try:
            self_wxid = self.channel.get_self_wxid()
        except Exception:  # noqa: BLE001
            self_wxid = ""
        data = {
            "at": time.time(),
            "pid": os.getpid(),
            "channel": getattr(self.channel, "name", ""),
            "self": self_wxid or "",
            "hook": hook,
            "agents": len(self.pool.agents) if self.pool is not None else 0,
            "multi": bool(self.multi),
        }
        path = os.path.join(
            getattr(self.pool, "_base", None) or os.path.dirname(self.media_dir),
            "health", "bridge.json")
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, path)
        except Exception as e:  # noqa: BLE001  心跳写失败不影响收发
            self.log(f"健康心跳写入失败：{e!r}")

    def _health_loop(self):
        while not self.stop.wait(15.0):
            self._write_health()

    def _multi_ticker(self):
        """多用户调度：每个用户独立评估主动行为与反思（间隔 = tick.fast_seconds）。"""
        interval = max(15.0, float(getattr(self.cfg.tick, "fast_seconds", 60)))
        while not self.stop.wait(interval):
            groups_proactive = bool(getattr(self.cfg.wechat, "groups_proactive", True))
            for wxid, agent in list(self.pool.agents.items()):
                if not self._chat_enabled(wxid):
                    continue   # 启用范围外：不跑主动评估（省 token）
                if wxid.endswith("@chatroom") and not groups_proactive:
                    continue   # 群主动关闭时只回应被@
                self._tick_agent(wxid, agent)

    def _tick_agent(self, wxid: str, agent):
        """单用户的主动评估 + 反思。群发主动时 @ 最近说话人（免得像自言自语）。"""
        if wxid.endswith("@chatroom"):
            agent.group_self_names = self._group_self_names(agent)
            agent.group_owner_name = self._group_owner_name()
        # 自己 4 分钟内刚说过话就不主动：紧跟自己回复蹦罐头短语（实测
        # '乐，就这？'在回复后 24s 冒出）是抽风不是主动
        if time.time() - self._last_reply_at.get(wxid, 0) < 240:
            return
        # 桥刚启动不主动：刚重启就自己找话说，用户看到的是"一开机就刷屏"
        if time.time() - self._started_at < float(
                getattr(self.cfg.wechat, "proactive_warmup_seconds", 300)):
            return
        try:
            observe = self._observe_chat(wxid)
            result = agent.evaluate_proactive(observe=observe)
            if observe and (result or {}).get("observed"):
                item = result["observed"]
                self.log(f"observe 未发送 {wxid} L{item.get('level')} "
                         f"{item.get('intent')}：{str(item.get('text') or '')[:40]}")
                return
            sent = (result or {}).get("sent")
            if sent and sent.get("text"):
                text = sent["text"]
                if wxid.endswith("@chatroom"):
                    last = self._group_last.get(wxid)
                    if last and last[1]:
                        text = f"@{last[1]} {text}"
                self._send_text(text, wxid, self_initiated=True)
                self.log(f"主动 L{sent.get('level')} → {wxid}：{sent['text'][:30]}")
        except Exception as e:  # noqa: BLE001
            self.log(f"主动评估失败 {wxid}：{e!r}")
        try:
            agent.maybe_reflect()
        except Exception as e:  # noqa: BLE001
            self.log(f"反思失败 {wxid}：{e!r}")

    def _owner_ok(self, peer: str) -> bool:
        if self.state.owner_wxid:
            return peer == self.state.owner_wxid
        # 未绑定：第一个私聊的人成为唯一主人（配置 wechat.owner_wxid 可固定）
        self.state.owner_wxid = peer
        self.state.save()
        self.log(f"已绑定主人 wxid={peer}（config wechat.owner_wxid 可显式固定）")
        return True

    def _post_chat(self, text: str, images=None) -> dict:
        """POST /chat；后台故障时限频提示（不吞掉主人的消息体验）。"""
        try:
            return self.api.chat(text, images)
        except Exception as e:  # noqa: BLE001
            self.log(f"/chat 失败：{e!r}")
            now = time.time()
            if now - self._last_err_note > 600 and self.state.owner_wxid:
                self._last_err_note = now
                self._send_text("（我这边后台掉线了一小会儿，正在重连…）",
                                self.state.owner_wxid)
            return {}

    def _after_chat(self, result: dict, peer: str, was_voice: bool):
        """/chat 返回后的即时副作用：发手部产出的图片；登记语音回复意愿。"""
        hand = ((result.get("meta") or {}).get("hand") or {})
        for img in hand.get("images") or []:
            try:
                code = self.channel.send_image(str(img), peer)
            except Exception as e:  # noqa: BLE001
                code = -1
                self.log(f"发图异常：{e!r}")
            self.log(f"发图 {'ok' if code == 0 else f'fail({code})'}: {img}")
            if code != 0:
                self._enqueue_media(peer, "image", {"path": str(img)},
                                    dedupe_key="image:%s|%s" % (peer, img))
        if was_voice and self.cfg.wechat.voice_reply == "mirror":
            self._voice_pending += 1

    def _on_text(self, msg, peer):
        result = self._post_chat((msg.content or "").strip())
        if result:
            self._after_chat(result, peer, was_voice=False)

    def _on_image(self, msg, peer):
        images = self._fetch_image_b64(msg, peer)
        if images is None:
            return
        result = self._post_chat("", images)
        if result:
            self._after_chat(result, peer, was_voice=False)

    def _on_voice(self, msg, peer):
        text = self._fetch_voice_text(msg, peer)
        if text is None:
            return
        result = self._post_chat(f"[语音] {text}")
        if result:
            self._after_chat(result, peer, was_voice=True)

    def _on_link(self, msg, peer):
        link = self._parse_link(msg)
        if link is None:
            self.log(f"appmsg（非链接）跳过 msg.id={msg.id}")
            return
        result = self._post_chat(link)
        if result:
            self._after_chat(result, peer, was_voice=False)

    # ---------------- 投递路径（回复 + 主动消息共用，水位线去重） ----------------

    def bootstrap_watermark(self):
        """启动水位线：沿用持久化值并夹到当前最大 id（防 DB 重建后重播/漏发）。"""
        max_id = self.api.latest_message_id()
        saved = self.state.watermark
        self.state.watermark = max_id if saved <= 0 else min(saved, max_id)
        self.state.save()
        self.log(f"投递水位线：{self.state.watermark}（库内最大 id={max_id}）")

    def deliver_pending(self) -> int:
        """一个轮询周期：拉 /pending -> 逐条发微信 -> 推进水位线。返回发送条数。"""
        try:
            items = self.api.pending(self.state.watermark)
        except Exception as e:  # noqa: BLE001
            self.log(f"/pending 失败：{e!r}")
            return 0
        sent = 0
        for item in items:
            try:
                self._deliver_item(item)
                sent += 1
            except Exception as e:  # noqa: BLE001  单条失败仍推进，避免毒消息循环
                self.log(f"投递失败 #{item.get('id')}: {e!r}")
            self.state.watermark = max(self.state.watermark, int(item.get("id") or 0))
        if items:
            self.state.save()
        return sent

    def _deliver_item(self, item: dict):
        peer = self.state.owner_wxid
        content = (item.get("content") or "").strip()
        if not content or not peer:
            return
        if not self._chat_enabled(peer):
            return   # 启用范围外：服务端产生的主动消息也不投递
        is_reply = not (item.get("meta") or {}).get("proactive")
        want_voice = (self.cfg.wechat.voice_reply == "always"
                      or (is_reply and self._voice_pending > 0))
        if want_voice:
            if self._send_voice_reply(content, peer):
                if self._voice_pending > 0:
                    self._voice_pending -= 1
                return
            # 语音链路不可用/超长 -> 回落文字（意愿照样消费，不泄漏）
            if self._voice_pending > 0:
                self._voice_pending -= 1
        self._send_reply(content, peer, self_initiated=not is_reply)

    def _send_voice_reply(self, text: str, peer: str) -> bool:
        """按通道偏好发语音：wxhook=SILK 真气泡；wcf=mp3 文件卡片。
        成功 True；超长或链路失败 False（回落文字）。"""
        if len(text) > self.cfg.wechat.voice_max_chars:
            return False
        self.stop.wait(random.uniform(1.0, 2.5))   # 录语音前的小停顿
        try:
            audio = self._tts_bytes(text)
            if not isinstance(audio, (bytes, bytearray)) or not audio:
                return False
        except BridgeError as e:
            self.log(f"TTS 不可用，回落文字：{e}")
            return False
        if self.channel.voice_prefers == "silk":
            try:
                from .silk_audio import mp3_to_silk
                silk_path = os.path.join(
                    self.media_dir, "reply_%s.silk" % time.strftime("%Y%m%d_%H%M%S"))
                info = mp3_to_silk(audio, silk_path, work_dir=self.media_dir)
                code = self.channel.send_voice_silk(info["path"], peer, info["duration_ms"])
                self.log(f"语音气泡 {'ok' if code == 0 else f'fail({code})'}"
                         f"（{info['duration_ms']}ms/{info['bytes']}B）")
                return code == 0
            except Exception as e:  # noqa: BLE001  编码/发送失败都回落文字
                self.log(f"语音气泡失败，回落文字：{e!r}")
                return False
        # wcf：mp3 文件卡片（点开即播）
        path = os.path.join(self.media_dir,
                            "reply_%s.mp3" % time.strftime("%Y%m%d_%H%M%S"))
        with open(path, "wb") as f:
            f.write(audio)
        code = self.channel.send_file(path, peer)
        self.log(f"语音回复（文件卡片）{'ok' if code == 0 else f'fail({code})'}")
        return code == 0

    def _send_async_result(self, peer: str, info: dict):
        """慢手部完成后的新事件投递。

        图片：图片 + 一句固定人话；bilibili.watch 这类需要“讲内容”的任务：
        回 Agent 做第二次人格渲染，再由这里发出，避免系统摘要直接冒充她说话。
        """
        info = info or {}
        result = info.get("result") or {}
        action = str(info.get("action") or "")
        if not result.get("ok"):
            self.log(f"后台动作失败 {peer}/{action}：{str(result.get('summary'))[:160]}")
        if action == "image.draw":
            if result.get("ok"):
                img = result.get("image")
                if isinstance(img, str) and img and os.path.exists(img):
                    try:
                        code = self.channel.send_image(img, peer)
                    except Exception as e:  # noqa: BLE001
                        code = -1
                        self.log(f"后台图片发送失败：{e!r}")
                    if code != 0:
                        self._enqueue_media(peer, "image", {"path": img},
                                            dedupe_key="image:%s|%s" % (peer, img))
                    else:
                        self.log(f"后台图片发送 ok: {img}")
                self._send_text("画好了，给你", peer)
            else:
                self._send_text("这次没画成…等下再试", peer)
            return
        # bilibili.watch 等“内容型”后台任务：让 Agent 用当前人格再说一遍。
        agent = self.pool.agents.get(peer) if self.pool is not None else None
        text = ""
        if agent is not None and hasattr(agent, "on_task_result"):
            try:
                out = agent.on_task_result(info) or {}
                text = str(out.get("reply") or "").strip()
            except Exception as e:  # noqa: BLE001  二次渲染失败回落摘要
                self.log(f"任务结果二次渲染失败 {peer}：{e!r}")
        if not text:
            summary = str(result.get("summary") or "").strip()
            if result.get("ok") and summary:
                text = summary
            elif result.get("ok"):
                text = "刚才那件事做完了，不过结果没整理出来"
            else:
                text = "刚才那件事没弄成…等下再试"
        self.log(f"后台任务结果 → {peer}：{text[:40]}")
        self._send_reply(text, peer)

    def _deliver_song(self, peer: str, payload: dict, notice: bool = True) -> bool:
        """按 send_mode 把一首歌发出去；返回是否至少有一段成功。"""
        path = str(payload.get("path") or "")
        if not path or not os.path.exists(path):
            self.log(f"唱歌文件不存在 {peer}：{path!r}")
            return False
        duration_ms = int(payload.get("duration_ms") or 0)
        mode = str(getattr(self.cfg.song_gen, "send_mode", "bubble")).lower()
        max_send_ms = int(float(getattr(self.cfg.song_gen, "max_send_seconds", 55)) * 1000)
        too_long = 0 < duration_ms > max_send_ms   # 超气泡上限
        ok = False
        if too_long and self.channel.voice_prefers == "silk":
            # wxhook 没有文件消息接口：长歌切成多段语音气泡发，别再走注定失败的文件卡片
            ok = self._send_song_split_bubbles(peer, path, max_send_ms,
                                               notice=notice) or ok
        else:
            if mode in ("bubble", "both") and not too_long:
                # both 模式下气泡失败不再内部回落文件，避免下面重复发一次
                ok = self._send_song_bubble(peer, path, duration_ms,
                                            allow_file_fallback=(mode == "bubble")) or ok
            if mode in ("file", "both") or too_long:
                ok = self._send_song_file(peer, path) or ok
        label = "split-bubbles(超气泡上限)" if too_long and self.channel.voice_prefers == "silk" \
            else ("file-only(超气泡上限)" if too_long else mode)
        self.log(f"发歌声[{label}] {'ok' if ok else 'fail'}: {os.path.basename(path)}")
        return ok

    def _send_song_result(self, peer: str, info: dict):
        """唱歌后台任务回调：发语音/文件；失败落 outbox，别让任务 ready 但人没收到。"""
        info = info or {}
        if info.get("error"):
            self.log(f"唱歌失败 {peer}：{str(info['error'])[:160]}")
            self._send_text("诶，这次没唱成…等下再试", peer)
            return
        path = str(info.get("path") or "")
        if not path or not os.path.exists(path):
            self.log(f"唱歌文件不存在 {peer}：{path!r}")
            self._send_text("唱完了但音频没存下来，等下再试", peer)
            return
        payload = {"path": path, "duration_ms": int(info.get("duration_ms") or 0),
                   "title": info.get("title") or ""}
        if not WeChatBridge._deliver_song(self, peer, payload, notice=True):
            self._enqueue_media(peer, "song", payload,
                                dedupe_key="song:%s|%s" % (peer, path))

    def _send_song_split_bubbles(self, peer: str, path: str, max_send_ms: int,
                                 notice: bool = True) -> bool:
        """长歌按 max_send_seconds 切成多段 SILK 语音气泡（微信单条语音上限 60s）。"""
        import glob
        import shutil
        import subprocess
        import tempfile
        try:
            import imageio_ffmpeg
            ff = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as e:  # noqa: BLE001
            self.log(f"分段发歌失败（无 ffmpeg）：{e!r}")
            return False
        seg_sec = max(5.0, max_send_ms / 1000.0 - 1.0)   # 留 1 秒余量
        tmpdir = tempfile.mkdtemp(prefix="song_split_")
        try:
            pattern = os.path.join(tmpdir, "seg_%03d.mp3")
            r = subprocess.run(
                [ff, "-hide_banner", "-loglevel", "error", "-y", "-i", path,
                 "-f", "segment", "-segment_time", "%.3f" % seg_sec,
                 "-codec:a", "libmp3lame", "-b:a", "320k", pattern],
                capture_output=True, text=True, timeout=300)
            segs = sorted(glob.glob(os.path.join(tmpdir, "seg_*.mp3")))
            if r.returncode != 0 or not segs:
                self.log(f"分段发歌失败（ffmpeg rc={r.returncode}）：{(r.stderr or '')[-160:]}")
                return False
            # 先说明一下，避免突然连发好几条语音
            if notice:
                self._send_text("整首有点长，微信语音一条最多 60 秒，我分 %d 段发你～" % len(segs), peer)
            ok_any = False
            for i, seg in enumerate(segs, 1):
                try:
                    from .silk_audio import mp3_to_silk
                    with open(seg, "rb") as f:
                        audio = f.read()
                    silk_path = os.path.join(
                        self.media_dir,
                        "song_seg_%s_%02d.silk" % (time.strftime("%Y%m%d_%H%M%S"), i))
                    conv = mp3_to_silk(audio, silk_path, work_dir=self.media_dir)
                    code = self.channel.send_voice_silk(conv["path"], peer, conv["duration_ms"])
                    self.log(f"歌声分段 {i}/{len(segs)} "
                             f"{'ok' if code == 0 else 'fail(%s)' % code}"
                             f"（{conv['duration_ms']}ms/{conv['bytes']}B）")
                    ok_any = ok_any or code == 0
                except Exception as e:  # noqa: BLE001
                    self.log(f"歌声分段 {i}/{len(segs)} 失败：{e!r}")
                time.sleep(1.2)      # 别把微信发送频率打满
            return ok_any
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _send_song_file(self, peer: str, path: str) -> bool:
        """高音质文件卡片（mp3 320kbps / 48kHz 立体声原样发出）。"""
        try:
            code = self.channel.send_file(path, peer)
            self.log(f"歌声文件卡片 {'ok' if code == 0 else f'fail({code})'}")
            return code == 0
        except Exception as e:  # noqa: BLE001
            self.log(f"发歌声文件失败：{e!r}")
            return False

    def _send_song_bubble(self, peer: str, path: str, duration_ms: int = 0,
                          allow_file_fallback: bool = True) -> bool:
        """优先 SILK 真语音气泡；编码/发送失败可按参数回落 mp3 文件卡片。"""
        max_send_ms = int(float(getattr(self.cfg.song_gen, "max_send_seconds", 55)) * 1000)
        if (self.channel.voice_prefers == "silk" and path.lower().endswith(".mp3")
                and (duration_ms <= 0 or duration_ms <= max_send_ms)):
            try:
                from .silk_audio import mp3_to_silk
                with open(path, "rb") as f:
                    audio = f.read()
                silk_path = os.path.join(
                    self.media_dir, "song_%s.silk" % time.strftime("%Y%m%d_%H%M%S"))
                conv = mp3_to_silk(audio, silk_path, work_dir=self.media_dir)
                code = self.channel.send_voice_silk(conv["path"], peer, conv["duration_ms"])
                self.log(f"歌声气泡 {'ok' if code == 0 else f'fail({code})'}"
                         f"（{conv['duration_ms']}ms/{conv['bytes']}B）")
                if code == 0:
                    return True
            except Exception as e:  # noqa: BLE001  气泡失败回落文件
                self.log(f"歌声气泡失败，回落文件：{e!r}")
        if allow_file_fallback:
            return self._send_song_file(peer, path)
        return False


    def _send_sticker_reply(self, text: str, peer: str, agent=None,
                            self_initiated: bool = False):
        """带表情包的发送：解析 [sticker:情绪词] -> 先发图，再发正文。"""
        store = getattr(agent, "stickers", None) if agent is not None else None
        if store is None or not getattr(store, "enabled", False):
            self._send_reply(text, peer, self_initiated=self_initiated)
            return
        from .stickers import extract_stickers
        clean, paths = extract_stickers(text, store)
        for p in paths:
            try:
                code = self.channel.send_image(p, peer)
                self.log(f"发表情 {'ok' if code == 0 else f'fail({code})'}: "
                         f"{p.rsplit(chr(92), 1)[-1].rsplit('/', 1)[-1]}")
            except Exception as e:  # noqa: BLE001  表情发失败不影响正文
                self.log(f"发表情失败：{e!r}")
        if clean:
            self._send_reply(clean, peer, self_initiated=self_initiated)

    def _send_reply(self, text: str, peer: str, self_initiated: bool = False):
        """拟人发送回复：拆成多条短消息、每条带打字延迟（秒回是大机器味）。
        系统提示/降级话术走 _send_text（立即、单条）。self_initiated=插嘴/主动。"""
        for bubble in split_bubbles(text):
            # 打字延迟：基础随机 + 按字数增长，封顶 12s
            delay = random.uniform(1.2, 3.0) + min(9.0, len(bubble) * 0.045)
            self.stop.wait(delay)
            if self.stop.is_set():
                return
            self._send_text(bubble, peer, self_initiated=self_initiated)

    def _send_ok(self, peer: str, self_initiated: bool = False) -> bool:
        """同一会话 60s 内的自发消息条数上限（硬闸）。

        只掐"自己找话说"的消息（插嘴/主动）：直接回复永远不丢——用户问了就必须答，
        宁可答得多，也不能因为限速把人晾着。"""
        now = time.time()
        q = self._send_log.setdefault(peer, [])
        q[:] = [x for x in q if now - x < 60.0]
        q.append(now)
        if not self_initiated:
            return True
        limit = int(getattr(self.cfg.wechat, "group_send_max_per_minute", 6) or 0)
        return limit <= 0 or len(q) <= limit

    def _send_text(self, text: str, peer: str, self_initiated: bool = False):
        text = strip_media_markers(strip_role_prefix(text))
        if not text:
            return
        if not self._send_ok(peer, self_initiated):
            self.log(f"发送限速：{peer} 60s 内自发消息已达上限，丢弃：{text[:30]}")
            return
        self._last_send[peer] = time.time()
        if self._send_text_raw(text, peer):
            return
        # 三次直发都失败：不丢，落出站队列，后台按退避继续试。
        self._enqueue_text(text, peer, self_initiated)

    def _send_text_raw(self, text: str, peer: str) -> bool:
        """单条文本直发，最多三次；返回是否成功。不查限速、不入队。"""
        last = ""
        for attempt in range(3):
            try:
                code = self.channel.send_text(text, peer)
                if code == 0:
                    return True
                last = f"ret={code}"
            except Exception as e:  # noqa: BLE001  单次失败继续重试
                last = repr(e)
            if attempt < 2:
                self.stop.wait(0.6 * (attempt + 1))
        self.log(f"发文本失败（已重试3次）{last}: {text[:40]}")
        return False

    def _outbox_for(self, peer: str):
        if self.pool is None:
            return None
        agent = self.pool.agents.get(peer)
        return getattr(agent, "outbox", None)

    def _enqueue_media(self, peer: str, kind: str, payload: dict,
                       dedupe_key: str = None) -> None:
        """媒体结果（歌/图/文件）发送失败也落出站队列；只靠日志等于丢。"""
        store = self._outbox_for(peer)
        if store is None:
            self.log(f"出站队列不可用（无 Agent），{kind} 丢失：{str(payload)[:60]}")
            return
        try:
            item_id = store.enqueue(peer, kind, payload, dedupe_key=dedupe_key)
            self.log(f"{kind} 入出站队列 #{item_id}（{peer}）")
        except Exception as e:  # noqa: BLE001
            self.log(f"{kind} 入队失败，结果丢失：{e!r}")

    def _notify_orphans(self, peer: str, agent) -> None:
        """桥重启后：对上一个进程留下的近期任务补一句人话，不让它静默失败。"""
        rows = list(getattr(getattr(agent, "hands", None), "orphaned_tasks", []) or [])
        try:
            agent.hands.orphaned_tasks = []
        except Exception:  # noqa: BLE001
            pass
        for t in rows:
            kind = str(t.get("kind") or "")
            if not kind.startswith("song"):
                continue
            try:
                age = time.time() - float(t.get("updated_at") or 0)
            except (TypeError, ValueError):
                age = 0.0
            if age > 1800:
                continue
            title = ""
            try:
                p = json.loads(t.get("params") or "{}")
                title = str(p.get("title") or p.get("query") or "")
            except Exception:  # noqa: BLE001
                pass
            what = f"《{title}》" if title else "刚才那首"
            self.log(f"孤儿任务通知 {peer}：{what}")
            self._enqueue_text(f"{what}没唱完，刚才进程断了…要重来一次吗？", peer)

    def _notify_task_failure(self, peer: str, agent, task: dict, reason: str) -> None:
        kind = str(task.get("kind") or "")
        if not kind.startswith("song"):
            return
        title = ""
        try:
            p = json.loads(task.get("params") or "{}")
            title = str(p.get("title") or p.get("query") or "")
        except Exception:  # noqa: BLE001
            pass
        what = f"《{title}》" if title else "刚才那首"
        self.log(f"任务卡死通知 {peer}：{what}（{reason}）")
        self._enqueue_text(f"{what}卡住了，这次没唱成…等下再试", peer)

    def _task_watch_loop(self):
        """任务看门狗：心跳超时的任务标失败并通知用户，不只等健康页变红。"""
        while not self.stop.wait(60.0):
            if self.pool is None:
                continue
            try:
                threshold = float(getattr(self.cfg.wechat, "task_stale_seconds", 1200))
            except (TypeError, ValueError):
                threshold = 1200.0
            for peer, agent in list(self.pool.agents.items()):
                store = getattr(getattr(agent, "hands", None), "tasks", None)
                if store is None:
                    continue
                try:
                    rows = store.fail_stale(
                        older_than=threshold,
                        reason=f"任务超过 {int(threshold)} 秒无心跳，已中断",
                        owner="bridge")
                except Exception as e:  # noqa: BLE001
                    self.log(f"任务看门狗读取失败 {peer}：{e!r}")
                    continue
                for task in rows:
                    self._notify_task_failure(peer, agent, task, "无心跳")

    def _enqueue_text(self, text: str, peer: str, self_initiated: bool = False) -> None:
        store = self._outbox_for(peer)
        if store is None:
            self.log(f"出站队列不可用（无 Agent），这条丢失：{text[:40]}")
            return
        try:
            key = hashlib.sha1(("%s|%s" % (peer, text)).encode("utf-8")).hexdigest()
            item_id = store.enqueue(peer, "text", {
                "text": text, "self_initiated": bool(self_initiated)}, dedupe_key=key)
            self.log(f"文本入出站队列 #{item_id}（{peer}）：{text[:30]}")
        except Exception as e:  # noqa: BLE001  入队失败只能记录
            self.log(f"文本入队失败，这条丢失：{e!r}")

    def _outbox_loop(self):
        while not self.stop.wait(5.0):
            self._drain_outbox()

    def _drain_outbox(self) -> int:
        """重试各会话的出站消息；成功标记 sent，失败按 5s 起指数退避。"""
        if self.pool is None:
            return 0
        sent = 0
        for peer, agent in list(self.pool.agents.items()):
            if not self._chat_enabled(peer):
                continue
            store = getattr(agent, "outbox", None)
            if store is None:
                continue
            try:
                items = store.due(limit=10)
            except Exception as e:  # noqa: BLE001  队列读失败不影响其他会话
                self.log(f"出站队列读取失败 {peer}：{e!r}")
                continue
            for item in items:
                try:
                    payload = json.loads(item.get("payload") or "{}")
                except Exception:  # noqa: BLE001
                    store.mark_failed(item["id"], "bad payload")
                    continue
                kind = str(item.get("kind") or "text")
                ok = False
                err = "send failed"
                if kind == "text":
                    text = str(payload.get("text") or "")
                    if not text:
                        store.mark_failed(item["id"], "empty text")
                        continue
                    ok = self._send_text_raw(text, peer)
                elif kind == "song":
                    ok = self._deliver_song(peer, payload, notice=False)
                    err = "song send failed"
                elif kind in ("image", "sticker"):
                    path = str(payload.get("path") or "")
                    if not path or not os.path.exists(path):
                        store.mark_failed(item["id"], "image missing")
                        continue
                    try:
                        code = self.channel.send_image(path, peer)
                    except Exception as e:  # noqa: BLE001
                        code, err = -1, repr(e)
                    ok = code == 0
                elif kind == "file":
                    path = str(payload.get("path") or "")
                    if not path or not os.path.exists(path):
                        store.mark_failed(item["id"], "file missing")
                        continue
                    try:
                        code = self.channel.send_file(path, peer)
                    except Exception as e:  # noqa: BLE001
                        code, err = -1, repr(e)
                    ok = code == 0
                else:
                    store.mark_failed(item["id"], "unsupported kind: " + kind)
                    continue
                if ok:
                    store.mark_sent(item["id"])
                    sent += 1
                    self.log(f"出站队列 #{item['id']} 补发成功（{peer}/{kind}）")
                else:
                    attempts = int(item.get("attempts") or 0)
                    delay = min(300.0, 5.0 * (2 ** min(attempts, 6)))
                    status = store.retry(item["id"], err, delay=delay)
                    self.log(f"出站队列 #{item['id']} 补发失败（{kind}），{status}，"
                             f"{int(delay)}s 后再试")
        return sent

    # ---------------- 线程 ----------------

    def _receive_loop(self):
        while not self.stop.is_set():
            try:
                msg = self.channel.get_msg(block=True)
            except queue.Empty:
                continue
            except Exception as e:  # noqa: BLE001
                self.log(f"取消息异常：{e!r}")
                self.stop.wait(1.0)
                continue
            self.handle_message(msg)

    def _deliver_loop(self):
        while not self.stop.wait(self.cfg.wechat.poll_seconds):
            self.deliver_pending()

    def run(self):
        if not self.channel.is_login():
            raise BridgeError("微信未登录（先在 PC 微信扫码登录，再启动桥）")
        self.log(f"桥就绪：通道={self.channel.name} self={self.channel.get_self_wxid()} "
                 f"server={self.api.base} 轮询={self.cfg.wechat.poll_seconds}s "
                 f"语音策略={self.cfg.wechat.voice_reply} "
                 f"语音形态={'气泡' if self.channel.voice_prefers == 'silk' else 'mp3文件'}")
        self.channel.enable_receiving()
        # 预热"我在群里叫什么"：群点名靠它；contact.db 没起来时用上次记住的名字兜底
        self._group_self_names(None)
        if self.multi:
            # 多用户：先建好已有会话 Agent，再补处理上次进程没来得及回复的消息；
            # 补处理放在收消息线程启动前，避免和同会话的新消息抢同一个 Agent。
            if self.pool is not None:
                self.pool.warm_sessions()
                for _peer, _agent in list(self.pool.agents.items()):
                    self._notify_orphans(_peer, _agent)
                self._recover_unanswered()
        threading.Thread(target=self._receive_loop, daemon=True, name="wx-recv").start()
        threading.Thread(target=self._outbox_loop, daemon=True, name="wx-outbox").start()
        threading.Thread(target=self._health_loop, daemon=True, name="wx-health").start()
        threading.Thread(target=self._task_watch_loop, daemon=True, name="wx-task-watch").start()
        self._write_health()
        if self.multi:
            # 多用户：独立 Agent 即时回复 + 每用户主动调度（不需要 REST/水位线）
            threading.Thread(target=self._multi_ticker, daemon=True, name="wx-tick").start()
            # 群智能插嘴看门狗：静默窗过后让 AI 自己决定接不接话
            threading.Thread(target=self._group_watch_loop, daemon=True,
                             name="wx-chime").start()
        else:
            self.bootstrap_watermark()
            threading.Thread(target=self._deliver_loop, daemon=True, name="wx-deliver").start()
        try:
            while not self.stop.is_set():
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.log("收到中断，关闭微信桥…")
        finally:
            self.stop.set()
            self.channel.disable_receiving()


# ---------------- 入口 ----------------

def _resolve_media_dir(cfg) -> str:
    d = cfg.wechat.media_dir
    return d if os.path.isabs(d) else os.path.join(project_root(), d)


def _make_channel(cfg, log):
    """按 wechat.channel 构造通道（wxhook 不需要 wcferry 包）。"""
    if cfg.wechat.channel == "wxhook":
        from .wxhook import WxHookChannel
        ch = WxHookChannel(cfg.wechat, _resolve_media_dir(cfg), log=log)
        ch.enable_voice_send()   # hook 语音发送默认关：启动即打开
        return ch
    try:
        from wcferry import Wcf
    except ImportError:
        raise BridgeError("缺少 wcferry：pip install --upgrade wcferry"
                          "（或把 wechat.channel 切到 wxhook；均见 docs/wechat_setup.md）") from None
    return WcfChannel(Wcf(debug=False, block=True))  # block=True：等微信登录完成


def run(probe: bool = False) -> int:
    load_env_dotfile()
    config_dir = os.environ.get("AGENT_CONFIG_DIR") or os.path.join(project_root(), "config")
    cfg = load_config(os.path.join(config_dir, "config.yaml"))
    # 部署友好覆盖（与 Agent 同名环境变量）
    if os.environ.get("AGENT_LLM_BASE_URL"):
        cfg.llm.base_url = os.environ["AGENT_LLM_BASE_URL"]
    if os.environ.get("AGENT_ASR_MODEL"):
        cfg.asr.model = os.environ["AGENT_ASR_MODEL"]
    if os.environ.get("AGENT_WECHAT_SERVER_URL"):
        cfg.wechat.server_url = os.environ["AGENT_WECHAT_SERVER_URL"]
    if os.environ.get("AGENT_WECHAT_OWNER"):
        cfg.wechat.owner_wxid = os.environ["AGENT_WECHAT_OWNER"]
    if os.environ.get("AGENT_WECHAT_CHANNEL"):
        cfg.wechat.channel = os.environ["AGENT_WECHAT_CHANNEL"]

    token = os.environ.get(cfg.wechat.api_token_env, "")
    log = lambda msg: print(f"[wechat {time.strftime('%H:%M:%S')}] {msg}")  # noqa: E731
    if probe:
        return _probe(cfg, token, log)

    api = AgentAPI(cfg.wechat.server_url, token)
    state = BridgeState(os.path.join(project_root(), "data", "wechat_bridge_state.json"))
    # 聊天命令改过的桥级参数（/off chime、/chat off…）覆盖 config.yaml，重启后仍生效
    from .chat_commands import apply_bridge_overrides
    applied = apply_bridge_overrides(cfg, state.overrides)
    if applied:
        log(f"命令覆盖已恢复：{', '.join(applied)}")
    from .asr_client import ASRClient
    bridge = WeChatBridge(cfg, _make_channel(cfg, log), api, state,
                          asr=ASRClient(cfg.asr, cfg.llm),
                          media_dir=_resolve_media_dir(cfg), log=log)
    bridge.run()
    return 0


def _silk_codec_check() -> str:
    """SILK 编解码器链：pysilk-mod / pilk / Node+silk-wasm 任一可用即可。"""
    import shutil
    try:
        import pysilk  # noqa: F401
        return "pysilk-mod"
    except ImportError:
        pass
    try:
        import pilk  # noqa: F401
        return "pilk"
    except ImportError:
        pass
    if shutil.which("node"):
        return "Node + silk-wasm（tools/silk_node）"
    raise BridgeError("无可用 SILK 编码器：pip install pysilk-mod（推荐）/ "
                      "pip install pilk，或安装 Node")


def _pkg_version(pkg: str) -> str:
    try:
        from importlib.metadata import version
        return version(pkg)
    except Exception:  # noqa: BLE001  老版本无 metadata 时退回导入探测
        try:
            return getattr(__import__(pkg), "__version__", "已安装")
        except ImportError:
            raise


def _probe(cfg, token: str, log=print) -> int:
    """连通性自检：按通道逐项打勾。"""
    ok = True

    def check(name: str, fn):
        nonlocal ok
        try:
            detail = fn()
            print(f"  [ok] {name}{': ' + detail if detail else ''}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  [FAIL] {name}: {e!r}")

    def _key_set(env_name: str) -> str:
        if not os.environ.get(env_name):
            raise BridgeError(f"环境变量 {env_name} 未设置（.env 或系统环境变量）")
        return "已设置"

    api = AgentAPI(cfg.wechat.server_url, token)
    channel = cfg.wechat.channel
    print(f"微信桥连通性自检（channel={channel}）：")
    check("Python", lambda: "%s @ %s" % (__import__("sys").version.split()[0],
                                         __import__("sys").platform))
    check(f"WithMe 服务端（{cfg.wechat.server_url}）",
          lambda: "在线 " + str(api.health().get("role_id")))
    check("edge-tts（语音回复用）", lambda: _pkg_version("edge-tts"))
    check(f"ASR key（{cfg.asr.api_key_env}）", lambda: _key_set(cfg.asr.api_key_env))
    check("画图（image_gen）",
          lambda: f"model={cfg.image_gen.model}" if cfg.image_gen.enabled else "已关闭")

    if channel == "wxhook":
        from .wxhook import WxHookChannel
        ch = WxHookChannel(cfg.wechat, _resolve_media_dir(cfg))

        def _hook():
            if not ch.is_login():
                raise BridgeError("hook 无响应或未登录（version.dll 放进微信目录、"
                                  "按启动参数拉起 Weixin.exe，见 docs/wechat_setup.md）")
            return f"已登录 wxid={ch.get_self_wxid()}"
        check(f"wechat-hook（{cfg.wechat.wxhook.base_url}）", _hook)
        check("SILK 编解码（语音气泡）", _silk_codec_check)
        check("zstandard（收消息解码）", lambda: _pkg_version("zstandard"))
        check("imageio-ffmpeg（mp3->PCM）", lambda: _pkg_version("imageio-ffmpeg"))
        check("语音气泡开关", lambda: "已打开" if ch.enable_voice_send() else "打开失败")
    else:
        check("wcferry（微信 hook）", lambda: _pkg_version("wcferry"))

        def _wx_login():
            from wcferry import Wcf
            wcf = Wcf(debug=False, block=False)
            if not wcf.is_login():
                raise BridgeError("微信未登录（PC 微信需安装 wcferry 对应版本并扫码登录）")
            return f"已登录 wxid={wcf.get_self_wxid()}"
        check("PC 微信 hook", _wx_login)
    print("自检结果：" + ("全部通过" if ok else "存在失败项，见上"))
    return 0 if ok else 1
