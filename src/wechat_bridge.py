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
import json
import os
import queue
import threading
import time
import urllib.error
import urllib.request

from .config import load_config
from .prompt_builder import strip_media_markers, strip_role_prefix
from .storage import project_root

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
        if os.path.exists(path):
            try:
                data = json.load(open(path, encoding="utf-8"))
                self.owner_wxid = data.get("owner_wxid") or ""
                self.watermark = int(data.get("watermark") or 0)
            except (json.JSONDecodeError, OSError, ValueError):
                pass

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"owner_wxid": self.owner_wxid, "watermark": self.watermark}, f)


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
                return agent
            if len(self.agents) >= self.cfg.wechat.max_users:
                return None
            from .main import Agent
            db = self._db_path_for(wxid)
            is_group = bool(wxid and wxid.endswith("@chatroom"))
            agent = Agent(db_path=db, threaded=True,
                          group_room=wxid if is_group else "",
                          group_name=self._display_name(wxid))
            agent.is_group = is_group
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
    按句末标点/换行切，碎片合并，最多 max_bubbles 条。"""
    text = (text or "").strip()
    if len(text) <= 48 or max_bubbles <= 1:
        return [text] if text else []
    parts = [p.strip() for p in _re.split(r"(?<=[。！？!?…\n])+", text) if p.strip()]
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

    def handle_message(self, msg) -> None:
        """一条入站消息 -> 路由。通道注入假件时本函数即测试面。"""
        try:
            if msg.from_self():
                return
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
        return [{"b64": base64.b64encode(raw).decode(), "mime": "image/jpeg"}]

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
            if "<emoji" in ref[:120]:
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
            self.log(f"引用图片已解出（{fmt}, {os.path.getsize(dst) // 1024}KB）")
            return [{"b64": base64.b64encode(f.read()).decode(),
                     "mime": "image/jpeg"}]

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
        hit = bool(getattr(msg, "at_me", False)) or is_owner or not mention_only
        # 频率素材对所有群消息记账（插嘴触发只数"没被回复的旁听"，回复不攒门）
        label = ""
        nick_getter = getattr(self.channel, "nickname_of", None)
        if nick_getter and sender:
            label = nick_getter(sender)
        self._group_last[msg.roomid] = (sender, label)
        self._group_note_activity(msg.roomid, overheard=not hit)
        if not hit:
            self.log(f"群 {msg.roomid}：旁听 {sender[-8:]}（未@我，静默等插嘴决策）")
            self._group_overhear(msg, sender)
            return
        if msg.type not in (MSG_TEXT, MSG_IMAGE, MSG_VOICE, 47, MSG_APPMSG):
            return   # 10000 系统提示等
        if msg.type == MSG_TEXT:
            # 角色不写进正文（正文会发给用户/被模型抄进回复），只放 meta
            msg.content = self._strip_group_noise(msg.content or "")
        peer = msg.roomid
        if msg.type == MSG_TEXT and not (msg.content or "").strip():
            return
        if not self._multi_ok(peer):
            return
        why = ("@我" if getattr(msg, "at_me", False) else
               "创造者" if is_owner else "全量")
        msg._from_owner = is_owner   # 创造者消息标记：用于记忆/关系归属，不做服从模式
        msg._speaker_name = label or (sender[-6:] if sender else "群友")
        msg._speaker_wxid = sender
        msg._speaker_role = "owner" if is_owner else "member"
        msg._addressee = ("你" if (is_owner or getattr(msg, "at_me", False)
                                   or self._group_addressed_to_me(msg, agent)) else "群聊")
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
        """群智能插嘴的频率门控：攒够 2 条未被回复的旁听 + 静默 6s（说完一轮）
        + 距上次插嘴超冷却 + 当日未超限 -> 触发一次 AI 决策。返回是否触发过
        （无论接没接）。门控先于 AI：不够格根本不问模型，不烧 token。

        被点名（正文出现我的名字）是硬信号：不攒批、不等静默窗、不受冷却和
        日限——叫名字必须接，否则就是群里的冷暴力。"""
        now = time.time() if now is None else now
        w = self._group_watch.get(room)
        if not w:
            return False
        directed = int(w.get("mention", 0)) > 0
        about_me = int(w.get("about_me", 0)) > 0
        if about_me and not directed:
            # 别人只是在跟第三方说话时提到你/让人骂你：不插嘴，别把自己当服务
            return False
        if directed:
            if now - float(w.get("last_ts", 0)) < 1.0:   # 连发未落地，稍等
                return False
        else:
            # >=2 条未被回复的旁听才问 AI：单句碎碎念不值得回，成一小轮才考虑接
            if int(w.get("oh", 0)) < 2:
                return False
            gap = now - float(w.get("last_ts", 0))
            if gap < 6.0 or gap > 180.0:  # 没说完不插；冷掉的话题不插旧梗
                return False
            if now - float(w.get("last_chime", 0)) < \
                    float(getattr(self.cfg.wechat, "group_chime_cooldown_seconds", 480)):
                # 冷却期不把旁听清空：继续攒，冷却结束话题还热就能接
                return False
            day = time.localtime(now).tm_yday
            if w.get("day") != day:
                w["day"], w["chimes"] = day, 0
            if int(w.get("chimes", 0)) >= int(getattr(self.cfg.wechat, "group_chime_daily_max", 12)):
                w["msgs"] = w["oh"] = 0
                return False
        w["msgs"] = w["oh"] = 0          # 本轮已评估，重开窗口
        w["mention"] = 0
        w["about_me"] = 0
        w["last_chime"] = now
        agent = self.pool.agents.get(room)
        if agent is None or not hasattr(agent, "chime_in"):
            return True
        if hasattr(agent, "group_self_names"):
            agent.group_self_names = self._group_self_names(agent)
            agent.group_owner_name = self._group_owner_name()
        try:
            rows = agent.conn.execute(
                "SELECT sender, content, meta, speaker, speaker_role"
                " FROM messages WHERE role_id=? ORDER BY id DESC LIMIT 8",
                (agent.role_id,)).fetchall()
            def _line(r):
                c = (r["content"] or "")[:80]
                if r["sender"] != "user":
                    return "你：" + c
                try:
                    m = json.loads(r["meta"] or "{}")
                except Exception:  # noqa: BLE001
                    m = {}
                role = r["speaker_role"] or m.get("speaker_role") or ""
                name = r["speaker"] or m.get("speaker") or ""
                if role == "owner":
                    return f"[创造者 {name}]：{c}"
                if role == "member":
                    return f"[群友 {name}]：{c}"
                return "群友：" + c
            dialogue = [_line(r) for r in reversed(rows)]
            text = agent.chime_in(dialogue, now=now,
                                  self_names=self._group_self_names(agent),
                                  directed=directed, about_me=about_me,
                                  owner_name=self._group_owner_name())
        except Exception as e:  # noqa: BLE001
            self.log(f"插嘴决策失败 {room}：{e!r}")
            return True
        if not text and directed:
            # 点名兜底：模型判断失误也不能冷暴力，至少应一声
            text = "嗯？我在"
            try:
                with agent._lock:
                    agent._log_message("agent", text, {"chime": True}, now)
            except Exception:  # noqa: BLE001
                pass
        if text:
            w["chimes"] = int(w.get("chimes", 0)) + 1
            self.log(f"群 {room}：AI 决定插嘴 -> {text[:40]}")
            self._send_reply(text, room)
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

    def _group_overhear(self, msg, sender: str):
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
            with agent._lock:
                agent.conn.execute(
                    "INSERT INTO messages(role_id, sender, content, meta, created_at,"
                    " speaker, speaker_role) VALUES(?,?,?,?,?,?,?)",
                    (agent.role_id, "user", text[:200],
                     json.dumps({"overheard": True, "speaker": label,
                                 "speaker_role": "member",
                                 "addressee": ("你" if direct else "群聊"),
                                 "about_me": bool(mentioned and not direct)},
                                ensure_ascii=False),
                     _t.time(), label, "member"))
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
        for m in batch:
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
                                           "mime": "image/jpeg"})
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
                    for p3 in paths:
                        with open(p3, "rb") as f:
                            raw = f.read()
                        total += len(raw)
                        images.append({"b64": base64.b64encode(raw).decode(),
                                       "mime": "image/jpeg"})
                    self.log(f"收表情包 {len(paths)} 帧 {total // 1024}KB -> 视觉理解")
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
                          speaker_wxid=speaker_wxid, addressee=addressee)

    def _multi_chat(self, text, images, was_voice, peer, voice_request=False,
                    from_owner=True, speaker_name="", speaker_role="",
                    speaker_wxid="", addressee="群聊"):
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
        try:
            result = agent.on_user_message(text, images=images, from_owner=from_owner,
                                           speaker=speaker_name,
                                           speaker_role=speaker_role,
                                           speaker_wxid=speaker_wxid,
                                           addressee=addressee)
        except Exception as e:  # noqa: BLE001
            self.log(f"多用户：{peer} 处理失败：{e!r}")
            return
        hand = ((result.get("meta") or {}).get("hand") or {})
        for img in hand.get("images") or []:
            code = self.channel.send_image(str(img), peer)
            self.log(f"发图 {'ok' if code == 0 else f'fail({code})'}: {img}")
        if result.get("addon"):
            self._send_reply(result["addon"], peer)
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
        if want_voice and not self._send_voice_reply(reply, peer):
            self._send_reply(reply, peer)  # 语音链路失败回落文字
        elif not want_voice:
            self._send_reply(reply, peer)

    def _multi_ticker(self):
        """多用户调度：每个用户独立评估主动行为与反思（间隔 = tick.fast_seconds）。"""
        interval = max(15.0, float(getattr(self.cfg.tick, "fast_seconds", 60)))
        while not self.stop.wait(interval):
            groups_proactive = bool(getattr(self.cfg.wechat, "groups_proactive", True))
            for wxid, agent in list(self.pool.agents.items()):
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
        try:
            result = agent.evaluate_proactive()
            sent = (result or {}).get("sent")
            if sent and sent.get("text"):
                text = sent["text"]
                if wxid.endswith("@chatroom"):
                    last = self._group_last.get(wxid)
                    if last and last[1]:
                        text = f"@{last[1]} {text}"
                self._send_text(text, wxid)
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
            code = self.channel.send_image(str(img), peer)
            self.log(f"发图 {'ok' if code == 0 else f'fail({code})'}: {img}")
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
        self._send_reply(content, peer)

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

    def _send_reply(self, text: str, peer: str):
        """拟人发送回复：拆成多条短消息、每条带打字延迟（秒回是大机器味）。
        系统提示/降级话术走 _send_text（立即、单条）。"""
        for bubble in split_bubbles(text):
            # 打字延迟：基础随机 + 按字数增长，封顶 12s
            delay = random.uniform(1.2, 3.0) + min(9.0, len(bubble) * 0.045)
            self.stop.wait(delay)
            if self.stop.is_set():
                return
            self._send_text(bubble, peer)

    def _send_text(self, text: str, peer: str):
        text = strip_media_markers(strip_role_prefix(text))
        code = self.channel.send_text(text, peer)
        if code != 0:
            self.log(f"发文本失败 ret={code}: {text[:40]}")

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
        threading.Thread(target=self._receive_loop, daemon=True, name="wx-recv").start()
        if self.multi:
            # 多用户：独立 Agent 即时回复 + 每用户主动调度（不需要 REST/水位线）
            # 预建已有群 Agent：群名/人物卡提前落库，控制台会话轴直接可读
            if self.pool is not None:
                self.pool.warm_sessions()
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
