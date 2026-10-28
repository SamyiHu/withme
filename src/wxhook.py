"""wxhook 通道适配器：对接 github.com/mengyunqwq/wechat-hook（微信 4.1.10.27 x64，
version.dll 代理 hook，本地 HTTP :30001）。该通道的关键能力是 **真语音气泡**：

    /SendVoiceMsg {wxidorgid, path: <silk 文件>, str0Mode: "bytes", durationMs}
    —— hook 读 silk 文件、以原始字节（自动补 0x02 前缀）调微信自己的语音消息工厂，
       群/私聊里出现一条 type 34 的原生语音气泡（2026-10 上游真机实测链路）。

收消息（通知式推送 + 查库）：
    hook 在消息写库时 POST {"t":"msg","table":"Msg_<md5(roomId)>"} 到
    http://127.0.0.1:<notify_port>/recv；本适配器收到通知后调
    /QueryDB/execute 查该表新行。列值是 hex 字符串，zstd 压缩的自动解压。
    另有兜底轮询线程（hook 推送没开/丢通知时仍能收到）。

媒体：
    语音收：media_0.db VoiceInfo.voice_data（BLOB=hex，明文 0x02+#!SILK_V3）→ pilk 转 WAV16
    图片收：<数据根>/msg/attach/<md5(roomId)>/<年-月>/Img/*.dat → /Decode_Pic 解密（失败降级）

可选依赖：pip install pilk imageio-ffmpeg zstandard
"""
import json
import os
import queue
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
HEX_CHARS = set("0123456789abcdefABCDEF")


class WxHookError(Exception):
    pass


def _is_hex(s: str) -> bool:
    return bool(s) and len(s) % 2 == 0 and all(c in HEX_CHARS for c in s)


def _default_zstd_decompress(data: bytes) -> bytes:
    import zstandard
    return zstandard.ZstdDecompressor().decompressobj().decompress(data)


def decode_content(raw, zstd_decompress=_default_zstd_decompress) -> str:
    """hook 返回的列值（hex 字符串，可能 zstd 压缩）-> 明文 utf8。"""
    s = str(raw or "")
    if not s:
        return ""
    if not _is_hex(s):
        return s  # 某些版本直接给明文
    buf = bytes.fromhex(s)
    if buf[:4] == ZSTD_MAGIC:
        if zstd_decompress is None:
            raise WxHookError("内容是 zstd 压缩：pip install zstandard")
        return zstd_decompress(buf).decode("utf-8", "replace")
    return buf.decode("utf-8", "replace")


def table_for_room(room_id: str) -> str:
    import hashlib
    return "Msg_" + hashlib.md5(room_id.encode("utf-8")).hexdigest()


def payload_xml(msg, markers) -> str:
    """挑出真正承载消息载荷的 XML（表情 cdnurl / 图片 aeskey / appmsg 链接都在这）。

    两个通道的载荷位置不同：wcf 在 msg.xml；wxhook 在 msg.content
    （=message_content），而 msg.xml 是 source 列的 <msgsource> 元数据
    （签名等，2026-10-27 实测它非空且什么都不含，会把真载荷整个遮住）。
    谁含 markers 任一标记就用谁；都不含退回 content 优先。"""
    content = getattr(msg, "content", "") or ""
    xml = getattr(msg, "xml", "") or ""
    for cand in (content, xml):
        if cand and any(mk in cand for mk in markers):
            return cand
    return content or xml


class Msg:
    """标准化入站消息（与 wcf 的 WxMsg 字段对齐，桥的路由层零改动）。"""

    def __init__(self, type_, id_, sender, roomid, content, xml="", extra="",
                 is_self=False, is_group=False, at_me=False):
        self.type = type_
        self.id = id_
        self.sender = sender
        self.roomid = roomid
        self.content = content
        self.xml = xml
        self.extra = extra
        self._self = is_self
        self._group = is_group
        self.at_me = at_me   # 群消息：atuserlist 命中自己（@ 的硬信号，不靠昵称猜）

    def from_self(self):
        return self._self

    def from_group(self):
        return self._group


class WxHookChannel:
    name = "wxhook"
    voice_prefers = "silk"       # 桥据此走 SILK 气泡链路（wcf 走 mp3 文件卡片）

    POLL_COLUMNS = ("local_id, local_type, create_time, real_sender_id,"
                    " message_content, source")

    def __init__(self, wechat_cfg, media_dir: str, log=None,
                 zstd_decompress=_default_zstd_decompress, silk_to_wav=None):
        self.cfg = wechat_cfg
        self.base = wechat_cfg.wxhook.base_url.rstrip("/")
        self.media_dir = media_dir
        os.makedirs(self.media_dir, exist_ok=True)
        self.log = log or (lambda m: print(f"[wxhook {time.strftime('%H:%M:%S')}] {m}"))
        self._zstd = zstd_decompress
        self._silk_to_wav = silk_to_wav or self._default_silk_to_wav
        self._q = queue.Queue()
        self._nick_cache = {}   # wxid -> 昵称（群聊标注谁在说话）
        self._stop = threading.Event()
        self._poll_lock = threading.Lock()
        self._last_local_id = {}     # 表 -> 已处理到的 local_id（首见只记水位不回放）
        self._room_map = {}          # 表名 -> roomId
        self._name2id = {}           # message_0.db Name2Id: rowid -> wxid
        self._media_chat_ids = {}    # roomId -> media_0.db Name2Id rowid
        self._self_wxid = ""
        self._tables_cache = []
        self._tables_cache_at = 0.0
        self._self_warned = False
        self._data_root = wechat_cfg.wxhook.data_root or ""
        self._server = None
        self._threads = []

    # ---------------- HTTP 底座 ----------------

    def _call(self, path: str, body: dict = None, timeout: int = 30, method: str = "POST") -> dict:
        # 上游 hook 的 /QueryDB/status 只注册了 GET（POST 返回 404）；其余接口为 POST。
        data = json.dumps(body or {}).encode("utf-8") if method != "GET" else None
        req = urllib.request.Request(
            self.base + path, data=data, method=method,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                out = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            raise WxHookError(f"{path} 请求失败（hook 在 {self.base}？）：{e!r}") from None
        if isinstance(out, dict) and out.get("ret") not in (None, 0):
            raise WxHookError(f"{path} ret={out.get('ret')} {out.get('msg') or out.get('retmsg')}")
        return out

    def _query(self, db: str, sql: str, timeout: int = 30) -> list:
        out = self._call("/QueryDB/execute", {"optDbName": db, "SQL": sql}, timeout)
        return out.get("data") or []

    # ---------------- 基础面 ----------------

    def is_login(self) -> bool:
        try:
            out = self._call("/QueryDB/status", timeout=5, method="GET")
            return out.get("IsLogin") == 1
        except WxHookError:
            return False

    def get_self_wxid(self) -> str:
        if not self._self_wxid:
            try:
                out = self._call("/GetSelfProfile", timeout=10)
                self._self_wxid = str(out.get("wxid") or "")
            except WxHookError:
                self._self_wxid = ""
        if not self._self_wxid:
            # 回退：数据目录名即本机登录账号（<xwechat_files>/<wxid>_<hash>/db_storage/…）。
            # GetSelfProfile 在点开一次「通讯录」前常为空（contact.db 懒加载），不等它。
            try:
                import re
                rows = self._query("message_0.db", "PRAGMA database_list")
                m = re.search(r"wxid_[a-z0-9]+", str((rows or [{}])[0].get("file") or ""))
                if m:
                    self._self_wxid = m.group(0)
            except Exception as e:  # noqa: BLE001
                self.log(f"由库路径推导 wxid 失败：{e!r}")
        return self._self_wxid

    # ---------------- 发送面 ----------------

    def send_text(self, msg: str, receiver: str, aters: str = "") -> int:
        return self._call("/SendTextMsg", {"wxidorgid": receiver, "msg": msg}).get("ret", -1)

    def send_image(self, path: str, receiver: str) -> int:
        return self._call("/SendImgMsg", {"wxidorgid": receiver, "path": path}).get("ret", -1)

    def send_file(self, path: str, receiver: str) -> int:
        # wxhook 无文件消息接口：图片文件改走图片，其余不支持（桥会回落文字）
        if str(path).lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp")):
            return self.send_image(path, receiver)
        self.log(f"wxhook 通道不支持发文件：{path}")
        return -1

    def send_voice_silk(self, silk_path: str, receiver: str, duration_ms: int = 0) -> int:
        """真语音气泡：str0Mode=bytes（hook 以 SILK 原始字节调微信语音工厂）。"""
        return self._call("/SendVoiceMsg",
                          {"wxidorgid": receiver, "path": str(silk_path),
                           "str0Mode": "bytes", "durationMs": int(duration_ms or 0)}).get("ret", -1)

    def enable_voice_send(self) -> bool:
        """hook 的语音发送默认关：运行时开关 + 标记文件双保险。
        微信进程崩溃重启后 DLL 里的运行时开关会丢（ret=-1001 disabled），
        必须落 <Weixin.exe 目录>\\hook-voice-send.on 才能自恢复。"""
        try:
            out = self._call("/Debug/VoiceSend", {"on": True})
            self.log(f"语音气泡开关：{json.dumps(out, ensure_ascii=False)[:120]}")
            marker = (out or {}).get("markerFile") or ""
            if marker and not os.path.exists(marker):
                try:
                    with open(marker, "w", encoding="utf-8") as f:
                        f.write("on")
                    self.log(f"语音开关标记文件已落盘：{marker}（微信重启后仍生效）")
                except OSError as e:
                    self.log(f"标记文件落盘失败（微信重启后语音会哑，需重开桥）：{e!r}")
            return True
        except WxHookError as e:
            self.log(f"打开语音发送开关失败（发送语音会 ret=-1001）：{e}")
            return False

    # ---------------- 媒体获取 ----------------

    @staticmethod
    def _default_silk_to_wav(silk: bytes, wav_path: str) -> str:
        from .silk_audio import silk_to_wav16
        return silk_to_wav16(silk, wav_path)

    def fetch_voice(self, msg) -> dict:
        """type 34 语音 -> 16kHz WAV（ASR 用）。失败返回 None（桥温和降级）。
        VoiceInfo 行的插入与消息行有先后竞争；media_0.db 句柄也是懒开的
        （微信收到/播放语音才打开），WxHookError 同样带重试而不是直接放弃。"""
        try:
            rid = int(msg.id)
            chat_id = self._media_chat_id(msg.roomid)
            where = (f"chat_name_id = {chat_id} AND local_id = {rid}" if chat_id
                     else f"local_id = {rid}")
            v = ""
            for attempt in range(5):
                try:
                    rows = self._query("media_0.db",
                                       "SELECT voice_data FROM VoiceInfo WHERE " + where
                                       + " ORDER BY create_time DESC LIMIT 1")
                    v = str(rows[0].get("voice_data") or "") if rows else ""
                except WxHookError as e:
                    if attempt == 0:
                        self.log(f"media_0.db 句柄未开（微信收到语音通常会自动开），"
                                 f"重试中：{e}")
                    v = ""
                if v:
                    break
                time.sleep(1.2)
            data = bytes.fromhex(v) if _is_hex(v) else v.encode("latin1")
            if len(data) < 16:
                return None
            wav = os.path.join(self.media_dir, f"voice_{msg.id}.wav")
            self._silk_to_wav(data, wav)
            return {"path": wav, "mime": "audio/wav"}
        except Exception as e:  # noqa: BLE001
            self.log(f"取语音失败 #{getattr(msg, 'id', '?')}：{e!r}")
            return None

    def _media_chat_id(self, room_id: str) -> int:
        if room_id in self._media_chat_ids:
            return self._media_chat_ids[room_id]
        esc = room_id.replace("'", "''")
        rows = self._query("media_0.db",
                           f"SELECT rowid AS rid FROM Name2Id WHERE user_name='{esc}' LIMIT 1")
        rid = int(rows[0]["rid"]) if rows else 0
        self._media_chat_ids[room_id] = rid
        return rid

    def last_media_msg(self, room: str):
        """该会话最近一条 图片/表情 消息（重看用）。失败返回 None。"""
        try:
            table = table_for_room(room)
            rows = self._query(
                "message_0.db",
                f"SELECT local_id, local_type, real_sender_id, message_content, source "
                f"FROM {table} WHERE (local_type & 4294967296 - 1) IN (3, 47) "
                f"ORDER BY local_id DESC LIMIT 1")
            if not rows:
                return None
            r = rows[0]
            lt = int(r.get("local_type") or 0) & 0xFFFFFFFF
            return Msg(lt, str(r["local_id"]), self._sender_of(r.get("real_sender_id")),
                       room, decode_content(r.get("message_content"), self._zstd),
                       decode_content(r.get("source"), self._zstd))
        except Exception:  # noqa: BLE001
            return None

    def _reread_col(self, msg, col: str = "message_content") -> str:
        """回读消息行的任意列（message_content / source，解码后）。

        微信 4.x 插入消息行与补全 content/source 是两步：通知一到就 SELECT，
        常读到空/半截（@检测的 atuserlist 在 source 里，content 里的表情
        XML/图片 aeskey 同理）。消息只在队列里过一次，错过只能按表名+local_id
        回查。群表也能查（_reread_content 的旧限制已放开）。失败返回空串。"""
        try:
            room = getattr(msg, "roomid", "") or ""
            if not room:
                return ""
            lid = int(msg.id)
            rows = self._query(
                "message_0.db",
                f"SELECT {col} FROM {table_for_room(room)} "
                f"WHERE local_id={lid} LIMIT 1")
            if not rows:
                return ""
            return decode_content(rows[0].get(col), self._zstd)
        except Exception:  # noqa: BLE001  回读是尽力而为，失败走原内容
            return ""

    def _reread_content(self, msg) -> str:
        return self._reread_col(msg, "message_content")

    def fetch_image(self, msg) -> str:
        """type 3 图片：消息 XML 自带 aeskey -> 定位 V2 .dat -> 本地解密。
        .dat 落盘比消息行晚几秒，带短重试；XML 缺 aeskey 时按轮回读消息行
        （通知路径常先拿到空 content，几秒后才补全）。返回 jpg；失败 None。"""
        try:
            root = self._resolve_data_root()
            if not root:
                self.log("取图失败：数据根推断不出（wechat.wxhook.data_root 可固定）")
                return None
            attach = os.path.join(root, "msg", "attach")
            xml = payload_xml(msg, ('aeskey',))
            dst = os.path.join(self.media_dir, f"img_{msg.id}.jpg")
            from .wximg import decode_img_xml
            from urllib.parse import urlparse
            port = urlparse(self.cfg.wxhook.base_url).port or 30001
            cache = os.path.join(os.path.dirname(self.media_dir),
                                 "wechat_image_key.json")
            for attempt in range(5):   # 等 CDN 下载落盘（约 2s x 5）
                if attempt > 0 and 'aeskey="' not in xml:
                    fresh = self._reread_content(msg)   # content 可能迟到，回读
                    if fresh:
                        xml = fresh
                try:
                    fmt = decode_img_xml(dst, xml, attach, log=self.log,
                                         hook_port=port, cache_path=cache)
                except Exception as e:  # noqa: BLE001
                    self.log(f"图片解密异常（第 {attempt + 1} 次）：{e!r}")
                    fmt = None
                if fmt and fmt != "WXGF" and os.path.exists(dst) and os.path.getsize(dst) > 100:
                    return dst
                if fmt == "WXGF":
                    self.log("WXGF(HEVC) 转码失败，跳过")
                    return None
                time.sleep(2.0)
            self.log(f"图片 .dat 未就绪或密钥缺失 msg.id={msg.id}"
                     f"（xml 有 aeskey={'aeskey=' in xml}）")
            return None
        except Exception as e:  # noqa: BLE001
            self.log(f"取图片失败 #{getattr(msg, 'id', '?')}：{e!r}")
            return None

    def fetch_emoji(self, msg):
        """type 47 表情包：XML 的 cdnurl 直接下载（明文图，实测无需解密）。
        静态图直接用；GIF 表情抽样多帧（首/中/尾，共 3 帧）——动画信息在动的过程里，
        只解第一帧会丢（第 0 帧常是空白铺垫帧）。返回 jpg 路径列表；失败返回 None。"""
        import html as _html
        import re as _re
        import subprocess as _sp
        import urllib.request as _ur
        xml = payload_xml(msg, ('cdnurl',))
        m = _re.search(r'cdnurl="([^"]+)"', xml)
        if not m:
            # 通知路径的竞争：message_content 这时可能还没写入，回读消息行
            for _ in range(4):
                time.sleep(1.2)
                fresh = self._reread_content(msg)
                if fresh:
                    xml = fresh
                    m = _re.search(r'cdnurl="([^"]+)"', xml)
                    if m:
                        break
        if not m:
            self.log(f"表情包 XML 一直没有 cdnurl #{msg.id}"
                     f"（len={len(xml)} head={xml[:60]!r}）")
            return None
        url = _html.unescape(m.group(1))
        # CDN 偶发抖动（实测 08:33 一次瞬时失败、重试即好）：三次机会
        data = b""
        for attempt in range(3):
            try:
                req = _ur.Request(url, headers={"User-Agent": "MicroMessenger Client"})
                with _ur.urlopen(req, timeout=20) as resp:
                    data = resp.read(8_000_000)
                if data:
                    break
            except Exception as e:  # noqa: BLE001
                self.log(f"表情包下载失败（第 {attempt + 1} 次）：{e!r}")
                time.sleep(1.0)
        if not data:
            # 兜底：encrypturl + aeskey（两种密钥形态都试，命中即用）
            data = self._fetch_emoji_encrypted(xml) or b""
        if not data:
            return None
        base = os.path.join(self.media_dir, f"emoji_{msg.id}")
        if data[:3] != bytes([0x47, 0x49, 0x46]):   # 静态图：原样一张
            dst = base + ".jpg"
            with open(dst, "wb") as f:
                f.write(data)
            return [dst]
        # GIF：探测时长 -> 抽首/中/尾 3 帧
        tmp = base + ".gif"
        try:
            from .silk_audio import ffmpeg_exe
            ff = ffmpeg_exe()
            with open(tmp, "wb") as f:
                f.write(data)
            probe = _sp.run([ff, "-hide_banner", "-i", tmp],
                            capture_output=True, timeout=30)
            mm = _re.search(r"Duration: *([0-9]+):([0-9]+):([0-9.]+)",
                            (probe.stderr or b"").decode("utf-8", "replace"))
            duration = 0.0
            if mm:
                duration = int(mm.group(1)) * 3600 + int(mm.group(2)) * 60 + float(mm.group(3))
            stamps = [0.0]
            if duration > 0.6:
                stamps = [0.0, round(duration / 2, 2), max(0.0, round(duration - 0.2, 2))]
            outs = []
            for i, t in enumerate(stamps[:3]):
                dst = f"{base}_{i}.jpg"
                r = _sp.run([ff, "-hide_banner", "-loglevel", "error",
                             "-ss", str(t), "-i", tmp, "-frames:v", "1", "-y", dst],
                            capture_output=True, timeout=60)
                if r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 100:
                    outs.append(dst)
            if not outs:   # 时长探测失败兜底：至少给第一帧
                dst = base + "_f0.jpg"
                r = _sp.run([ff, "-hide_banner", "-loglevel", "error",
                             "-i", tmp, "-frames:v", "1", "-y", dst],
                            capture_output=True, timeout=60)
                if r.returncode == 0 and os.path.exists(dst):
                    outs.append(dst)
            if not outs:
                self.log(f"表情包 GIF 抽帧全失败 #{msg.id}"
                         f"（ffmpeg rc={getattr(r, 'returncode', '?')}）")
            return outs or None
        except Exception as e:  # noqa: BLE001
            self.log(f"表情包抽帧失败：{e!r}")
            return None
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass

    def _fetch_emoji_encrypted(self, xml: str) -> bytes:
        """encrypturl + aeskey 兜底下载（裸 AES，两种密钥形态；多数自定义表情
        走不到这——cdnurl 重试已覆盖，这里留给加密方案匹配的商店表情）。"""
        import binascii as _ba
        import re as _re2
        import urllib.request as _ur2
        m_url = _re2.search(r'encrypturl="([^"]+)"', xml or "")
        m_key = _re2.search(r'aeskey="([0-9a-fA-F]{32})"', xml or "")
        if not (m_url and m_key):
            return b""
        import html as _h2
        url = _h2.unescape(m_url.group(1))
        try:
            req = _ur2.Request(url, headers={"User-Agent": "MicroMessenger Client"})
            with _ur2.urlopen(req, timeout=20) as resp:
                data = resp.read(8_000_000)
        except Exception:  # noqa: BLE001
            return b""
        from Crypto.Cipher import AES
        key_hex = m_key.group(1)
        for key in (_ba.unhexlify(key_hex), key_hex.encode()):
            try:
                dec = AES.new(key, AES.MODE_ECB).decrypt(data)
                pad = dec[-1]
                out = dec[:-pad] if 1 <= pad <= 16 and len(dec) > pad else dec
                if out[:3] == bytes([0xFF, 0xD8, 0xFF]) or out[:3] == bytes([0x47, 0x49, 0x46]):
                    return out
            except Exception:  # noqa: BLE001
                continue
        return b""

    def _resolve_data_root(self) -> str:
        if self._data_root and os.path.isdir(os.path.join(self._data_root, "msg")):
            return self._data_root
        # 从 message_0.db 的真实路径向上找账户根（含 msg/ 的那层）。
        # 不能用 contact.db：它在你点开一次「通讯录」前根本没打开（句柄不存在）。
        try:
            rows = self._query("message_0.db", "PRAGMA database_list")
            file = str((rows or [{}])[0].get("file") or "")
            d = os.path.dirname(file)
            for _ in range(5):
                if os.path.isdir(os.path.join(d, "msg")):
                    self._data_root = d
                    return d
                parent = os.path.dirname(d)
                if parent == d:
                    break
                d = parent
            self.log(f"数据根推断失败（{file}），可在 wechat.wxhook.data_root 固定")
        except WxHookError as e:
            self.log(f"推断数据目录失败（可在 wechat.wxhook.data_root 固定）：{e}")
        return ""

    @staticmethod
    def _newest_dat(chat_dir: str, near_ts: float) -> str:
        """chat 目录下（含各月份子目录）最新 .dat；优先原图（无 _t/_h 后缀）。"""
        candidates = []
        for base, _dirs, files in os.walk(chat_dir):
            for f in files:
                if not f.endswith(".dat"):
                    continue
                p = os.path.join(base, f)
                try:
                    mtime = os.path.getmtime(p)
                except OSError:
                    continue
                rank = 2 if f.endswith(".dat") else 0
                if f.endswith("_t.dat"):
                    rank = 1  # 缩略图，原图不可用时兜底
                candidates.append((rank, -abs(mtime - near_ts), p))
        if not candidates:
            return ""
        candidates.sort()
        return candidates[0][2]

    # ---------------- 接收面 ----------------

    def bootstrap_watermarks(self):
        """启动时把“已存在的消息表”水位记到当前最大值（跳过历史）；
        之后新出现的表（新会话/首条消息）按新表处理，从 0 开始拉。"""
        try:
            for table in self._tables():
                if table in self._last_local_id:
                    continue
                rows = self._query("message_0.db",
                                   f"SELECT COALESCE(MAX(local_id),0) AS m FROM {table}")
                self._last_local_id[table] = int(rows[0]["m"]) if rows else 0
            self.log(f"启动水位：已存在 {len(self._last_local_id)} 张消息表")
        except WxHookError as e:
            self.log(f"启动水位失败（不影响轮询）：{e}")

    def enable_receiving(self):
        self.bootstrap_watermarks()
        self._start_notify_server()
        t = threading.Thread(target=self._poll_loop, daemon=True, name="wxhook-poll")
        t.start()
        self._threads.append(t)

    def disable_receiving(self):
        self._stop.set()
        if self._server:
            try:
                self._server.shutdown()
            except Exception:  # noqa: BLE001
                pass

    def get_msg(self, block: bool = True):
        return self._q.get(timeout=1.0 if block else 0.05)

    def _start_notify_server(self):
        channel = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                    body = self.rfile.read(length) if length else b"{}"
                    data = json.loads(body.decode("utf-8", "replace") or "{}")
                except Exception:  # noqa: BLE001
                    data = {}
                # 立刻应答（hook 推送超时 1~2s，绝不阻塞微信线程）
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"ok":true}')
                if data.get("t") == "msg" and data.get("table"):
                    threading.Thread(target=channel.poll_table,
                                     args=(data["table"],), daemon=True).start()

            def log_message(self, fmt, *args):
                pass

        try:
            self._server = ThreadingHTTPServer(
                ("127.0.0.1", self.cfg.wxhook.notify_port), Handler)
            self._server.daemon_threads = True
            threading.Thread(target=self._server.serve_forever, daemon=True,
                             name="wxhook-notify").start()
            self.log(f"通知接收端：127.0.0.1:{self.cfg.wxhook.notify_port}/recv")
        except OSError as e:
            self.log(f"通知端口 {self.cfg.wxhook.notify_port} 起不来（改端口或排查占用）：{e!r}"
                     f"；将只靠兜底轮询收消息")

    def _poll_loop(self):
        # 兜底轮询：默认 60s（0=关）。hook 的 SQL 与客户端写库存在竞争（上游已知未根治），
        # 高频轮询会实际造成库损坏——收消息以通知推送为主，这里只防通知丢失。
        interval = getattr(self.cfg.wxhook, "fallback_poll_seconds", 60.0)
        if not interval or interval <= 0:
            self.log("兜底轮询已关闭（仅通知推送）")
            return
        cycles = 0
        while not self._stop.wait(max(5.0, interval)):
            cycles += 1
            try:
                got = self.poll_tables()
                if got:
                    self.log(f"兜底轮询补收 {got} 条")
                elif cycles % 10 == 0:
                    self.log(f"兜底轮询心跳：0 新（第 {cycles} 周期）")
            except Exception as e:  # noqa: BLE001
                self.log(f"兜底轮询失败：{e!r}")
                # 断连快重试：hook 短暂失联时 5s 一探，恢复即补收，不等下一整轮
                for _ in range(12):
                    if self._stop.wait(5.0):
                        return
                    try:
                        self.poll_tables()
                        self.log("hook 已恢复，兜底轮询回归正常间隔")
                        break
                    except Exception:  # noqa: BLE001
                        continue

    # ---- 表/房间/发送者映射 ----

    def _tables(self) -> list:
        # 表清单缓存 10 分钟：省掉每轮兜底轮询的 sqlite_master 查询（减少竞争面）
        now = time.time()
        if self._tables_cache and now - self._tables_cache_at < 600:
            return list(self._tables_cache)
        rows = self._query("message_0.db",
                           "SELECT name FROM sqlite_master WHERE type='table'"
                           " AND name LIKE 'Msg_%'")
        self._tables_cache = [r.get("name") for r in rows if r.get("name")]
        self._tables_cache_at = now
        return list(self._tables_cache)

    def _refresh_room_map(self):
        # 微信 4.x 的会话表在 session.db（3.9 时代在 message_0.db），两处都试
        rows = []
        for db in ("session.db", "message_0.db"):
            try:
                rows = self._query(db, "SELECT username FROM SessionTable")
                if rows:
                    break
            except WxHookError:
                continue
        for r in rows:
            room = str(r.get("username") or "")
            if room:
                self._room_map[table_for_room(room)] = room

    def _room_of(self, table: str) -> str:
        room = self._room_map.get(table)
        if room is None:
            self._refresh_room_map()
            room = self._room_map.get(table)
        return room or ""

    def _refresh_name2id(self):
        rows = self._query("message_0.db", "SELECT rowid AS rid, user_name FROM Name2Id")
        self._name2id = {int(r["rid"]): str(r.get("user_name") or "") for r in rows}

    def _sender_of(self, rid) -> str:
        if not self._name2id:
            self._refresh_name2id()
        return self._name2id.get(int(rid), "")

    # ---- 拉新消息 ----

    def poll_tables(self) -> int:
        """兜底轮询入口：扫全部私聊表 +（开关打开时的）群表。返回拉到的新消息数。"""
        groups_on = bool(getattr(self.cfg, "groups_enabled", False))
        got = 0
        for table in self._tables():
            room = self._room_of(table)
            if room.endswith("@chatroom") and not groups_on:
                continue
            if not room:
                continue
            before = self._q.qsize()
            self.poll_table(table)
            got += self._q.qsize() - before
        return got

    def poll_table(self, table: str):
        """拉一张 Msg_ 表的新行 -> 标准化 -> 入队。并发安全（通知线程与轮询共用）。"""
        with self._poll_lock:
            try:
                room = self._room_of(table)
                is_group = room.endswith("@chatroom")
                groups_on = bool(getattr(self.cfg, "groups_enabled", False))
                if (is_group and not groups_on) or not room:
                    return
                last = self._last_local_id.get(table)
                if last is None:  # 启动后才出现的表 = 新会话，从 0 开始拉
                    self._last_local_id[table] = 0
                    last = 0
                self_wxid = self.get_self_wxid()
                if not self_wxid:
                    if not self._self_warned:
                        self._self_warned = True
                        self.log("自身 wxid 未解析：暂停处理一切消息（防自回复死循环），"
                                 "将在解析成功后自动恢复")
                    return
                rows = self._query(
                    "message_0.db",
                    f"SELECT {self.POLL_COLUMNS} FROM {table}"
                    f" WHERE local_id > {last} ORDER BY local_id ASC LIMIT 50")
                for r in rows:
                    local_id = int(r.get("local_id") or 0)
                    if not local_id or local_id <= self._last_local_id[table]:
                        continue  # 客户端水位守卫（与 SQL WHERE 双保险）
                    self._last_local_id[table] = local_id
                    sender = self._sender_of(r.get("real_sender_id"))
                    source = decode_content(r.get("source"), self._zstd)
                    # 4.x 的 local_type 是复合编码（高32位=子类型如引用，低32位=基础类型），
                    # 按低 32 位路由（引用消息的 57<<32|49 -> 49 appmsg）
                    msg = Msg(
                        int(r.get("local_type") or 0) & 0xFFFFFFFF, str(local_id), sender, room,
                        decode_content(r.get("message_content"), self._zstd),
                        source,
                        is_self=bool(sender and sender == self_wxid),
                        is_group=is_group,
                        at_me=(is_group and self._at_me(source, self_wxid)))
                    self._q.put(msg)
            except WxHookError as e:
                self.log(f"拉取 {table} 失败：{e}")
            except Exception as e:  # noqa: BLE001
                self.log(f"拉取 {table} 异常：{e!r}")

    @staticmethod
    def _at_me(source: str, self_wxid: str) -> bool:
        """群消息 source 列的 <atuserlist> 是被@者的 wxid 列表（CDATA 可选）——
        @ 的硬信号，不靠昵称文本猜。"""
        import re as _re
        if not source or not self_wxid:
            return False
        m = _re.search(r"atuserlist>(?:<!\[CDATA\[)?([^<]+?)(?:\]\]>)?</atuserlist", source)
        if not m:
            return False
        return self_wxid in [u.strip() for u in m.group(1).split(",") if u.strip()]

    def avatar_bytes(self, wxid: str) -> bytes:
        """wxid -> 头像 bytes（contact.db big_head_url，磁盘缓存按 wxid）。"""
        if not wxid:
            return b""
        cache = os.path.join(self.media_dir, f"avatar_{wxid}.jpg")
        if os.path.exists(cache) and os.path.getsize(cache) > 1000:
            with open(cache, "rb") as f:
                return f.read()
        esc = wxid.replace("'", "''")
        rows = self._query(
            "contact.db",
            f"SELECT big_head_url FROM contact WHERE username='{esc}' LIMIT 1")
        url = str(rows[0].get("big_head_url") or "").replace(chr(92) + "/", "/") if rows else ""
        if not url:
            return b""
        import urllib.request as _u
        raw = _u.urlopen(_u.Request(url, headers={"User-Agent": "Mozilla/5.0"}),
                         timeout=20).read()
        if len(raw) > 1000:
            with open(cache, "wb") as f:
                f.write(raw)
        return raw

    def nickname_of(self, wxid: str) -> str:
        """wxid -> 昵称（contact.db，remark 优先）。查不到返回空串。群聊里
        标注"谁在说话"用；带缓存，失败静默（显示层降级，不影响收发）。"""
        if not wxid:
            return ""
        if wxid in self._nick_cache:
            return self._nick_cache[wxid]
        name = ""
        try:
            esc = wxid.replace("'", "''")
            rows = self._query(
                "contact.db",
                f"SELECT nick_name, remark FROM contact WHERE username='{esc}' LIMIT 1")
            if rows:
                name = str(rows[0].get("remark") or rows[0].get("nick_name") or "")
        except Exception:  # noqa: BLE001  昵称是展示层增强，缺了不挡路
            pass
        self._nick_cache[wxid] = name
        return name
