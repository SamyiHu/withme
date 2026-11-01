"""唱歌合成：本地 ACE-Step 1.5（acestep.cpp / Vulkan）后端。

链路（服务端异步 job）：
  POST /lm      {caption, lyrics, duration, vocal_language, ...} -> {"id": ...}
  GET  /job?id=N                          -> {"status": "running|done|failed"}
  GET  /job?id=N&result=1                 -> AceRequest JSON（含 audio_codes）
  POST /synth   <上一步的 JSON>            -> {"id": ...}
  GET  /job?id=N&result=1                 -> multipart/mixed（audio + latent）
  GET  /v1/audio?path=...                 （官方 Python 后端的下载路径，这里不用）

设计原则与 imagegen 一致：纯标准库；服务端不可用时抛 SongGenError，由手部
动作转成一句可执行的说明，绝不假装唱了。
"""
import email
import json
import math
import os
import re
import struct
import subprocess
import time
import urllib.error
import urllib.request
import uuid

from .storage import project_root

DEFAULT_LYRICS = (
    "夜色落在窗台\n"
    "你把心事藏起来\n"
    "我轻轻唱给你听\n"
    "明天会好起来"
)

_LINE_TAG_RE = re.compile(
    r"^\s*[\[【](?:verse|chorus|bridge|intro|outro|pre-chorus|hook)[^\]】]*[\]】]\s*$",
    re.I)
_INLINE_TAG_RE = re.compile(r"^\s*[\[【][^\]】]{1,24}[\]】]\s*")


class SongGenError(Exception):
    pass


# ---------------- 时长解析 ----------------

_DUR_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(秒|s|sec|seconds?|分钟|分|min|minutes?)?\s*$",
                     re.I)


def parse_duration(value, default=None, lo: float = 10.0, hi: float = 300.0):
    """'30' / '30秒' / '1分钟' / 30 / 62.5 -> 秒(float)；解析不了返回 default。

    对话里感知层可能给数字也可能给带单位的文本，这里统一收口并夹到
    [lo, hi]（ACE-Step 支持 10-600 秒；上限由 song_gen.max_duration_seconds 决定）。
    """
    if value is None or value == "":
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        sec = float(value)
    else:
        m = _DUR_RE.match(str(value))
        if not m:
            return default
        sec = float(m.group(1))
        unit = (m.group(2) or "").lower()
        if unit.startswith(("分钟", "分", "min")):
            sec *= 60.0
    return max(lo, min(hi, sec))


# ---------------- 歌词整理 ----------------

def sanitize_lyrics(text: str, max_chars: int = 300) -> str:
    """把模型给的歌词整理成 ACE-Step 好唱的短句。

    - 去掉 [Verse]/[Chorus] 结构标签（模型偶尔会把它们写进歌词）
    - 去掉拖腔符号 ～（会被当成字唱出来）
    - 最多 12 行（用户送入的具体歌词可能不止 4 句，不截太狠）；一整段没有
      换行时按中文标点拆句
    - 空/无效回落到内置四句
    """
    raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    raw = raw.replace("～", " ").replace("~", " ")
    lines = []
    for ln in raw.split("\n"):
        ln = ln.strip().strip("“”\"'")
        if not ln or _LINE_TAG_RE.match(ln):
            continue
        ln = _INLINE_TAG_RE.sub("", ln).strip()
        if ln:
            lines.append(ln)
        if len(lines) >= 12:
            break
    if len(lines) == 1 and len(lines[0]) > 18:
        parts = re.split(r"[，。！？；、]", lines[0])
        lines = [p.strip() for p in parts if p.strip()][:12]
    out = "\n".join(lines)[:max_chars].strip()
    return out or DEFAULT_LYRICS


# ---------------- HTTP 基础 ----------------

def _base(cfg) -> str:
    return str(getattr(cfg, "base_url", "") or "").rstrip("/")


def _get(base: str, path: str, timeout: float):
    req = urllib.request.Request(base + path, headers={"Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read(), resp.headers.get("Content-Type", "")


def _post_json(base: str, path: str, obj, timeout: float):
    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        base + path, data=body, method="POST",
        headers={"Content-Type": "application/json; charset=utf-8"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:300]
        except Exception:  # noqa: BLE001
            pass
        raise SongGenError(f"HTTP {e.code}: {detail}") from None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        raise SongGenError(f"请求失败：{e!r}") from None


def health_check(cfg, timeout: float = 3.0) -> bool:
    """本地歌声服务是否活着。mock 永远可用。"""
    if str(getattr(cfg, "provider", "")).lower() == "mock":
        return True
    base = _base(cfg)
    if not base:
        return False
    try:
        body, _ = _get(base, "/health", timeout)
        return json.loads(body.decode("utf-8")).get("status") == "ok"
    except Exception:  # noqa: BLE001
        return False


# ---------------- 异步 job ----------------

def _wait_job(base: str, job_id: str, cfg, logger=None, label: str = "song"):
    deadline = time.time() + float(getattr(cfg, "timeout_seconds", 240))
    interval = max(0.2, float(getattr(cfg, "poll_seconds", 1.0)))
    while time.time() < deadline:
        try:
            body, _ = _get(base, f"/job?id={job_id}", 20)
            state = json.loads(body.decode("utf-8")).get("status")
        except Exception as e:  # noqa: BLE001  瞬时错误继续轮询
            if logger:
                logger(f"{label}: 轮询失败 {e!r}")
            state = None
        if state == "done":
            return
        if state in ("failed", "cancelled"):
            raise SongGenError(f"{label} 任务 {state}")
        if logger:
            logger(f"{label}: {state or 'waiting'}")
        time.sleep(interval)
    raise SongGenError(f"{label} 超时（{cfg.timeout_seconds}s）")


def _extract_audio(body: bytes, content_type: str) -> bytes:
    """从 /synth 的 multipart/mixed 里取音频；直接 audio/* 也兼容。"""
    ctype = (content_type or "").lower()
    if ctype.startswith("audio/"):
        return body
    if "multipart" not in ctype:
        raise SongGenError(f"合成响应不是音频：{content_type!r}")
    msg = email.message_from_bytes(
        b"Content-Type: " + content_type.encode("latin-1", "replace")
        + b"\r\nMIME-Version: 1.0\r\n\r\n" + body)
    for part in msg.walk():
        if part.get_content_maintype() == "audio":
            data = part.get_payload(decode=True)
            if data:
                return data
    raise SongGenError("合成响应里没有音频 part")


# ---------------- 时长 ----------------

def audio_duration_ms(path: str):
    """用 ffmpeg 读真实时长；没有 ffmpeg 返回 None。"""
    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001  可选依赖
        return None
    try:
        r = subprocess.run([ff, "-hide_banner", "-i", path],
                           capture_output=True, text=True, timeout=30)
    except Exception:  # noqa: BLE001
        return None
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", r.stderr or "")
    if not m:
        return None
    h, mi, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
    return int(round((h * 3600 + mi * 60 + s) * 1000))


# ---------------- 提供者 ----------------

def _out_dir(cfg) -> str:
    d = str(getattr(cfg, "out_dir", "data/songs") or "data/songs")
    if not os.path.isabs(d):
        d = os.path.join(project_root(), d)
    os.makedirs(d, exist_ok=True)
    return d


def _acestep_cpp(lyrics: str, style: str, cfg, out_dir: str,
                 voice: str = "", duration: float = None, logger=None) -> dict:
    base = _base(cfg)
    if not base:
        raise SongGenError("song_gen.base_url 未配置")
    caption = style or str(getattr(cfg, "style", "") or "")
    if voice and voice not in caption:
        caption = (caption + "，" + voice).strip("，")
    # 时长：对话里指定 > 配置默认；这里只夹 ACE-Step 的硬限（10-600）。
    # 发送侧（气泡 60s 上限）由桥决定发气泡还是文件，不在生成时砍。
    dur = float(duration) if duration else float(cfg.duration_seconds)
    dur = max(10.0, min(600.0, dur))
    req = {
        "caption": caption,
        "lyrics": lyrics,
        "duration": dur,
        "vocal_language": cfg.vocal_language or "zh",
        "lm_mode": "generate",
        "output_format": cfg.output_format or "mp3",
        "mp3_bitrate": int(getattr(cfg, "mp3_bitrate", 320)),
        "inference_steps": int(cfg.inference_steps),
        "guidance_scale": float(cfg.guidance_scale),
    }
    if cfg.lm_model:
        req["lm_model"] = cfg.lm_model
    if cfg.model:
        req["synth_model"] = cfg.model
    if logger:
        logger(f"唱歌：提交歌词/编曲（{caption[:40]}）")
    job = _post_json(base, "/lm", req, 30)
    job_id = str(job.get("id") or "")
    if not job_id:
        raise SongGenError(f"/lm 未返回 job id：{job}")
    _wait_job(base, job_id, cfg, logger, "lm")
    raw, _ = _get(base, f"/job?id={job_id}&result=1", 60)
    lm = json.loads(raw.decode("utf-8"))
    lm_req = lm[0] if isinstance(lm, list) and lm else lm
    if not isinstance(lm_req, dict):
        raise SongGenError(f"/lm 结果格式异常：{str(lm)[:200]}")
    if logger:
        logger("唱歌：歌词已就绪，开始合成音频")
    job2 = _post_json(base, "/synth", lm, 30)
    job2_id = str(job2.get("id") or "")
    if not job2_id:
        raise SongGenError(f"/synth 未返回 job id：{job2}")
    _wait_job(base, job2_id, cfg, logger, "synth")
    body, ctype = _get(base, f"/job?id={job2_id}&result=1", 120)
    audio = _extract_audio(body, ctype)
    path = os.path.join(out_dir, "song_%s_%s.mp3" % (
        time.strftime("%Y%m%d_%H%M%S"), uuid.uuid4().hex[:6]))
    with open(path, "wb") as f:
        f.write(audio)
    duration_ms = audio_duration_ms(path)
    if not duration_ms:
        try:
            duration_ms = int(float(lm_req.get("duration") or dur) * 1000)
        except (TypeError, ValueError):
            duration_ms = int(dur * 1000)
    return {
        "path": path,
        "duration_ms": duration_ms,
        "lyrics": lm_req.get("lyrics") or lyrics,
        "style": lm_req.get("caption") or caption,
        "voice": voice,
        "provider": "acestep_cpp",
        "meta": {
            "bpm": lm_req.get("bpm"),
            "keyscale": lm_req.get("keyscale"),
            "duration": lm_req.get("duration"),
            "lm_model": lm_req.get("lm_model"),
            "synth_model": lm_req.get("synth_model"),
        },
    }


def _mock(lyrics: str, style: str, cfg, out_dir: str,
          voice: str = "", logger=None) -> dict:
    """零依赖测试用：1 秒 440Hz 正弦波。"""
    path = os.path.join(out_dir, "mock_song.wav")
    import wave
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        frames = bytearray()
        for i in range(16000):
            frames += struct.pack("<h", int(12000 * math.sin(2 * math.pi * 440 * i / 16000)))
        w.writeframes(bytes(frames))
    return {"path": path, "duration_ms": 1000, "lyrics": lyrics,
            "style": style, "voice": voice, "provider": "mock", "meta": {}}


def generate_song(lyrics: str, style: str, cfg, out_dir: str = None,
                  voice: str = "", duration=None, logger=None) -> dict:
    """按配置的 provider 生成一首歌，返回 {path, duration_ms, lyrics, style, voice}。

    duration：'30' / '1分钟' / 90 等；空则用 cfg.duration_seconds，
    统一夹到 [10, song_gen.max_duration_seconds]。"""
    if not getattr(cfg, "enabled", False):
        raise SongGenError("唱歌未开启（song_gen.enabled=false）")
    lyrics = sanitize_lyrics(lyrics, int(getattr(cfg, "max_lyrics_chars", 600)))
    style = str(style or getattr(cfg, "style", "") or "").strip()[:120]
    voice = str(voice or getattr(cfg, "voice", "") or "").strip()[:60]
    dur = parse_duration(duration, default=float(cfg.duration_seconds),
                         lo=10.0, hi=float(getattr(cfg, "max_duration_seconds", 300)))
    out_dir = out_dir or _out_dir(cfg)
    provider = str(getattr(cfg, "provider", "acestep_cpp")).lower()
    if provider == "mock":
        return _mock(lyrics, style, cfg, out_dir, voice, logger)
    if provider == "acestep_cpp":
        return _acestep_cpp(lyrics, style, cfg, out_dir, voice, dur, logger)
    raise SongGenError(f"未知唱歌 provider：{provider}")
