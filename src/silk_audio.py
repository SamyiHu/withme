"""音频转 SILK 链路（微信语音气泡的编解码）。

微信语音（type 34）的格式是 `0x02 + #!SILK_V3 + SILK 帧`。编码/解码提供者链
（自动择一，均**无需本地 C 编译器**，按可用性顺序）：

1. `pysilk-mod`（`pip install pysilk-mod`，官方 wheel 覆盖 cp36-cp312/win）
   —— 字节级 encode/decode，输出直接带 0x02 前缀
2. `pilk`（`pip install pilk`，wheel 覆盖到 cp311）—— 上游 wxhook 原方案
3. Node + `silk-wasm`（机器上有 Node 即可用；首次自动 `npm install` 到
   `tools/silk_node/`）—— 纯 WASM，零编译

实测坑（上游 2026-10 真机验证，勿改）：
- 编码假定输入 24kHz/单声道/PCM16（ffmpeg 必须出 24k，否则语音变速）
- pilk.decode 返回值不可信；时长一律用 PCM 字节数反算（24k×2B/s）
"""
import json
import os
import shutil
import subprocess
import time

from .storage import project_root

SILK_MAGIC = b"#!SILK_V3"
PCM_RATE = 24000
PCM_WIDTH = 2
FFMPEG_TIMEOUT = 120
NODE_TIMEOUT = 60


class SilkError(Exception):
    pass


def ffmpeg_exe() -> str:
    """取 ffmpeg：优先 imageio-ffmpeg 静态版，其次 PATH，再次 FFMPEG_EXE。"""
    try:
        import imageio_ffmpeg
        p = imageio_ffmpeg.get_ffmpeg_exe()
        if p and os.path.exists(p):
            return p
    except Exception:  # noqa: BLE001
        pass
    p = os.environ.get("FFMPEG_EXE") or shutil.which("ffmpeg")
    if p and os.path.exists(p):
        return p
    raise SilkError("找不到 ffmpeg：pip install imageio-ffmpeg（不依赖系统 PATH）")


def strip_silk_prefix(data: bytes) -> bytes:
    """剥掉容器前缀（0x02 等），返回从 #!SILK_V3 开始的字节；找不到头原样返回。"""
    off = data.find(SILK_MAGIC)
    return data[off:] if off > 0 else data


def _silk_ok(data: bytes) -> bool:
    return data[:1] == b"\x02" and data[1:10] == SILK_MAGIC or data[:9] == SILK_MAGIC


# ---------------- 编码提供者 ----------------

def _encode_pysilk(pcm: bytes) -> bytes:
    import pysilk
    out = pysilk.encode(pcm, sample_rate=PCM_RATE)
    return bytes(out)


def _encode_pilk(pcm_path: str) -> bytes:
    import pilk
    silk_path = pcm_path + ".silk"
    try:
        try:
            pilk.encode(pcm_path, silk_path, pcm_rate=PCM_RATE)
        except TypeError:  # 某些构建无 pcm_rate 关键字；无参调用默认 24kHz
            pilk.encode(pcm_path, silk_path)
    finally:
        pass
    with open(silk_path, "rb") as f:
        data = f.read()
    os.remove(silk_path)
    return data


def _node_dir() -> str:
    return os.path.join(project_root(), "tools", "silk_node")


def _encode_node(pcm_path: str) -> bytes:
    silk_path = pcm_path + ".nsilk"
    _run_node(["encode", pcm_path, silk_path, str(PCM_RATE)])
    with open(silk_path, "rb") as f:
        data = f.read()
    os.remove(silk_path)
    return data


def _run_node(args: list):
    node = shutil.which("node")
    if not node:
        raise SilkError("没有 Node（silk-wasm 兜底需要）：装 Node 或 pip install pysilk-mod")
    _ensure_node_deps()
    script = os.path.join(_node_dir(), "codec.mjs")
    r = subprocess.run([node, script] + args, capture_output=True, timeout=NODE_TIMEOUT)
    if r.returncode != 0:
        raise SilkError("silk-wasm 失败：" + r.stderr.decode("utf-8", "replace")[:200])


def _ensure_node_deps():
    if os.path.exists(os.path.join(_node_dir(), "node_modules", "silk-wasm")):
        return
    npm = shutil.which("npm")
    if not npm:
        raise SilkError("没有 npm，无法安装 silk-wasm")
    r = subprocess.run([npm, "install", "--silent"], cwd=_node_dir(),
                       capture_output=True, timeout=300)
    if r.returncode != 0 or not os.path.exists(
            os.path.join(_node_dir(), "node_modules", "silk-wasm")):
        raise SilkError("npm install silk-wasm 失败："
                        + r.stderr.decode("utf-8", "replace")[:200])


def _pcm_to_silk(pcm_path: str) -> bytes:
    """24kHz PCM 文件 -> SILK 字节（含微信前缀）。提供者链择一。"""
    with open(pcm_path, "rb") as f:
        pcm = f.read()
    errors = []
    for name, fn in (("pysilk-mod", lambda: _encode_pysilk(pcm)),
                     ("pilk", lambda: _encode_pilk(pcm_path)),
                     ("silk-wasm", lambda: _encode_node(pcm_path))):
        try:
            data = fn()
            if data and _silk_ok(data):
                return data
            errors.append(f"{name}: 输出不是 SILK")
        except Exception as e:  # noqa: BLE001  逐个提供者降级
            errors.append(f"{name}: {e}")
    raise SilkError("无可用 SILK 编码器（" + "；".join(errors)
                    + "）。任选其一：pip install pysilk-mod（推荐，有 wheel）/ "
                      "pip install pilk / 安装 Node（silk-wasm 兜底）")


def mp3_to_silk(mp3: bytes, silk_path: str, work_dir: str = None) -> dict:
    """mp3 字节 -> SILK 文件。返回 {"path", "duration_ms", "bytes"}；失败抛 SilkError。"""
    work_dir = work_dir or os.path.dirname(os.path.abspath(silk_path)) or "."
    os.makedirs(work_dir, exist_ok=True)
    tag = "tts-%d-%s" % (os.getpid(), "%03d" % (time.time() % 1000))
    mp3_path = os.path.join(work_dir, tag + ".mp3")
    pcm_path = os.path.join(work_dir, tag + ".pcm")
    with open(mp3_path, "wb") as f:
        f.write(mp3)
    try:
        ff = ffmpeg_exe()
        r = subprocess.run(
            [ff, "-hide_banner", "-loglevel", "error", "-i", mp3_path, "-vn",
             "-ac", "1", "-ar", str(PCM_RATE), "-acodec", "pcm_s16le",
             "-f", "s16le", "-y", pcm_path],
            capture_output=True, timeout=FFMPEG_TIMEOUT)
        if r.returncode != 0 or not os.path.exists(pcm_path):
            detail = ((r.stderr or b"") + (r.stdout or b"")).decode("utf-8", "replace")[:200]
            raise SilkError(f"mp3->PCM 失败（rc={r.returncode}）：{detail}")
        pcm_bytes = os.path.getsize(pcm_path)
        if pcm_bytes < PCM_RATE * PCM_WIDTH // 10:
            raise SilkError(f"PCM 太小（{pcm_bytes} 字节），音频像是空的")
        if pcm_bytes % 2:
            raise SilkError(f"PCM 字节数不是偶数（{pcm_bytes}）")
        silk = _pcm_to_silk(pcm_path)
        with open(silk_path, "wb") as f:
            f.write(silk)
        duration_ms = round(pcm_bytes / (PCM_RATE * PCM_WIDTH) * 1000)
        return {"path": os.path.abspath(silk_path), "duration_ms": duration_ms,
                "bytes": len(silk)}
    finally:
        for p in (mp3_path, pcm_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except OSError:
                pass


# ---------------- 解码（收语音 -> 16kHz WAV，ASR 用） ----------------

def _pcm_to_wav16(pcm: bytes, wav_path: str, rate: int = 16000) -> str:
    """裸 PCM16/单声道 -> canonical 44 字节头 WAV。"""
    import struct
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " + struct.pack(
        "<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16) + b"data" + struct.pack(
        "<I", len(pcm))
    with open(wav_path, "wb") as f:
        f.write(header + pcm)
    return wav_path


def silk_to_wav16(silk: bytes, wav_path: str) -> str:
    """SILK 字节（含 0x02 前缀）-> 16kHz/mono/PCM16 WAV 文件。提供者链择一。"""
    os.makedirs(os.path.dirname(os.path.abspath(wav_path)) or ".", exist_ok=True)
    errors = []
    try:
        import pysilk  # noqa: F401
        pcm = bytes(pysilk_decode(silk))
        if pcm:
            return _pcm_to_wav16(pcm, wav_path)
        errors.append("pysilk-mod: 空输出")
    except ImportError:
        errors.append("pysilk-mod: 未安装")
    except Exception as e:  # noqa: BLE001
        errors.append(f"pysilk-mod: {e}")
    try:
        import pilk
        silk_path = wav_path + ".silk"
        with open(silk_path, "wb") as f:
            f.write(strip_silk_prefix(silk))
        try:
            pilk.silk_to_wav(silk_path, wav_path, 16000)
            if os.path.exists(wav_path) and os.path.getsize(wav_path) > 44:
                return wav_path
            errors.append("pilk: 输出不可用")
        finally:
            try:
                os.remove(silk_path)
            except OSError:
                pass
    except ImportError:
        errors.append("pilk: 未安装")
    except Exception as e:  # noqa: BLE001
        errors.append(f"pilk: {e}")
    # 兜底：Node silk-wasm 解码成 PCM 再包 WAV
    try:
        tmp_silk = wav_path + ".nsilk"
        tmp_pcm = wav_path + ".npcm"
        with open(tmp_silk, "wb") as f:
            f.write(silk)
        try:
            _run_node(["decode", tmp_silk, tmp_pcm, "16000"])
            with open(tmp_pcm, "rb") as f:
                pcm = f.read()
            if pcm:
                return _pcm_to_wav16(pcm, wav_path)
            errors.append("silk-wasm: 空输出")
        finally:
            for p in (tmp_silk, tmp_pcm):
                try:
                    os.remove(p)
                except OSError:
                    pass
    except Exception as e:  # noqa: BLE001
        errors.append(f"silk-wasm: {e}")
    raise SilkError("无可用 SILK 解码器（" + "；".join(errors) + "）")


def pysilk_decode(silk: bytes) -> bytes:
    import pysilk
    return pysilk.decode(silk, sample_rate=16000)
