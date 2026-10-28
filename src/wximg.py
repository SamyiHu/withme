"""微信 4.x 收图解密（V2 .dat 格式）——不依赖 hook 的 /Decode_Pic（该接口在
多数构建上是空操作）。方案移植自上游 wxhook 社区（wechat-bot 的
decode-wechat-image.py，其 XOR 算法来自 328336690/wechat-decrypt）；
cfgDword 派生密钥部分移植自 wechatauto（whale-girl 同源，实测 3000/3000）。

V2 .dat 结构：
    [6B 签名 07 08 "V2" 08 07][4B aes_size LE][4B xor_size LE][1B padding]
    [AES-128-ECB 密文][明文段][单字节 XOR 段]
密钥（cfgDword 派生，全档通用，4.1.10.27 活体捕获交叉验证）：
    AES = MD5(str(cfgDword)+wxid)[:16]，XOR = cfgDword&0xFF；
    cfgDword 从 Weixin.exe 的 global_config 结构提取（锚点随版本可能变，
    多候选偏移逐个试）。派生不依赖"本进程显示过图片"，微信刚重启也能用；
    内存扫描（老的候选试解法）与消息 XML aeskey 作兜底。
明文布局注意：AES 区与 XOR 区之间还有 16 字节加密 PKCS 填充块（账号级
常量），不属于明文，解密时跳过——把它混进码流会让 HEVC/JPEG 解码烂掉。
XOR 兜底由文件尾两字节推导（JPEG 以 FF D9 结尾），多文件投票。

依赖：pip install pycryptodome（纯加解密；进程内存读取用 ctypes 标准调用）。
WXGF(HEVC) 明文经 ffmpeg（或 imageio-ffmpeg）转 JPEG。
只读内存、只扫监听 hook 端口的那个微信进程，不碰日常微信。
"""
import ctypes
import glob
import json
import os
import re
import struct
import subprocess
import time
from ctypes import wintypes

V2_MAGIC = b"\x07\x08V2\x08\x07"
RE_KEY16 = re.compile(rb"(?<![a-zA-Z0-9])[a-zA-Z0-9]{16}(?![a-zA-Z0-9])")
RE_KEY32 = re.compile(rb"(?<![a-zA-Z0-9])[a-zA-Z0-9]{32}(?![a-zA-Z0-9])")

PROCESS_VM_READ = 0x0010
PROCESS_QUERY_INFORMATION = 0x0400
MEM_COMMIT = 0x1000
PAGE_NOACCESS = 0x01
PAGE_GUARD = 0x100
PAGE_READWRITE = 0x04
PAGE_WRITECOPY = 0x08
PAGE_EXECUTE_READWRITE = 0x40
PAGE_EXECUTE_WRITECOPY = 0x80

kernel32 = ctypes.windll.kernel32


class MEMORY_BASIC_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BaseAddress", ctypes.c_void_p),
        ("AllocationBase", ctypes.c_void_p),
        ("AllocationProtect", wintypes.DWORD),
        ("RegionSize", ctypes.c_size_t),
        ("State", wintypes.DWORD),
        ("Protect", wintypes.DWORD),
        ("Type", wintypes.DWORD),
    ]


# ---------------- 基础件 ----------------

def find_xor_key(attach_dir: str):
    """单字节 XOR 密钥（V2 尾段）：JPEG 明文以 FF D9 结尾 -> 尾两字节异或可反推，
    对最近的缩略图投票取众数。"""
    counts = {}
    files = sorted(glob.glob(os.path.join(attach_dir, "*", "*", "Img", "*_t.dat")),
                   key=os.path.getmtime, reverse=True)
    for f in files[:32]:
        try:
            sz = os.path.getsize(f)
            if sz < 16:
                continue
            with open(f, "rb") as fp:
                head = fp.read(6)
                fp.seek(sz - 2)
                tail = fp.read(2)
            if head == V2_MAGIC and len(tail) == 2:
                counts[(tail[0], tail[1])] = counts.get((tail[0], tail[1]), 0) + 1
        except OSError:
            continue
    if not counts:
        return None
    (x, y), _ = max(counts.items(), key=lambda kv: kv[1])
    k = x ^ 0xFF
    return k if (y ^ 0xD9) == k else k  # 不一致仍用众数值（上游实测有效）


def cipher_block(attach_dir: str):
    """取最近一个 V2 文件的前 16B 密文（试密钥用）。"""
    pats = [os.path.join(attach_dir, "*", "*", "Img", "*.dat"),
            os.path.join(attach_dir, "*", "Img", "*.dat")]
    files = sorted({f for p in pats for f in glob.glob(p)},
                   key=os.path.getmtime, reverse=True)[:200]
    for f in files:
        try:
            with open(f, "rb") as fp:
                head = fp.read(31)
            if head[:6] == V2_MAGIC and len(head) >= 31:
                return head[15:31]
        except OSError:
            continue
    return None


def try_key(key: bytes, ct: bytes):
    from Crypto.Cipher import AES
    try:
        dec = AES.new(key, AES.MODE_ECB).decrypt(ct)
    except Exception:  # noqa: BLE001
        return None
    for magic, fmt in ((b"\xFF\xD8\xFF", "JPEG"), (b"\x89PNG", "PNG"),
                       (b"RIFF", "WEBP"), (b"GIF", "GIF"), (b"wxgf", "WXGF")):
        if dec.startswith(magic):
            return fmt
    return None


# ---------------- 进程内存找 AES 密钥 ----------------

def bot_exe_from_port(port: int):
    """监听 hook 端口的那个 Weixin.exe 的路径（= 机器人客户端本体）。"""
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         f"(Get-NetTCPConnection -LocalPort {port} -State Listen -EA SilentlyContinue)"
         ".OwningProcess | Select-Object -First 1"],
        capture_output=True, text=True, timeout=20)
    pid = (out.stdout or "").strip().split("\n")[0].strip()
    if not pid.isdigit():
        return None
    out2 = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         f"(Get-CimInstance Win32_Process -Filter \"ProcessId={pid}\").ExecutablePath"],
        capture_output=True, text=True, timeout=20)
    path = (out2.stdout or "").strip()
    return path or None


def bot_pids(bot_exe: str):
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process -Filter \"Name='Weixin.exe'\" | "
         f"Where-Object {{ $_.ExecutablePath -eq '{bot_exe}' }} | "
         "Select-Object ProcessId,WorkingSetSize | ConvertTo-Json -Compress"],
        capture_output=True, text=True, timeout=30)
    try:
        data = json.loads((out.stdout or "").strip() or "[]")
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = [data]
    data.sort(key=lambda x: x.get("WorkingSetSize", 0), reverse=True)
    return [(int(x["ProcessId"]), int(x.get("WorkingSetSize", 0))) for x in data]


def _scan_regions(h, regions, ct, budget_s=180):
    t0 = time.time()
    for base, size in regions:
        if time.time() - t0 > budget_s or size <= 0 or size > 64 * 1024 * 1024:
            continue
        buf = ctypes.create_string_buffer(size)
        read = ctypes.c_size_t(0)
        if not kernel32.ReadProcessMemory(h, ctypes.c_void_p(base), buf, size,
                                          ctypes.byref(read)):
            continue
        data = buf.raw[:read.value]
        for m in RE_KEY16.finditer(data):
            if try_key(m.group(), ct):
                return m.group().decode("ascii")
        for m in RE_KEY32.finditer(data):
            if try_key(m.group()[:16], ct):
                return m.group()[:16].decode("ascii")
    return None


def find_aes_key(bot_exe: str, attach_dir: str, log=None):
    ct = cipher_block(attach_dir)
    if not ct:
        return None
    for pid, ws in bot_pids(bot_exe):
        h = kernel32.OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFORMATION, False, pid)
        if not h:
            continue
        try:
            addr = 0
            mbi = MEMORY_BASIC_INFORMATION()
            rw, allr = [], []
            while addr < 0x7FFFFFFFFFFF:
                if kernel32.VirtualQueryEx(h, ctypes.c_void_p(addr), ctypes.byref(mbi),
                                           ctypes.sizeof(mbi)) == 0:
                    break
                if (mbi.State == MEM_COMMIT and mbi.Protect != PAGE_NOACCESS
                        and not (mbi.Protect & PAGE_GUARD)
                        and mbi.RegionSize <= 64 * 1024 * 1024):
                    reg = (mbi.BaseAddress or 0, mbi.RegionSize)
                    allr.append(reg)
                    if mbi.Protect & (PAGE_READWRITE | PAGE_WRITECOPY
                                      | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY):
                        rw.append(reg)
                nxt = addr + mbi.RegionSize
                if nxt <= addr:
                    break
                addr = nxt
            if log:
                log(f"扫 PID {pid}（{ws // 1048576}MB，可写区 {len(rw)}/{len(allr)}）找图片密钥…")
            key = _scan_regions(h, rw, ct) or _scan_regions(h, allr, ct)
            if key:
                return key
        finally:
            kernel32.CloseHandle(h)
    return None


def load_key(cache_path: str, bot_exe: str, attach_dir: str, log=None) -> str:
    try:
        with open(cache_path, encoding="utf-8") as fp:
            k = json.load(fp).get("image_aes_key")
        if k:
            return k
    except (OSError, json.JSONDecodeError):
        pass
    k = find_aes_key(bot_exe, attach_dir, log=log)
    if k:
        try:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as fp:
                json.dump({"image_aes_key": k, "found_at": int(time.time())}, fp)
            if log:
                log(f"图片 AES 密钥已找到并缓存")
        except OSError:
            pass
    return k


# ---------------- cfgDword 派生密钥（whale-girl/wechatauto 同源方案） ----------------
# 账号级图片密钥不扫内存也能拿：cfgDword 存在 Weixin.exe 的 global_config 结构里，
# AES=MD5(str(cfgDword)+wxid)[:16]，XOR=cfgDword&0xFF（上游实测 3000/3000 全对）。
# 好处：不依赖"本进程显示过图片"，微信刚重启、只剩缩略图时照样可解。
CFG_LANDMARK = b"global_config"   # cfg 对象地标字符串（SSO 内联）
CFG_PTR_BACKS = (0x138, 0x130, 0x128, 0x140)   # 地标前指针链回退偏移（版本敏感：
                                  # 4.1.10.27/31 实测 0x130，0x138 为上游默认，逐个试）
CFG_OFFSET = 0x68                 # v18 -> cfg 指针偏移（版本敏感）
CFG_DWORD_OFF = 0x40              # cfgDword
CFG_WXID_OFF = 0x48               # wxid std::string


class _MODULEENTRY32W(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("th32ModuleID", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("GlblcntUsage", wintypes.DWORD),
        ("ProccntUsage", wintypes.DWORD),
        ("modBaseAddr", ctypes.c_void_p),
        ("modBaseSize", wintypes.DWORD),
        ("hModule", wintypes.HMODULE),
        ("szModule", ctypes.c_wchar * 256),
        ("szExePath", ctypes.c_wchar * 260),
    ]


def _weixin_module(h, pid):
    """进程内 weixin.dll 的 (基址, 大小)。"""
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    k32.CloseHandle.argtypes = [ctypes.c_void_p]
    k32.Module32FirstW.argtypes = [ctypes.c_void_p, ctypes.POINTER(_MODULEENTRY32W)]
    k32.Module32NextW.argtypes = [ctypes.c_void_p, ctypes.POINTER(_MODULEENTRY32W)]
    snap = k32.CreateToolhelp32Snapshot(0x08 | 0x10, pid)
    if not snap or snap == -1 or snap == (1 << 64) - 1:
        return None
    try:
        me = _MODULEENTRY32W()
        me.dwSize = ctypes.sizeof(me)
        ok = k32.Module32FirstW(snap, ctypes.byref(me))
        while ok:
            if me.szModule and me.szModule.lower() == "weixin.dll":
                return (me.modBaseAddr or 0), me.modBaseSize
            me.dwSize = ctypes.sizeof(me)
            ok = k32.Module32NextW(snap, ctypes.byref(me))
    finally:
        k32.CloseHandle(snap)
    return None


def _remote_read_factory(h):
    def read(addr, n):
        if not addr or n <= 0:
            return None
        buf = ctypes.create_string_buffer(n)
        got = ctypes.c_size_t(0)
        if not kernel32.ReadProcessMemory(h, ctypes.c_void_p(addr), buf,
                                          n, ctypes.byref(got)) or not got.value:
            return None
        return buf.raw[:got.value]
    return read


def _remote_string(read, addr):
    """跨进程读 MSVC x64 std::string（SSO：size<=15 内联，否则堆指针）。"""
    sz = read(addr + 16, 8)
    if not sz:
        return ""
    n = struct.unpack_from("<Q", sz)[0]
    if n <= 0 or n > 0x7FFFFFFF:
        return ""
    if n <= 15:
        data = read(addr, n)
    else:
        p = read(addr, 8)
        if not p:
            return ""
        data = read(struct.unpack_from("<Q", p)[0], n)
    return (data or b"")[:n].decode("utf-8", "replace")


def extract_cfg_dword(pid: int, log=None):
    """从 Weixin.exe 提取 (cfgDword, wxid)。锚点/偏移每版本可能变，失败返回 None。"""
    h = kernel32.OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFORMATION, False, pid)
    if not h:
        return None
    try:
        mod = _weixin_module(h, pid)
        if not mod:
            return None
        base, size = mod
        if not base or not size or size >= 0x40000000:
            return None
        read = _remote_read_factory(h)
        image = read(base, size)
        if not image or len(image) != size:
            return None
        # 倒序扫 global_config 地标（SSO：size==13 且 cap|0xF==0xF）
        pos = -1
        for i in range(len(image) - 8, 0, -8):
            if struct.unpack_from("<I", image, i)[0] == len(CFG_LANDMARK):
                cap = struct.unpack_from("<I", image, i + 8)[0]
                if cap and (cap | 0xF) == 0xF and i - 16 >= 0:
                    if image[i - 16:i - 16 + len(CFG_LANDMARK)] == CFG_LANDMARK:
                        pos = i
                        break
        if pos < 0:
            if log:
                log("cfgDword 提取：没找到 global_config 地标（版本变了？）")
            return None
        for back in CFG_PTR_BACKS:
            v18b = read(base + pos - back, 8)
            if not v18b:
                continue
            v18 = struct.unpack_from("<Q", v18b)[0]
            if not (0x10000000000 < v18 < 0x7FF000000000):
                continue
            cfgb = read(v18 + CFG_OFFSET, 8)
            if not cfgb:
                continue
            cfg = struct.unpack_from("<Q", cfgb)[0]
            if not (0x10000 <= cfg < 0x800000000000):
                continue
            dwb = read(cfg + CFG_DWORD_OFF, 4)
            if not dwb:
                continue
            dw = struct.unpack_from("<I", dwb)[0]
            wxid = _remote_string(read, cfg + CFG_WXID_OFF)
            # wxid 是 4~64 位可打印 ASCII（自定义微信号不一定带 wxid_ 前缀）
            if (dw and wxid and 4 <= len(wxid) <= 64
                    and all(32 < ord(c) < 127 for c in wxid)):
                return dw, wxid
        return None
    finally:
        kernel32.CloseHandle(h)


def derive_image_keys(cfg_dword: int, wxid: str):
    """cfgDword -> (AES 16 字节, XOR 单字节)。"""
    import hashlib
    aes = hashlib.md5(("%d" % cfg_dword + wxid).encode("utf-8")).hexdigest()[:16]
    return aes.encode("ascii"), cfg_dword & 0xFF


def account_keys(attach_dir: str, hook_port: int = None, cache_path: str = None,
                 log=None):
    """账号级图片密钥 (aes_bytes, xor_int)，三级来源：
    缓存 -> cfgDword 派生（主路，不依赖浏览历史）-> 内存扫描（旧路兜底）。
    派生结果用最近 .dat 首块验魔数；拿不到样本时信任派生（后续解码自会验）。"""
    ct = cipher_block(attach_dir)
    if cache_path:
        try:
            with open(cache_path, encoding="utf-8") as fp:
                d = json.load(fp)
            aes = str(d.get("image_aes_key", "")).encode("ascii")
            xor = d.get("image_xor_key")
            if len(aes) == 16 and xor is not None and (not ct or try_key(aes, ct)):
                return aes, int(xor)
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    def _cache(aes, xor):
        if not cache_path:
            return
        try:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as fp:
                json.dump({"image_aes_key": aes.decode("ascii"),
                           "image_xor_key": int(xor),
                           "found_at": int(time.time())}, fp)
        except OSError:
            pass

    exe = bot_exe_from_port(hook_port) if hook_port else None
    if exe:
        for pid, _ws in bot_pids(exe):
            got = extract_cfg_dword(pid, log=log)
            if not got:
                continue
            dw, wxid = got
            aes, xor = derive_image_keys(dw, wxid)
            if ct and not try_key(aes, ct):
                if log:
                    log(f"cfgDword 派生密钥验证未过（dw={dw}），退内存扫描")
                continue
            if log:
                log(f"图片密钥走 cfgDword 派生（wxid={wxid}）")
            _cache(aes, xor)
            return aes, xor
        k = find_aes_key(exe, attach_dir, log=log)
        if k:
            xor = find_xor_key(attach_dir)
            _cache(k.encode("ascii"), xor if xor is not None else 0)
            return k.encode("ascii"), xor
    return None


def wxgf_to_jpg(data: bytes) -> bytes:
    """wxgf（HEVC 容器）-> JPEG 首帧，靠 ffmpeg。失败返回空 bytes。

    真码流从 VPS/SPS/PPS 起始码切入：容器元数据（ICC 色块等）里也有
    00 00 00 01 假阳性，从第一个起始码解出来是绿屏（HEVC 缺 slice 的填充色）。"""
    import shutil
    import subprocess
    import tempfile
    start = -1
    for m in re.finditer(re.escape(b"\x00\x00\x00\x01"), data):
        nxt = data[m.end()] if m.end() < len(data) else 0
        if ((nxt >> 1) & 0x3F) in (32, 33, 34):   # VPS / SPS / PPS
            start = m.start()
            break
    if start < 0:
        start = data.find(b"\x00\x00\x00\x01")
    if start < 0:
        return b""
    exe = shutil.which("ffmpeg")
    if not exe:
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:  # noqa: BLE001
            return b""
    try:
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "in.hevc")
            dst = os.path.join(td, "out.jpg")
            with open(src, "wb") as f:
                f.write(data[start:])
            r = subprocess.run([exe, "-y", "-v", "error", "-i", src,
                                "-frames:v", "1", dst],
                               capture_output=True, timeout=30)
            if r.returncode == 0 and os.path.exists(dst):
                with open(dst, "rb") as f:
                    out = f.read()
                return out if out[:3] == b"\xFF\xD8\xFF" else b""
    except Exception:  # noqa: BLE001
        return b""
    return b""


# ---------------- 解密 ----------------

def decode_dat(src: str, dst: str, aes_key: bytes, xor_key):
    """V2 .dat -> 明文图片文件（aes_key/xor_key 用 cfgDword 派生，见 account_keys）。

    4.1.10.x 实测（活体解码缓冲捕获交叉验证）：明文 = [AES-128-ECB 头
    aes_size 字节（含 PKCS 填充不剪）] + [剩余区域 ^ xor_key 全区单字节]；
    AES 区与 XOR 区之间的 16 字节是加密过的 PKCS 填充块（解密恒为 0x10×16，
    账号级常量），不属于明文，跳过——此前"解不出大图"就是把它混进了码流。
    WXGF 返回原样字节（调用方用 wxgf_to_jpg 转码）。"""
    from Crypto.Cipher import AES
    with open(src, "rb") as fp:
        data = fp.read()
    if data[:6] != V2_MAGIC:
        raise ValueError("不是 V2 格式: " + src)
    _sig, aes_size, xor_size = struct.unpack_from("<6sLL", data)
    aligned = aes_size + ((16 - aes_size % 16) % 16)
    key = aes_key[:16]
    dec = AES.new(key, AES.MODE_ECB).decrypt(data[15:15 + aligned])[:aes_size]
    region = data[15 + aes_size + 16:15 + aes_size + 16 + xor_size]
    out = dec + bytes(b ^ xor_key for b in region)
    fmt = "unknown"
    if out[:3] == b"\xFF\xD8\xFF":
        fmt = "JPEG"
    elif out[:4] == b"\x89PNG":
        fmt = "PNG"
    elif out[:4] == b"RIFF":
        fmt = "WEBP"
    elif out[:3] == b"GIF":
        fmt = "GIF"
    elif out[:4] == b"wxgf":
        fmt = "WXGF"
    with open(dst, "wb") as fp:
        fp.write(out)
    return fmt

def decode_img_xml(dst: str, xml: str, attach_dir: str, log=None,
                   hook_port: int = None, cache_path: str = None):
    """解密图片消息对应的 .dat。
    XML: <img aeskey="32位hex" md5="文件名"/>；md5 定位文件，aeskey 是该图的
    CDN 下载密钥（对常规档/原件有效）。三档取舍：
      _h.dat(原件) / .dat(常规) 用消息 aeskey；
      只剩 _t.dat(缩略图) 时用账号级派生密钥（缩略图不认 per-image aeskey，
      家族群"只有缩略图"的失败就是这么来的），aeskey 缺失时同样走账号密钥。
    WXGF(HEVC) 自动过 ffmpeg 转 JPEG。
    返回格式名（JPEG/PNG/…）；缺密钥或缺文件返回 None（调用方可重试等下载）。"""
    import binascii
    m_ak = re.search(r'aeskey="([0-9a-fA-F]{32})"', xml or "")
    m_md = re.search(r'\bmd5="([0-9a-fA-F]{32})"', xml or "")
    md5 = m_md.group(1) if m_md else None
    xml_key = binascii.unhexlify(m_ak.group(1)) if m_ak else None
    # 三档定位（同档多命中取最新）。md5 不可靠时退"最近落盘"启发式：
    #   - 4.x 直发图片消息的 img 标签经常没有 md5 属性；
    #   - 有 md5 时也可能是原图内容 md5，与盘上 CDN 文件名对不上。
    # 批处理在消息后几秒内解密，目录里最新的就是它——只认 5 分钟内的文件防老图误配
    found = {}
    if md5:
        for suffix, name in (("_h", "orig"), ("", "main"), ("_t", "thumb")):
            hits = glob.glob(os.path.join(attach_dir, "*", "*", "Img",
                                          md5 + suffix + ".dat"))
            if hits:
                found[name] = max(hits, key=os.path.getmtime)
    if not found:
        now = time.time()
        imgs = [f for f in glob.glob(os.path.join(attach_dir, "*", "*", "Img", "*.dat"))
                if now - os.path.getmtime(f) < 300]
        mains = [f for f in imgs
                 if f.endswith(".dat") and not f.endswith(("_t.dat", "_h.dat"))]
        if mains:
            found["main"] = max(mains, key=os.path.getmtime)
        thumbs = [f for f in imgs if f.endswith("_t.dat")]
        if thumbs:
            found["thumb"] = max(thumbs, key=os.path.getmtime)
    if not found:
        return None
    tail_xor = find_xor_key(attach_dir)

    attempts = []   # (说明, 文件, aes, xor)
    acct = account_keys(attach_dir, hook_port=hook_port,
                        cache_path=cache_path, log=log)
    if "main" in found or "orig" in found:
        src = found.get("main") or found["orig"]
        if acct:
            attempts.append(("账号派生密钥", src, acct[0], acct[1]))
        if xml_key:
            attempts.append(("消息aeskey", src, xml_key, tail_xor))
    if "thumb" in found:
        if acct:
            attempts.append(("账号派生密钥(缩略图)", found["thumb"], acct[0], acct[1]))
        if xml_key:
            attempts.append(("消息aeskey(缩略图)", found["thumb"], xml_key, tail_xor))

    fmt = None
    incomplete = False
    for desc, src, aes, xk in attempts:
        try:
            fmt = decode_dat(src, dst, aes, xk)
        except Exception as e:  # noqa: BLE001  换下一组密钥/档位
            if log:
                log(f"图片解密（{desc}）异常：{e!r}")
            fmt = None
            continue
        if fmt and fmt != "unknown":
            if fmt.startswith("?"):
                incomplete = True
                fmt = fmt[1:]
            if log:
                log(f"图片解密 {fmt}（{desc}）: {os.path.basename(src)}"
                    f"{'（密钥流不够长，部分未解）' if incomplete else ''}"
                    f" -> {os.path.basename(dst)}")
            break
        fmt = None   # 密钥不对（解不出图片魔数），试下一组
    if fmt == "WXGF":
        with open(dst, "rb") as f:
            jpg = wxgf_to_jpg(f.read())
        if jpg:
            with open(dst, "wb") as f:
                f.write(jpg)
            fmt = "JPEG"
            if log:
                log("wxgf(HEVC) 已转 JPEG")
    if incomplete and fmt:
        fmt = "?" + fmt
    return fmt
