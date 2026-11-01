"""搜歌+下载（网易云 / 酷我 / B站，免费无 key）。

翻唱链路的第一步：用户说歌名 → 搜到歌 → 下载 mp3 → 交给 vc-worker 换色。

源优先级：
1. 原唱是男声时，优先找明确标注「女声/女生/女版」的翻唱（B站、酷我），
   这样源音频本身就在她的音区，不用移调；
2. 找不到女声版再用原唱（网易云；VIP 失败回退酷我），由 worker 整轨移调。
"""
import ast
import html
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request


class SongSearchError(Exception):
    pass


def _headers(referer="https://music.163.com"):
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": referer,
    }


def _clean(s: str) -> str:
    s = html.unescape(str(s or ""))
    s = s.replace("\\u0026", "&").replace("\\u003c", "<").replace("\\u003e", ">")
    s = re.sub(r"<[^>]+>", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _norm(s: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", str(s or "").lower())


def _duration_s(text) -> int:
    parts = [p for p in str(text or "").split(":") if p.strip().isdigit()]
    if not parts:
        return 0
    total = 0
    for p in parts:
        total = total * 60 + int(p)
    return total


def search_song(query: str, limit: int = 5) -> list:
    """网易云搜歌，返回 [{id, name, artist, duration_s, album}]。"""
    q = urllib.parse.quote(query.strip())
    url = f"https://music.163.com/api/search/get?s={q}&type=1&limit={limit}"
    req = urllib.request.Request(url, headers=_headers())
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception as e:
        raise SongSearchError(f"搜索失败：{e!r}") from None
    songs = data.get("result", {}).get("songs", [])
    out = []
    for s in songs:
        artists = ", ".join(a["name"] for a in s.get("artists", []))
        out.append({
            "id": s["id"],
            "name": s.get("name", ""),
            "artist": artists,
            "duration_s": s.get("duration", 0) // 1000,
            "album": (s.get("album") or {}).get("name", ""),
        })
    return out


def download_song(song_id: int, out_path: str, timeout: int = 30) -> dict:
    """下载网易云 mp3。VIP 歌曲可能返回 404 或空文件。"""
    url = f"https://music.163.com/song/media/outer/url?id={song_id}.mp3"
    req = urllib.request.Request(url, headers=_headers())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            ct = r.headers.get("Content-Type", "")
            raw = r.read(50_000_000)  # ≤50MB
    except urllib.error.HTTPError as e:
        raise SongSearchError(f"下载失败 HTTP {e.code}（可能需要 VIP）") from None
    except Exception as e:
        raise SongSearchError(f"下载失败：{e!r}") from None
    if len(raw) < 50_000 or "audio" not in ct:
        raise SongSearchError(f"下载的内容不是音频（{ct}，{len(raw)//1024}KB，可能需要 VIP）")
    with open(out_path, "wb") as f:
        f.write(raw)
    return {"path": os.path.abspath(out_path), "size": len(raw)}


# ---------------- 酷我 ----------------

def search_kuwo(query: str, limit: int = 8) -> list:
    """酷我搜索，返回 [{source, id, name, artist, duration_s}]。"""
    url = ("http://search.kuwo.cn/r.s?all=%s&ft=music&rformat=json&encoding=utf8"
           "&rn=%d&pn=0" % (urllib.parse.quote(query.strip()), limit))
    req = urllib.request.Request(url, headers=_headers("http://www.kuwo.cn"))
    try:
        raw = urllib.request.urlopen(req, timeout=10).read().decode("utf-8", "ignore")
        data = ast.literal_eval(raw)
    except Exception as e:
        raise SongSearchError(f"酷我搜索失败：{e!r}") from None
    out = []
    for s in data.get("abslist") or []:
        rid = str(s.get("MUSICRID", "")).replace("MUSIC_", "")
        if not rid:
            continue
        out.append({
            "source": "kuwo",
            "id": rid,
            "name": _clean(s.get("SONGNAME")),
            "artist": _clean(s.get("ARTIST")),
            "duration_s": int(s.get("DURATION") or 0),
        })
    return out


def download_kuwo(rid: str, out_path: str, timeout: int = 30) -> dict:
    """酷我直链下载 mp3。"""
    u = ("http://antiserver.kuwo.cn/anti.s?type=convert_url&rid=MUSIC_%s"
         "&format=mp3&response=url" % str(rid).replace("MUSIC_", ""))
    try:
        url = urllib.request.urlopen(
            urllib.request.Request(u, headers=_headers("http://www.kuwo.cn")),
            timeout=10).read().decode("utf-8", "ignore").strip()
    except Exception as e:
        raise SongSearchError(f"酷我取链失败：{e!r}") from None
    if not url.startswith("http"):
        raise SongSearchError(f"酷我没拿到播放地址：{url[:80]}")
    try:
        with urllib.request.urlopen(
                urllib.request.Request(url, headers=_headers("http://www.kuwo.cn")),
                timeout=timeout) as r:
            ct = r.headers.get("Content-Type", "")
            raw = r.read(50_000_000)
    except Exception as e:
        raise SongSearchError(f"酷我下载失败：{e!r}") from None
    if len(raw) < 50_000 or "audio" not in ct:
        raise SongSearchError(f"酷我返回的不是音频（{ct}，{len(raw)//1024}KB）")
    with open(out_path, "wb") as f:
        f.write(raw)
    return {"path": os.path.abspath(out_path), "size": len(raw)}


# ---------------- B站 ----------------

def search_bilibili(query: str, limit: int = 8) -> list:
    """B站视频搜索，返回 [{source, id, name, artist, duration_s, url}]。"""
    url = ("https://api.bilibili.com/x/web-interface/search/type?search_type=video"
           "&keyword=" + urllib.parse.quote(query.strip()))
    req = urllib.request.Request(url, headers=_headers("https://www.bilibili.com/"))
    try:
        with urllib.request.urlopen(req, timeout=12) as r:
            data = json.loads(r.read().decode("utf-8", "ignore"))
    except Exception as e:
        raise SongSearchError(f"B站搜索失败：{e!r}") from None
    out = []
    for v in ((data.get("data") or {}).get("result") or []):
        bvid = v.get("bvid")
        if not bvid:
            continue
        out.append({
            "source": "bilibili",
            "id": bvid,
            "name": _clean(v.get("title")),
            "artist": _clean(v.get("author")),
            "duration_s": _duration_s(v.get("duration")),
            "url": "https://www.bilibili.com/video/" + bvid,
        })
    return out


def download_bilibili(bvid: str, out_path: str) -> dict:
    """yt-dlp 下 B站音轨（m4a），再用 ffmpeg 转 mp3。"""
    tmpdir = tempfile.mkdtemp(prefix="bili_")
    try:
        cmd = [sys.executable, "-m", "yt_dlp", "-f", "bestaudio",
               "--no-playlist", "-o", os.path.join(tmpdir, "%(id)s.%(ext)s"),
               "https://www.bilibili.com/video/" + bvid]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if r.returncode != 0:
            raise SongSearchError(f"B站下载失败：{(r.stderr or '')[-200:]}")
        files = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir)
                 if not f.endswith((".part", ".ytdl"))]
        if not files:
            raise SongSearchError("B站下载没产出文件")
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        conv = subprocess.run(
            [ff, "-hide_banner", "-loglevel", "error", "-y", "-i", files[0],
             "-codec:a", "libmp3lame", "-b:a", "192k", out_path],
            capture_output=True, text=True, timeout=120)
        if conv.returncode != 0 or not os.path.exists(out_path):
            raise SongSearchError(f"B站音轨转 mp3 失败：{(conv.stderr or '')[-200:]}")
        return {"path": os.path.abspath(out_path),
                "size": os.path.getsize(out_path)}
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------- 选源 ----------------

def _female_candidates(title: str, artist: str, limit: int = 4) -> list:
    """找明确标注女声/女生的翻唱，且标题里带原歌名/歌手，避免搜到同名异曲。"""
    title_n, artist_n = _norm(title), _norm(artist)
    if not title_n:
        return []
    queries = [f"{title} {artist} 女声 翻唱".strip(),
               f"{title} {artist} 女生 翻唱".strip(),
               f"{title} 女声版".strip()]
    seen, out = set(), []
    for q in queries:
        for fn in (search_bilibili, search_kuwo):
            try:
                results = fn(q, 8)
            except SongSearchError:
                continue
            for s in results:
                key = (s.get("source"), s.get("id"))
                if key in seen:
                    continue
                name_n = _norm(s.get("name"))
                if title_n not in name_n and name_n not in title_n:
                    continue
                # 标题已经匹配原歌名；翻唱标题经常不写原唱（只写 cover），
                # 不再强制要求 artist，否则会漏掉 B 站大量女声独唱。
                if not any(k in s.get("name", "") for k in ("女声", "女生", "女版")):
                    continue
                if any(k in s.get("name", "").lower()
                       for k in ("伴奏", "instrumental", "karaoke", "纯音乐",
                                 "消音", "伴唱", "无人声", "off vocal",
                                 "钢琴版", "吉他版", "钢琴曲")):
                    continue
                if not (60 <= int(s.get("duration_s") or 0) <= 420):
                    continue
                if _is_duet(s.get("name", "")):
                    continue   # 双人/对唱/和声版：单人 RVC 会被低音声部带偏
                seen.add(key)
                out.append(s)
    # 优先“独唱/单人/女声部分”，其次带“女声”，再次时长接近 3-5 分钟
    out.sort(key=lambda s: (0 if any(k in s["name"] for k in
                                    ("独唱", "单人", "女声部分", "女声版")) else 1,
                            0 if "女声" in s["name"] else 1,
                            0 if 120 <= s["duration_s"] <= 360 else 1))
    return out[:limit]


def _download(song: dict, out_dir: str) -> dict:
    src = song.get("source")
    if src == "kuwo":
        path = os.path.join(out_dir, "src_kuwo_%s.mp3" % song["id"])
        info = download_kuwo(song["id"], path)
    elif src == "bilibili":
        path = os.path.join(out_dir, "src_bili_%s.mp3" % song["id"])
        info = download_bilibili(song["id"], path)
    else:
        path = os.path.join(out_dir, "src_%d.mp3" % song["id"])
        info = download_song(song["id"], path)
    info.update({"name": song["name"], "artist": song["artist"],
                 "duration_s": song["duration_s"], "source": src or "netease",
                 "id": song.get("id")})
    return info


def find_chorus(path: str, duration: float = 30.0, sr: int = 22050) -> float:
    """定位最像副歌/高潮的起点（秒）：响度高 + 段落重复度高。

    用 2 秒步长滑窗，窗口内 RMS 均值 + 与其它非重叠窗口的 chroma 最大相似度，
    取综合分最高的一段；找不到返回 0。
    """
    import numpy as np
    import librosa

    y, _ = librosa.load(path, sr=sr, mono=True)
    if len(y) <= int(duration * sr):
        return 0.0
    hop = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    smooth = max(1, int(round(sr / hop)))          # 约 1 秒
    rms = np.convolve(rms, np.ones(smooth) / smooth, mode="same")
    chroma = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop)
    frames = min(len(rms), chroma.shape[1])
    rms, chroma = rms[:frames], chroma[:, :frames]

    dur_f = max(1, int(duration * sr / hop))
    step = max(1, int(2.0 * sr / hop))
    starts = list(range(0, max(1, frames - dur_f), step))
    if not starts:
        return 0.0
    feats, energies = [], []
    for s in starts:
        energies.append(float(np.mean(rms[s:s + dur_f])))
        c = chroma[:, s:s + dur_f].mean(axis=1)
        feats.append(c / (np.linalg.norm(c) + 1e-9))
    feats = np.asarray(feats)
    energies = np.asarray(energies)
    sim = feats @ feats.T
    rep = np.zeros(len(starts))
    for i, s in enumerate(starts):
        mask = np.abs(np.asarray(starts) - s) >= dur_f
        rep[i] = float(np.max(sim[i][mask])) if mask.any() else 0.0

    def z(a):
        return (a - a.mean()) / (a.std() + 1e-9)

    score = z(energies) + 0.7 * z(rep)
    for i, s in enumerate(starts):                  # 避开最前/最后的纯前奏尾奏
        if s < 0.03 * frames:
            score[i] -= 0.5
        if s + dur_f > 0.97 * frames:
            score[i] -= 0.5
    best_i = int(np.argmax(score))
    # 在最高分附近取最早的一段：优先第一次副歌，而不是最响的末段高潮
    strong = np.where(score >= score[best_i] - 0.5)[0]
    pick = int(strong[0]) if len(strong) else best_i
    return round(starts[pick] * hop / sr, 2)


def get_lyric(song_id: int, timeout: int = 12) -> str:
    """网易云 LRC 歌词（逐行时间轴）。"""
    url = ("https://music.163.com/api/song/lyric?id=%s&lv=1&kv=1&tv=-1"
           % song_id)
    req = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    return ((data.get("lrc") or {}).get("lyric") or "").strip()


def find_lyric_time(lrc: str, query: str):
    """在 LRC 里模糊找一句词，返回 (开始秒, 匹配到的行)；找不到 (None, "")。"""
    import difflib
    q = _norm(query)
    if not q:
        return None, ""
    skip = ("作词", "作曲", "编曲", "制作", "吉他", "贝斯", "和声", "录音",
            "混音", "母带", "监制", "出品", "发行", "词：", "曲：")
    best = None
    for line in (lrc or "").splitlines():
        m = re.match(r"\[(\d+):(\d+(?:\.\d+)?)\](.*)", line)
        if not m:
            continue
        sec = int(m.group(1)) * 60 + float(m.group(2))
        text = m.group(3).strip()
        if not text or any(k in text for k in skip):
            continue
        nt = _norm(text)
        if not nt:
            continue
        if q in nt or nt in q:
            ratio = 1.0
        else:
            ratio = difflib.SequenceMatcher(None, q, nt).ratio()
        if ratio < 0.55:
            continue
        # 同一句多次出现时取最早的一次（第一次副歌/主歌）
        if best is None or ratio > best[0] + 1e-9 or (
                abs(ratio - best[0]) < 1e-9 and sec < best[1]):
            best = (ratio, sec, text)
    if not best:
        return None, ""
    return round(best[1], 2), best[2]


def find_lyric_start(title: str, artist: str, lyric: str,
                     song_id: int = None):
    """按歌词定位起点：优先用已下载网易云歌曲的 id，否则按歌名/歌手搜。"""
    ids = []
    if song_id:
        ids.append(int(song_id))
    else:
        try:
            artist_n = _norm(artist)
            for s in search_song(f"{title} {artist}".strip(), 6):
                an = _norm(s.get("artist", ""))
                if not artist_n or artist_n in an or an in artist_n:
                    ids.append(int(s["id"]))
        except SongSearchError:
            pass
    for sid in ids[:3]:
        try:
            lrc = get_lyric(sid)
        except Exception:  # noqa: BLE001
            continue
        sec, line = find_lyric_time(lrc, lyric)
        if sec is not None:
            return sec, line
    return None, ""


def _is_instrumental(name: str) -> bool:
    n = str(name or "").lower()
    return any(k in n for k in (
        "伴奏", "纯音乐", "instrumental", "karaoke", "消音", "伴唱",
        "无人声", "off vocal", "钢琴版", "吉他版", "钢琴曲"))


_DUET_MARKERS = (
    "双人", "二人", "两人", "合唱", "对唱", "男女", "男声", "女声对唱",
    "两个声部", "分饰两角", "和声版", "duet", "feat", " ft.", "ft ",
)


def _is_duet(name: str) -> bool:
    """双人/对唱/和声版不适合单人 RVC：自动定调会被低音声部带偏。"""
    n = str(name or "").lower()
    return any(k in n for k in _DUET_MARKERS)


_BAD_SOURCE_MARKERS = (
    "suno", "人工智能", "ai改编", "ai cover", "live", "现场", "试听",
    "片段", "剪辑", "demo", "抢听", "预告", "串烧", "组曲", "联唱", "medley",
)


def _is_bad_source(name: str) -> bool:
    """低质量/AI/现场片段源：RVC 会被源本身的电音、混响、截断放大。"""
    n = str(name or "").lower()
    return any(k in n for k in _BAD_SOURCE_MARKERS)


def _actual_duration(path: str) -> float:
    """读真实音频时长；元数据经常是整首，实际下载只有 5 秒试听。"""
    try:
        import soundfile as sf
        return float(sf.info(path).duration)
    except Exception:  # noqa: BLE001  读不到就按 0 处理，后面会跳过
        return 0.0


def find_and_download(query: str, out_dir: str, title: str = "", artist: str = "",
                      prefer_female: bool = False, solo: bool = False,
                      min_duration_s: float = 30.0) -> dict:
    """搜索+下载一步到位。

    prefer_female=True（原唱是男声）时，先找明确标注女声/女生的翻唱；
    solo=True（清唱）时跳过双人/对唱/和声版，避免自动定调被低音声部带偏。
    min_duration_s：下载后按真实音频校验，元数据写整首但实际只有 5 秒
    试听/片段的源直接丢掉。找不到再回退原唱。
    返回 {path, size, name, artist, duration_s, source, is_female}。
    """
    os.makedirs(out_dir, exist_ok=True)
    errors = []

    def _accept(song: dict, female: bool) -> dict:
        info = _download(song, out_dir)
        actual = _actual_duration(info["path"])
        if actual < float(min_duration_s):
            raise SongSearchError(
                "实际只有 %.1fs（需要 ≥%.0fs）" % (actual, min_duration_s))
        info["duration_s"] = int(round(actual))
        info["is_female"] = bool(female)
        return info

    if prefer_female and title:
        for song in _female_candidates(title, artist):
            if solo and _is_duet(song.get("name")):
                errors.append(f"{song.get('name')}: 双人/对唱/和声版跳过")
                continue
            if _is_bad_source(song.get("name")):
                errors.append(f"{song.get('name')}: 低质量/AI/现场片段跳过")
                continue
            try:
                return _accept(song, True)
            except SongSearchError as e:
                errors.append(f"{song.get('source')}/{song.get('name')}: {e}")

    # 原唱：网易云
    try:
        results = search_song(query)
    except SongSearchError as e:
        results, errors = [], errors + [str(e)]
    for song in results[:3]:
        if _is_instrumental(song.get("name")):
            errors.append(f"{song.get('name')}: 伴奏/纯音乐跳过")
            continue
        if not (60 <= int(song.get("duration_s") or 0) <= 420):
            errors.append(f"{song.get('name')}: 时长异常跳过")
            continue
        if solo and _is_duet(song.get("name")):
            errors.append(f"{song.get('name')}: 双人/对唱/和声版跳过")
            continue
        if _is_bad_source(song.get("name")):
            errors.append(f"{song.get('name')}: 低质量/AI/现场片段跳过")
            continue
        try:
            return _accept(song, any(
                k in (song.get("name") or "") for k in ("女声", "女生", "女版")))
        except SongSearchError as e:
            errors.append(f"{song['name']}: {e}")

    # 原唱：酷我兜底
    try:
        results = search_kuwo(query)
    except SongSearchError as e:
        results, errors = [], errors + [str(e)]
    for song in results[:4]:
        if _is_instrumental(song.get("name")):
            errors.append(f"{song.get('name')}: 伴奏/纯音乐跳过")
            continue
        if not (60 <= int(song.get("duration_s") or 0) <= 420):
            errors.append(f"{song.get('name')}: 时长异常跳过")
            continue
        if solo and _is_duet(song.get("name")):
            errors.append(f"{song.get('name')}: 双人/对唱/和声版跳过")
            continue
        if _is_bad_source(song.get("name")):
            errors.append(f"{song.get('name')}: 低质量/AI/现场片段跳过")
            continue
        try:
            return _accept(song, any(
                k in (song.get("name") or "") for k in ("女声", "女生", "女版")))
        except SongSearchError as e:
            errors.append(f"{song['name']}: {e}")

    raise SongSearchError("；".join(errors[-6:]) or f"「{query}」没搜到")


def clip_audio(path: str, out_path: str, start_s: float = 0, duration_s: float = None,
               ffmpeg: str = None) -> str:
    """裁剪音频（翻唱时只取一段）。"""
    if ffmpeg is None:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
           "-i", path, "-ss", str(start_s)]
    if duration_s:
        cmd += ["-t", str(duration_s)]
    cmd += ["-codec:a", "libmp3lame", "-b:a", "192k", out_path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise SongSearchError(f"裁剪失败：{r.stderr[-200:]}")
    return out_path
