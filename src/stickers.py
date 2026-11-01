"""表情包库（贴纸）：按情绪/场景检索 + 按需下载 + 桥接发送。

数据源：kiriyuki.com（作者：B站 可以叫我云大人）的贴纸包，元数据随仓库放在
assets/stickers/nekowhale.json（来自 dsh-nekowhale-stickers 插件，MIT/资产许可见
同目录 NOTICE / LICENSE-ASSETS）。

用法：
    python -m src.main stickers sync     # 拉取配置里几个包的图片到 data/stickers/
    python -m src.main stickers search 摸鱼
模型侧不需要直接调这里：回复里写 `[sticker:摸鱼]`，桥会换成真表情图。
"""
import hashlib
import json
import os
import random
import re
import urllib.request

from .storage import project_root

CJK = re.compile(r"[\u3400-\u9fff]")

# 检索权重（移植自 dsh-nekowhale-stickers 的 core/search.ts）
_WEIGHTS = (("name", 6), ("tags", 5), ("emotion", 4), ("use", 3),
            ("desc", 1), ("en", 2), ("meme", 3))


def _bigrams(s):
    return [s[i:i + 2] for i in range(len(s) - 1)]


def _terms(q):
    return [t for t in re.split(r"[\s,，。.!！?？、;；:：/|]+", (q or "").lower()) if t]


# 模型写 [sticker:笑死]/[sticker:累] 这类口语词时，映射到库里的情绪标签
_SYNONYMS = {
    "笑死": "大笑", "好笑": "大笑", "搞笑": "大笑", "乐": "大笑", "绷不住": "大笑",
    "开心": "开心", "高兴": "开心", "爽": "开心", "摸鱼": "开心", "偷懒": "开心",
    "累": "疲惫", "累死": "疲惫", "困": "疲惫", "加班": "疲惫",
    "无语": "无语", "翻白眼": "无语", "服了": "无语",
    "生气": "生气", "怒": "生气", "烦": "生气", "气死": "生气",
    "难过": "难过", "emo": "难过", "委屈": "难过",
    "害羞": "害羞", "不好意思": "害羞",
    "惊讶": "惊讶", "震惊": "惊讶", "惊了": "惊讶",
    "疑惑": "疑惑", "好奇": "疑惑", "问号": "疑惑",
    "得意": "得意", "骄傲": "得意", "炫耀": "得意",
    "可爱": "卖萌", "卖萌": "卖萌", "撒娇": "卖萌",
    "安慰": "安慰", "抱抱": "安慰", "心疼": "安慰",
    "感谢": "感谢", "谢谢": "感谢",
    "打招呼": "打招呼", "在吗": "打招呼", "嗨": "打招呼",
}


class StickerStore:
    """贴纸索引 + 按需下载。线程内使用即可（下载串行）。"""

    def __init__(self, meta_path: str = None, media_dir: str = None, site: str = None,
                 packs: list = None, enabled: bool = True, log=None,
                 ffmpeg_path: str = "", send_mode: str = "static"):
        self.ffmpeg_path = ffmpeg_path or ""
        self.send_mode = send_mode or "static"
        self.enabled = bool(enabled)
        self.log = log or (lambda m: None)
        self.site = (site or "https://kiriyuki.com").rstrip("/")
        self.packs = list(packs or [])
        self.dir = media_dir or os.path.join(project_root(), "data", "stickers")
        self.meta_path = meta_path or os.path.join(
            project_root(), "assets", "stickers", "nekowhale.json")
        self.items = []
        self.by_id = {}
        self._manifest = None
        self._cache = {}
        self._load()

    # ---------------- 索引 ----------------
    def _load(self):
        if not self.enabled:
            return
        try:
            with open(self.meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            self.log(f"贴纸元数据加载失败：{e!r}")
            return
        for s in data.get("stickers") or []:
            sid = str(s.get("id") or "")
            if not sid:
                continue
            pack = sid.split("/")[0]
            item = {
                "id": sid, "pack": pack,
                "name": sid.split("/")[-1],
                "file": sid.split("/")[-1] + ".gif",
                "emotion": s.get("emotion") or "",
                "tags": list(s.get("tags") or []),
                "use": s.get("use") or "", "desc": s.get("desc") or "",
                "meme": s.get("meme") or "", "en": s.get("en") or "",
                "dir": s.get("dir") or "", "hash": s.get("hash") or "",
            }
            self.items.append(item)
            self.by_id[sid] = item
        if self.packs:
            self.items = [i for i in self.items if i["pack"] in self.packs]
        self.log(f"贴纸库：{len(self.items)} 张（包：{self.packs or '全部'}）")

    # ---------------- 检索 ----------------
    def _fields(self, it):
        return {
            "name": it["name"].lower(),
            "tags": " ".join(it["tags"]).lower(),
            "emotion": (it["emotion"] or "").lower(),
            "use": (it["use"] or "").lower(),
            "desc": (it["desc"] or "").lower(),
            "en": (it["en"] or "").lower(),
            "meme": (it["meme"] or "").lower(),
        }

    def score(self, it, qs):
        f = self._cache.get(it["id"])
        if f is None:
            f = self._cache[it["id"]] = self._fields(it)
        total = 0
        for q in qs:
            best = 0
            for key, w in _WEIGHTS:
                v = f.get(key) or ""
                if not v:
                    continue
                if q in v:
                    best = max(best, w * (1.5 if v == q else 1.0))
                elif CJK.search(q) and len(q) > 2:
                    bg = _bigrams(q)
                    hit = sum(1 for b in bg if b in v)
                    if hit:
                        best = max(best, w * hit / len(bg) * 0.8)
            total += best
        return total

    def search(self, query: str, limit: int = 3) -> list:
        if not self.enabled or not self.items:
            return []
        qs = _terms(query)
        if not qs:
            return []
        scored = [(self.score(it, qs), it) for it in self.items]
        scored = [(s, it) for s, it in scored if s > 0]
        scored.sort(key=lambda x: -x[0])
        return [it for _s, it in scored[:limit]]

    def pick(self, query: str):
        """按名字/情绪/场景词挑一张；完全没命中就按情绪随机，再不行随机。"""
        if not self.enabled or not self.items:
            return None
        q = (query or "").strip()
        if not q:
            return None
        for it in self.items:
            if it["name"] == q or it["id"] == q:
                return it
        hits = self.search(q, limit=8)
        # 名字/标签里直接包含查询词的，先用最匹配那张（别把"摸鱼"随机成"嚼薯片"）
        for it in hits:
            if q in it["name"] or any(q in tg for tg in it["tags"]):
                return it
        exact = [it for it in self.items if it["emotion"] == q]
        if exact:
            return random.choice(exact)
        emo = _SYNONYMS.get(q)
        if emo:
            same = [it for it in self.items if it["emotion"] == emo]
            if same:
                return random.choice(same)
        return random.choice(hits[:3]) if hits else None

    # ---------------- 下载 ----------------
    def _manifest_items(self):
        if self._manifest is not None:
            return self._manifest
        url = self.site + "/api/manifest"
        try:
            with urllib.request.urlopen(url, timeout=25) as r:
                data = json.loads(r.read().decode("utf-8", "replace"))
            self._manifest = data.get("items") or []
            self._gif_base = (data.get("gif_base") or "/media/gif").rstrip("/") + "/"
        except Exception as e:  # noqa: BLE001  离线时用本地缓存
            self.log(f"贴纸 manifest 拉取失败：{e!r}")
            self._manifest = []
        return self._manifest

    def local_path(self, item) -> str:
        safe = item["id"].replace("/", "__")
        return os.path.join(self.dir, safe + ".gif")

    def download(self, item) -> str:
        """下载单张到本地缓存，返回路径（失败返回空串）。"""
        path = self.local_path(item)
        if os.path.exists(path) and os.path.getsize(path) > 512:
            return path
        manifest = self._manifest_items()
        base = self.site + getattr(self, "_gif_base", "/media/gif/")
        target = None
        for m in manifest:
            if m.get("id") == item["id"]:
                target = m
                break
        if target is None:
            self.log(f"贴纸不在站内清单里：{item['id']}")
            return ""
        # 站内布局是 /media/gif/<pack>/<file>；文件名是中文，必须百分号编码
        # （否则 urllib 的请求行是 ascii 会直接炸）
        import urllib.parse
        url = (base + urllib.parse.quote(item["pack"]) + "/"
               + urllib.parse.quote(target["file"]))
        try:
            os.makedirs(self.dir, exist_ok=True)
            with urllib.request.urlopen(url, timeout=40) as r:
                raw = r.read()
            want = (target.get("hash") or "")
            if want and hashlib.sha1(raw).hexdigest()[:12] != want:
                self.log(f"贴纸校验失败：{item['id']}")
                return ""
            with open(path, "wb") as f:
                f.write(raw)
            return path
        except Exception as e:  # noqa: BLE001
            self.log(f"贴纸下载失败 {item['id']}：{e!r}")
            return ""

    def _manifest_of(self, item):
        for m in self._manifest_items():
            if m.get("id") == item["id"]:
                return m
        return None

    def thumb_path(self, item) -> str:
        safe = item["id"].replace("/", "__")
        return os.path.join(self.dir, safe + ".png")

    def download_thumb(self, item) -> str:
        """下载静态缩略图（PNG，300x300）。动图太大/帧太多时用它兜底。"""
        path = self.thumb_path(item)
        if os.path.exists(path) and os.path.getsize(path) > 200:
            return path
        meta = self._manifest_of(item)
        if not meta or not meta.get("thumb"):
            return ""
        import urllib.parse
        url = self.site + "/media/thumb/" + urllib.parse.quote(meta["thumb"])
        try:
            os.makedirs(self.dir, exist_ok=True)
            with urllib.request.urlopen(url, timeout=25) as r:
                raw = r.read()
            if len(raw) < 200:
                return ""
            with open(path, "wb") as f:
                f.write(raw)
            return path
        except Exception as e:  # noqa: BLE001
            self.log(f"缩略图下载失败 {item['id']}：{e!r}")
            return ""

    # PATH 上的 ffmpeg（Trae/Electron 自带的常是精简版，没有 gif demuxer，要先验一下）；
    # 本机专用路径走 config 的 stickers.ffmpeg_path，不进仓库
    FFMPEG_CANDIDATES = ("ffmpeg",)

    def _ffmpeg(self):
        if getattr(self, "_ff", "unset") != "unset":
            return self._ff
        import shutil
        import subprocess
        cands = []
        if getattr(self, "ffmpeg_path", ""):
            cands.append(self.ffmpeg_path)
        cands.extend(self.FFMPEG_CANDIDATES)
        for c in cands:
            exe = c if os.path.isabs(c) else (shutil.which(c) or "")
            if not exe or not os.path.exists(exe):
                continue
            try:
                out = subprocess.run([exe, "-hide_banner", "-demuxers"],
                                     capture_output=True, text=True, timeout=20)
                if "gif" in (out.stdout or ""):
                    self._ff = exe
                    return exe
            except Exception:  # noqa: BLE001
                continue
        self._ff = ""
        return ""

    def min_path(self, item) -> str:
        safe = item["id"].replace("/", "__")
        return os.path.join(self.dir, safe + ".min.gif")

    def ensure_animated(self, item, force: bool = False) -> str:
        """返回可发送的动图：本来就小就用原图；太大/太长用 ffmpeg 压成小动图（缓存）。"""
        src = self.download(item)
        if not src:
            return ""
        meta = self._manifest_of(item) or {}
        big = (meta.get("bytes") or 0) > 500_000 or (meta.get("frames") or 0) > 30
        if not big and not force:
            return src
        dst = self.min_path(item)
        if os.path.exists(dst) and os.path.getsize(dst) > 1000:
            return dst
        exe = self._ffmpeg()
        if not exe:
            return src        # 没有能用的 ffmpeg：先发动图，发送失败再回落缩略图
        import subprocess
        try:
            os.makedirs(self.dir, exist_ok=True)
            subprocess.run(
                [exe, "-y", "-i", src,
                 "-vf", "fps=12,scale=240:-1:flags=lanczos,split[s0][s1];"
                        "[s0]palettegen=max_colors=128[p];[s1][p]paletteuse",
                 dst], capture_output=True, timeout=60)
            if os.path.exists(dst) and os.path.getsize(dst) > 1000:
                return dst
        except Exception as e:  # noqa: BLE001
            self.log(f"贴纸压缩失败 {item['id']}：{e!r}")
        return src

    def sendable_path(self, item) -> str:
        """发送用路径。

        send_mode=static（默认）：发站内静态缩略图（PNG，几十 KB）。
        原因：当前 wxhook 只有 /SendImgMsg，发出去就是一张图片，GIF 动图在微信端
        不播（2026-10-29 实测），大 GIF 还会白框；缩略图小、清晰、必显。
        send_mode=animated：发压缩过的小动图（等 hook 支持表情/文件消息再切回来）。
        """
        if getattr(self, "send_mode", "static") != "animated":
            return self.download_thumb(item)
        gif = self.ensure_animated(item)
        return gif or self.download_thumb(item)

    def sync(self, limit: int = None) -> dict:
        got = fail = 0
        for it in self.items[:limit] if limit else self.items:
            if self.download(it):
                got += 1
            else:
                fail += 1
        return {"ok": got, "fail": fail, "total": len(self.items)}


_STICKER_RE = re.compile(r"\[sticker[:：]\s*([^\]\n]{1,24})\]")


def extract_stickers(text: str, store: "StickerStore" = None):
    """从回复里抽出 [sticker:xxx] 标记，返回 (去掉标记的正文, [本地图片路径])。"""
    raw = str(text or "")
    paths = []
    for m in _STICKER_RE.finditer(raw):
        if store is None:
            break
        item = store.pick(m.group(1))
        if not item:
            continue
        p = (store.sendable_path(item) if hasattr(store, "sendable_path")
             else store.download(item))
        if p:
            paths.append(p)
    clean = _STICKER_RE.sub("", raw)
    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
    clean = re.sub(r"[，,、;；\s]+$", "", clean).strip()   # 去掉标记留下的悬空逗号
    return clean, paths[:1]
