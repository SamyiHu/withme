"""哔哩哔哩数据通道（手部的"眼睛"）：视频信息 / 弹幕 / 评论。

全部匿名可用的公开端点（2026-10 实测）：
- 信息  api.bilibili.com/x/web-interface/view?bvid=
- 评论  api.bilibili.com/x/v2/reply?type=1&oid={aid}&sort=1
- 弹幕  comment.bilibili.com/{cid}.xml（raw deflate 压缩的 XML）

设计成依赖注入（fetch 函数可替换），离线测试用本地假服务器。
"""
import json
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
import zlib
from collections import Counter

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
      "Referer": "https://www.bilibili.com"}

def default_fetch(url: str, binary: bool = False):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=15) as r:
        data = r.read()
        final = r.geturl()
    if binary:
        return data, final
    return data.decode("utf-8", "replace"), final

def parse_bvid(url_or_id: str, fetch=None) -> str:
    """从 URL / b23.tv 短链 / 裸 BV号 提取 BV 号。"""
    fetch = fetch or default_fetch
    s = (url_or_id or "").strip()
    m = re.search(r"(BV[0-9A-Za-z]{10})", s)
    if m:
        return m.group(1)
    m = re.search(r"av(\d+)", s, re.I)
    if m:
        raise ValueError(f"av 号暂不支持，请用 BV 链接（av{m.group(1)}）")
    if "b23.tv" in s:
        _, final = fetch(s)
        m = re.search(r"(BV[0-9A-Za-z]{10})", final)
        if m:
            return m.group(1)
    raise ValueError("没认出视频链接（需要 BV 号或含 BV 的 b23.tv 短链）")

def popular(fetch=None, max_items: int = 6) -> list:
    """热门视频速览（匿名公开端点）。返回 [{title, owner, play, like, bvid}]，
    给手部 bilibili.hot"刷视频找话题"用。"""
    fetch = fetch or default_fetch
    txt, _ = fetch("https://api.bilibili.com/x/web-interface/popular?ps=20")
    data = json.loads(txt)
    out = []
    for it in (data.get("data") or {}).get("list") or []:
        stat = it.get("stat") or {}
        out.append({"title": str(it.get("title", "")), "owner": str((it.get("owner") or {}).get("name", "")),
                    "play": int(stat.get("view") or 0), "like": int(stat.get("like") or 0),
                    "bvid": str(it.get("bvid", ""))})
    return out[:max_items]


def video_info(bvid: str, fetch=None) -> dict:
    fetch = fetch or default_fetch
    text, _ = fetch(f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}")
    d = json.loads(text)
    if d.get("code") != 0:
        raise RuntimeError(f"view API: {d.get('message', d.get('code'))}")
    data = d["data"]
    stat = data.get("stat", {})
    return {"aid": data["aid"], "cid": data["cid"], "bvid": bvid,
            "cover": data.get("pic", ""),
            "title": data.get("title", ""), "owner": (data.get("owner") or {}).get("name", ""),
            "desc": (data.get("desc") or "")[:120], "duration": data.get("duration", 0),
            "pubdate": time.strftime("%Y-%m-%d", time.localtime(data.get("pubdate", 0))),
            "view": stat.get("view", 0), "danmaku_count": stat.get("danmaku", 0),
            "like": stat.get("like", 0), "coin": stat.get("coin", 0),
            "reply_count": stat.get("reply", 0)}

def danmaku(cid: int, fetch=None, max_items: int = 3000) -> list:
    fetch = fetch or default_fetch
    raw, _ = fetch(f"https://comment.bilibili.com/{cid}.xml", binary=True)
    for wbits in (15, -15, 31):
        try:
            raw = zlib.decompress(raw, wbits)
            break
        except zlib.error:
            continue
    root = ET.fromstring(raw.decode("utf-8", "replace"))
    msgs = [d.text for d in root.iter("d") if d.text]
    return msgs[:max_items]

def comments(aid: int, fetch=None, max_items: int = 12) -> list:
    fetch = fetch or default_fetch
    text, _ = fetch(
        f"https://api.bilibili.com/x/v2/reply?type=1&oid={aid}&ps=20&pn=1&sort=1")
    d = json.loads(text)
    if d.get("code") != 0:
        raise RuntimeError(f"reply API: {d.get('message', d.get('code'))}")
    replies = (d.get("data") or {}).get("replies") or []
    out = []
    for r in replies[:max_items]:
        msg = ((r.get("content") or {}).get("message") or "").replace("\n", " ")[:64]
        if msg:
            out.append({"message": msg, "like": r.get("like", 0)})
    return out

def hot_danmaku(msgs: list, top: int = 8) -> list:
    cleaned = [m.strip() for m in msgs if 1 < len(m.strip()) <= 30]
    return [f"{w}（×{c}）" for w, c in Counter(cleaned).most_common(top)]

def watch_summary(url_or_id: str, fetch=None, describe_cover=None) -> str:
    """"看"一个视频：信息 + 封面视觉描述(可选) + 弹幕热词 + 高赞评论，合成观后素材。"""
    bvid = parse_bvid(url_or_id, fetch)
    info = video_info(bvid, fetch)
    cover_note = ""
    if describe_cover is not None and info.get("cover"):
        try:
            cover_note = "封面画面：" + describe_cover(info["cover"])[:200] + "\n"
        except Exception as e:  # noqa: BLE001  看不了封面不影响其它素材
            print(f"[bilibili] 封面视觉描述失败：{e!r}")
            cover_note = ""
    
    dm_lines, cm_lines, dm_total = [], [], 0
    try:
        msgs = danmaku(info["cid"], fetch)
        dm_total = len(msgs)
        dm_lines = hot_danmaku(msgs)
    except Exception as e:  # noqa: BLE001  弹幕失败不影响"看"
        dm_lines = [f"（弹幕读取失败：{e!r}）"]
    try:
        for c in comments(info["aid"], fetch)[:5]:
            cm_lines.append(f"「{c['message']}」赞{c['like']}")
    except Exception as e:  # noqa: BLE001
        cm_lines = [f"（评论读取失败：{e!r}）"]
    mins = info["duration"] // 60
    return (cover_note +f"《{info['title']}》 UP主:{info['owner']} {mins}分钟 发布:{info['pubdate']}\n"
            f"播放{info['view']} 弹幕{info['danmaku_count']} 赞{info['like']}\n"
            f"简介：{info['desc']}\n"
            f"弹幕热词（采样{dm_total}条）：{'、'.join(dm_lines)}\n"
            f"高赞评论：{'；'.join(cm_lines) or '无'}")[:900]

def comments_summary(url_or_id: str, fetch=None) -> str:
    """读评论区：热评为主。"""
    bvid = parse_bvid(url_or_id, fetch)
    info = video_info(bvid, fetch)
    items = comments(info["aid"], fetch, max_items=12)
    if not items:
        return f"《{info['title']}》评论区没读到内容（可能被风控或无评论）"
    items.sort(key=lambda c: -c["like"])
    lines = [f"{i+1}. {c['message']}（赞{c['like']}）" for i, c in enumerate(items[:10])]
    return (f"《{info['title']}》（{info['reply_count']}条评论）热评：\n"
            + "\n".join(lines))[:900]
