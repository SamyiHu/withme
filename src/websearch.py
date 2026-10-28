"""联网搜索（手部的"搜索"）：智谱 Web Search Pro。
POST /api/paas/v4/tools（OpenAI 风格 tools 端点），返回结构化 search_result。
标准库手工实现，核心零依赖；失败抛 SearchError，由手部层转成温和降级。

上游没有独立的发布时间字段——日期以"（发布时间：YYYY-MM-DD …）"后缀拼在
title 尾部（实测多数结果带）。必须在截断 title 之前把后缀解析出来，否则
80 字符截断会把它拦腰截断；解析出的日期单独放 date 字段透传给模型。
"""
import json
import os
import re
import time
import urllib.error
import urllib.request

# （发布时间：2026-10-23 20:08:27）/（发布时间：2026-10-23）——括号可全半角
_PUBLISH_TIME_RE = re.compile(
    r"[（(]\s*发布时间[：:]\s*(\d{4}-\d{2}-\d{2})(?:[ T]\d{2}:\d{2}(?::\d{2})?)?\s*[）)]\s*$")


class SearchError(Exception):
    pass


def split_publish_date(title: str) -> tuple:
    """title -> (干净标题, 日期YYYY-MM-DD或空)。后缀不在尾部时不剥（防误伤）。"""
    m = _PUBLISH_TIME_RE.search(title or "")
    if not m:
        return (title or "").strip(), ""
    return title[:m.start()].strip(), m.group(1)


def search(query: str, cfg, fetch=None) -> list:
    """query -> [{title, content, link, date}]（最多 cfg.max_results 条）。"""
    key = os.environ.get(cfg.api_key_env, "")
    if not key:
        raise SearchError(f"缺 {cfg.api_key_env}（智谱 key，配在 .env）")
    # 附上今天日期：上游按语义解析"最新"，没有锚点会按快照期返回旧闻
    today = time.strftime("%Y-%m-%d")
    message = f"{(query or '')[:180]}（今天是{today}）"
    body = json.dumps({
        "model": cfg.model,
        "messages": [{"role": "user", "content": message}],
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        cfg.base_url.rstrip("/") + "/tools", data=body, method="POST",
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
    fetch = fetch or _default_fetch
    raw = fetch(req, cfg.timeout_seconds)
    try:
        out = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as e:
        raise SearchError(f"响应不是 JSON：{e}") from None
    items = []
    for choice in out.get("choices") or []:
        for call in ((choice.get("message") or {}).get("tool_calls")) or []:
            for r in call.get("search_result") or []:
                if isinstance(r, dict) and (r.get("content") or r.get("title")):
                    title, date = split_publish_date(str(r.get("title") or ""))
                    items.append({"title": title[:80],
                                  "content": str(r.get("content") or "")[:300],
                                  "link": str(r.get("link") or "")[:120],
                                  "date": date})
    return items[:cfg.max_results]


def _default_fetch(req, timeout):
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", "replace")[:150]
        except Exception:  # noqa: BLE001
            detail = ""
        raise SearchError(f"HTTP {e.code}: {detail}") from None
