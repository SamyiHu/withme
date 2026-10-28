"""Cush（本机万能搜）MCP 客户端：spawn `cush mcp` 子进程，说最小 MCP 协议。

为什么不用 CLI：CLI 输出是人看的表格；MCP 工具返回结构化 JSON（标题/链接/
摘要）。为什么有它：Cush 聚合 360/百度/搜狗/必应 + 学术源 + Playwright 真渲染，
是 bigmodel 网络搜索之外的本地全源兜底。stdio = 换行分隔 JSON-RPC。
"""
import json
import os
import queue
import subprocess
import threading


class CushSearchError(Exception):
    pass


def search(query: str, exe: str, limit: int = 4, timeout_s: float = 45.0) -> list:
    """query -> [{title, content, link}]。失败抛 CushSearchError。"""
    if not os.path.isfile(exe):
        raise CushSearchError(f"cush 不在 {exe}")
    proc = subprocess.Popen(
        [exe, "mcp"], cwd=os.path.dirname(os.path.dirname(os.path.dirname(exe))),
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, encoding="utf-8", errors="replace")

    def send(obj):
        proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
        proc.stdin.flush()

    lines = queue.Queue()

    def reader():
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)   # EOF

    threading.Thread(target=reader, daemon=True).start()
    try:
        send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
              "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                         "clientInfo": {"name": "withme", "version": "1.0"}}})
        _wait(lines, 1, timeout_s)
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        send({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
              "params": {"name": "search",
                         "arguments": {"query": (query or "")[:200],
                                        "limit": limit, "timeout_s": timeout_s - 10}}})
        resp = _wait(lines, 2, timeout_s)
    finally:
        try:
            proc.terminate()
        except Exception:  # noqa: BLE001
            pass
    if "error" in resp:
        raise CushSearchError(f"MCP 错误：{str(resp['error'])[:150]}")
    result = resp.get("result") or {}
    text = ""
    for part in result.get("content") or []:
        if isinstance(part, dict) and part.get("type") == "text":
            text += str(part.get("text") or "")
    try:
        out = json.loads(text)
    except ValueError:
        raise CushSearchError(f"工具返回不是 JSON：{text[:120]}") from None
    if not out.get("ok"):
        raise CushSearchError(str(out.get("error") or "搜索失败")[:150])
    items = []
    for r in out.get("results") or []:
        if isinstance(r, dict) and (r.get("title") or r.get("abstract") or r.get("url")):
            # Cush 的 Result 模型只有年份粒度（year），有就带上给模型判新旧
            year = r.get("year")
            items.append({"title": str(r.get("title") or "")[:80],
                          "content": str(r.get("abstract") or r.get("snippet") or "")[:300],
                          "link": str(r.get("url") or r.get("link") or "")[:120],
                          "date": str(year) if year else ""})
    return items[:limit]


def _wait(lines: queue.Queue, msg_id: int, timeout_s: float) -> dict:
    import time
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            line = lines.get(timeout=max(0.1, deadline - time.time()))
        except queue.Empty:
            break
        if line is None:
            raise CushSearchError("cush 进程提前退出")
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue   # 心跳/杂音
        if obj.get("id") == msg_id:
            return obj
    raise CushSearchError(f"等待 MCP 响应(id={msg_id})超时")
