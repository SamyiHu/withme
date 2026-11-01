#!/usr/bin/env python3
"""应用健康看门狗（项目内，不依赖外部脚本）。

桥/worker/任务队列的健康快照由 `GET /health/detail` 提供；本脚本只做三件事：
拉快照、写日志/状态文件、用退出码告诉计划任务“要不要人看”。

不自动重启任何进程：自动重启容易在根因未除时形成风暴；先把“哪里坏了”
稳定地暴露出来，再由人或上层编排决定。计划任务建议每 5 分钟跑一次：
`deploy/health_watch.bat`。
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(ROOT, "data", "health", "watch.json")
DEFAULT_LOG = os.path.join(ROOT, "logs", "health_watch.log")


def fetch(url: str, token: str, timeout: float = 10.0) -> dict:
    req = urllib.request.Request(url.rstrip("/") + "/health/detail")
    if token:
        req.add_header("Authorization", "Bearer " + token)
        req.add_header("X-API-Token", token)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8") or "{}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("AGENT_SERVER_URL", "http://127.0.0.1:8787"))
    ap.add_argument("--token-env", default="AGENT_API_TOKEN")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--log", default=DEFAULT_LOG)
    ap.add_argument("--timeout", type=float, default=10.0)
    args = ap.parse_args()

    token = os.environ.get(args.token_env, "")
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        data = fetch(args.url, token, args.timeout)
        level = str(data.get("level") or ("ok" if data.get("ok") else "error"))
        checks = data.get("checks") or []
        bad = [c for c in checks if c.get("level") in ("warn", "error")]
        summary = "；".join("%s:%s(%s)" % (c.get("name"), c.get("level"), c.get("detail"))
                           for c in bad) or "全部正常"
        state = {"at": time.time(), "level": level, "summary": summary,
                 "snapshot": data}
        code = 0 if level == "ok" else 2
    except Exception as e:  # noqa: BLE001  服务本身不可达也是“需要人看”
        level, summary = "error", "健康接口不可达：%r" % (e,)
        state = {"at": time.time(), "level": level, "summary": summary}
        code = 2

    for path in (args.out, args.log):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        except OSError:
            pass
    try:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=1)
    except OSError as e:
        print(f"[health-watch] 状态写入失败：{e!r}", file=sys.stderr)
    line = "[%s] %s %s" % (now, level.upper(), summary)
    try:
        with open(args.log, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass
    print(line)
    return code


if __name__ == "__main__":
    sys.exit(main())
