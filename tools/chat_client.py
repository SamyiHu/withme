"""终端聊天客户端：连接 WithMe HTTP 服务（serve 启动的那个）。

用法：
  python tools/chat_client.py                          # 连本机默认端口
  python tools/chat_client.py --url http://server:8787 --token xxx

后台线程每 20s 轮询 /pending，服务端主动发来的消息会随时插进来打印。
"""
import argparse
import json
import threading
import time
import urllib.error
import urllib.request


def call(url, token, path, body=None):
    req = urllib.request.Request(
        url.rstrip("/") + path,
        data=json.dumps(body).encode("utf-8") if body is not None else None,
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {token}"} if token else {})},
        method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser(description="WithMe 终端聊天客户端")
    ap.add_argument("--url", default="http://127.0.0.1:8787")
    ap.add_argument("--token", default="", help="服务端设置了 AGENT_API_TOKEN 时填写")
    args = ap.parse_args()
    state = {"after_id": 0}

    def poll_loop():
        while True:
            time.sleep(20)
            try:
                d = call(args.url, args.token,
                         f"/pending?after_id={state['after_id']}")
                for m in d.get("items", []):
                    tag = m["meta"].get("proactive", {})
                    prefix = f"〔主动·{tag.get('intent', '?')}〕" if tag else ""
                    print(f"\n  小满 {prefix}{m['content']}")
                    state["after_id"] = m["id"]
            except (urllib.error.URLError, OSError):
                pass
            except Exception as e:  # noqa: BLE001
                print(f"\n[轮询失败 {e!r}]")

    health = call(args.url, args.token, "/health")
    hist = call(args.url, args.token, "/messages?limit=20")
    for m in hist.get("items", []):
        who = "你" if m["sender"] == "user" else "小满"
        print(f"  {who}：{m['content'][:80]}")
        if m["sender"] == "agent":
            state["after_id"] = max(state["after_id"], m["id"])
    print(f"— 已连接 {args.url}（role={health.get('role_id')}，"
          f"渲染端={health.get('llm_provider')}）；/quit 退出 —")
    threading.Thread(target=poll_loop, daemon=True).start()
    while True:
        try:
            text = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text or text == "/quit":
            break
        result = call(args.url, args.token, "/chat", {"text": text})
        if result.get("addon"):
            print(f"  小满〔主动·{result['addon_meta']['intent']}〕{result['addon']}")
        print(f"  小满：{result['reply']}")
        state["after_id"] = max(state["after_id"], result["meta"].get("message_id", 0))


if __name__ == "__main__":
    main()
