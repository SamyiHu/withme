"""梯队1-4：LLM-as-judge 评估——抽样真实对话，按拟人度打分，替换关键词命中基线。

评分维度（各 1-5 分）：
- 人味：像不像熟人在聊天（vs 客服/作文腔）
- 一致性：语气/人设是否前后一致
- 分寸感：主动与追问是否得体、有没有越界感

用法：python tools/judge_eval.py [--limit 20] [--batch 5]
输出：reports/judge_report.md（逐条+均值+结论）
"""
import argparse
import json
import os
import statistics
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.perception import extract_json  # noqa: E402

JUDGE_TASK = """你是严格的对话质量评审。下面是陪伴型 AI"小满"（角色：熟人微信闲聊、
口语化、带语气词、话题会岔开又绕回、对用户好奇、不说金句不端着）与用户的真实对话片段。

逐条给三个维度打分（1-5 整数）并各给一句理由：
- human 人味：1=客服腔/作文腔，3=普通助手，5=像熟人聊天
- consist 一致性：1=语气漂移/人格断裂，5=全程同一个"人"
- tact 分寸感：1=越界/纠缠/机械追问，5=得体有分寸

只输出 JSON 数组，每项：{"human":n,"consist":n,"tact":n,"why":"一句话总评"}
对话片段：
"""


def fetch_pairs(db_path: str, limit: int) -> list:
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT sender, content FROM messages WHERE role_id='xiaoman'"
        " ORDER BY id DESC LIMIT ?", (limit * 2 + 10,)).fetchall()
    pairs, pending_user = [], None
    for r in reversed(rows):
        if r["sender"] == "user":
            pending_user = r["content"]
        elif r["sender"] == "agent" and pending_user and r["content"].strip():
            pairs.append((pending_user[:80], r["content"][:200]))
            pending_user = None
    conn.close()
    return pairs[-limit:]


def judge_batch(pairs, base_url, model, key):
    body = "".join(
        f"{i+1}. 用户：{u}\n   小满：{a}\n" for i, (u, a) in enumerate(pairs))
    payload = {"model": model,
               "messages": [{"role": "user",
                             "content": JUDGE_TASK + body}],
               "temperature": 0.2, "max_tokens": 1400}
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        raw = json.loads(r.read().decode())["choices"][0]["message"]["content"]
    obj = extract_json(raw)
    return obj if isinstance(obj, list) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join(ROOT, "data", "agent.db"))
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--batch", type=int, default=5)
    args = ap.parse_args()

    base_url = os.environ.get("AGENT_LLM_BASE_URL", "")
    model = os.environ.get("AGENT_LLM_MODEL", "")
    key = os.environ.get("LLM_API_KEY", "")
    if not (base_url and model and key):
        print("需要 AGENT_LLM_BASE_URL / AGENT_LLM_MODEL / LLM_API_KEY 环境变量")
        return 1

    pairs = fetch_pairs(args.db, args.limit)
    if not pairs:
        print("没有可评估的对话")
        return 1
    print(f"评估 {len(pairs)} 条对话（批大小 {args.batch}）…")

    scored = []
    for i in range(0, len(pairs), args.batch):
        chunk = pairs[i:i + args.batch]
        try:
            result = judge_batch(chunk, base_url, model, key)
        except Exception as e:  # noqa: BLE001
            print(f"  批 {i//args.batch + 1} 失败：{e!r}")
            continue
        if not result:
            print(f"  批 {i//args.batch + 1} 解析失败")
            continue
        for (u, a), s in zip(chunk, result):
            if all(k in s for k in ("human", "consist", "tact")):
                scored.append({"user": u, "reply": a,
                               "human": int(s["human"]), "consist": int(s["consist"]),
                               "tact": int(s["tact"]), "why": str(s.get("why", ""))[:60]})
        print(f"  批 {i//args.batch + 1}/{(len(pairs)+args.batch-1)//args.batch} 完成")

    if not scored:
        print("全部批次失败")
        return 1

    def mean(k):
        return statistics.mean(x[k] for x in scored)

    lines = ["# LLM-as-judge 评估报告", "",
             f"- 样本：{len(scored)} 条真实对话（评分模型 {model}）",
             f"- 人味 **{mean('human'):.2f}** / 一致性 **{mean('consist'):.2f}** /"
             f" 分寸感 **{mean('tact'):.2f}**（5 分制，验收线 ≥4.0）", "",
             f"**结论：{'通过' if min(mean('human'), mean('consist'), mean('tact')) >= 4.0 else '未通过'}**", "",
             "| # | 用户 | 小满（截断） | 人味 | 一致 | 分寸 | 评语 |",
             "|---|---|---|---|---|---|---|"]
    for i, s in enumerate(scored, 1):
        lines.append(f"| {i} | {s['user'][:20]} | {s['reply'][:40]} |"
                     f" {s['human']} | {s['consist']} | {s['tact']} | {s['why']} |")
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    out = os.path.join(ROOT, "reports", "judge_report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"人味 {mean('human'):.2f} | 一致 {mean('consist'):.2f} |"
          f" 分寸 {mean('tact'):.2f} -> reports/judge_report.md")
    return 0 if min(mean('human'), mean('consist'), mean('tact')) >= 4.0 else 1


if __name__ == "__main__":
    sys.exit(main())
