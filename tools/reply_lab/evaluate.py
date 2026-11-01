"""离线评测：拿公开语料里的真实对话，给模型出题、给人类原话当参照。

用法：
    python -m tools.reply_lab.evaluate --corpus data/reply_lab/lccc_toy.json --n 12

输出：reports/reply_lab/<时间>.md + .json（长度/标点/表情/多样性/复读/耗时，
以及和人类原话的二元组重合度）。不接触任何真实用户。
"""
import argparse
import json
import os
import re
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tools.reply_lab import corpus as corpus_mod
from tools.reply_lab.sandbox import Sandbox

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B50\u2764]")


def bigrams(s: str) -> set:
    s = re.sub(r"\s+", "", s or "")
    return {s[i:i + 2] for i in range(len(s) - 1)}


def overlap(a: str, b: str) -> float:
    ba, bb = bigrams(a), bigrams(b)
    if not ba or not bb:
        return 0.0
    return len(ba & bb) / len(ba | bb)


def summarize(rows: list) -> dict:
    bot = [r["bot"] for r in rows]
    hum = [r["human"] for r in rows]
    def stats(xs):
        lens = [len(x) for x in xs if x]
        return {
            "n": len(xs),
            "len_mean": round(statistics.mean(lens), 1) if lens else 0,
            "len_median": statistics.median(lens) if lens else 0,
            "len_over_36": sum(1 for L in lens if L > 36),
            "question_rate": round(sum(1 for x in xs if "？" in x or "?" in x) / max(len(xs), 1), 2),
            "period_rate": round(sum(1 for x in xs if x.rstrip().endswith("。")) / max(len(xs), 1), 2),
            "emoji_rate": round(sum(1 for x in xs if EMOJI.search(x)) / max(len(xs), 1), 2),
            "empty": sum(1 for x in xs if not x),
        }
    uniq = len({x for x in bot if x})
    dup = len([x for x in bot if x]) - uniq
    return {
        "cases": len(rows),
        "bot": stats(bot), "human": stats(hum),
        "distinct_ratio": round(uniq / max(len([x for x in bot if x]), 1), 2),
        "duplicates": dup,
        "overlap_mean": round(statistics.mean(r["overlap"] for r in rows), 3) if rows else 0,
        "latency_mean": round(statistics.mean(r["latency"] for r in rows), 1) if rows else 0,
    }


def to_markdown(s: dict, rows: list, corpus: str, note: str = "") -> str:
    b, h = s["bot"], s["human"]
    lines = [
        "# 回复实验室 · 评测报告", "",
        f"- 语料：`{corpus}`（公开聊天数据，离线使用）",
        f"- 用例：{s['cases']} 条；平均耗时 {s['latency_mean']}s/条",
        f"- 说明：{note or '机器人只在这个沙盒里回答，不对任何真实用户说话'}", "",
        "| 指标 | 模型 | 人类原话 |", "|---|---|---|",
        f"| 平均长度（字） | {b['len_mean']} | {h['len_mean']} |",
        f"| 长度中位数 | {b['len_median']} | {h['len_median']} |",
        f"| 超 36 字条数 | {b['len_over_36']} | {h['len_over_36']} |",
        f"| 带问号比例 | {b['question_rate']:.0%} | {h['question_rate']:.0%} |",
        f"| 句末句号比例 | {b['period_rate']:.0%} | {h['period_rate']:.0%} |",
        f"| 带 emoji 比例 | {b['emoji_rate']:.0%} | {h['emoji_rate']:.0%} |",
        f"| 复读条数 | {s['duplicates']} | — |",
        f"| 与人类原话的二元组重合 | {s['overlap_mean']} | — |", "",
        "## 逐条", "",
    ]
    for i, r in enumerate(rows, 1):
        lines += [f"**{i}. 对方**：{r['context'][-1]}",
                  f"- 人类：{r['human']}",
                  f"- 模型：{r['bot'] or '（空）'}", ""]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=os.path.join(ROOT, "data", "reply_lab", "lccc_toy.json"))
    ap.add_argument("--n", type=int, default=12, help="用例条数（上限 50）")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "reply_lab"))
    args = ap.parse_args(argv)

    n = max(1, min(args.n, 50))
    cases = corpus_mod.load(args.corpus, limit=None)
    if not cases:
        print("没有可用用例")
        return 1
    import random
    random.Random(args.seed).shuffle(cases)
    cases = cases[:n]

    rows = []
    for i, c in enumerate(cases, 1):
        # 每条用例一个干净沙盒：避免上一条对话污染下一条
        sb = Sandbox(os.path.join(ROOT, "config"))
        t0 = time.time()
        try:
            bot = sb.reply(c["context"][-1], now=time.time())
        except Exception as e:  # noqa: BLE001  单条失败不影响整体
            print(f"[{i}/{len(cases)}] 失败：{e!r}")
            bot = ""
        finally:
            sb.close()
        rows.append({"context": c["context"], "human": c["reply"], "bot": bot,
                     "overlap": overlap(bot, c["reply"]),
                     "latency": round(time.time() - t0, 1)})
        print(f"[{i}/{len(cases)}] {round(rows[-1]['latency'],1)}s 对方：{c['context'][-1][:24]}")
        print(f"        人类：{c['reply'][:36]}")
        print(f"        模型：{bot[:36]}")

    s = summarize(rows)
    os.makedirs(args.out, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    md = to_markdown(s, rows, args.corpus)
    with open(os.path.join(args.out, f"{stamp}.md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(args.out, f"{stamp}.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": s, "rows": rows}, f, ensure_ascii=False, indent=1)
    print(json.dumps(s, ensure_ascii=False, indent=1))
    print("报告：", os.path.join(args.out, f"{stamp}.md"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
