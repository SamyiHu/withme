"""对比两次评测报告（同一批 seed 才有可比性）。

    python -m tools.reply_lab.compare reports/reply_lab/A.json reports/reply_lab/B.json
"""
import json
import sys

KEYS = [("len_mean", "平均长度"), ("len_median", "中位长度"), ("len_over_36", "超36字"),
        ("question_rate", "问号率"), ("period_rate", "句号率"), ("emoji_rate", "emoji率")]


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["summary"]


def main(argv=None):
    argv = argv or sys.argv[1:]
    if len(argv) != 2:
        print(__doc__)
        return 2
    a, b = load(argv[0]), load(argv[1])
    print(f"{'指标':<10}{'A':>10}{'B':>10}{'Δ':>10}")
    for key, label in KEYS:
        va, vb = a["bot"][key], b["bot"][key]
        print(f"{label:<10}{va:>10}{vb:>10}{round(vb - va, 2):>10}")
    print(f"{'复读':<10}{a['duplicates']:>10}{b['duplicates']:>10}")
    print(f"{'耗时(s)':<10}{a['latency_mean']:>10}{b['latency_mean']:>10}")
    print(f"人类基线：长度 {a['human']['len_mean']} / 中位 {a['human']['len_median']} / "
          f"问号率 {a['human']['question_rate']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
