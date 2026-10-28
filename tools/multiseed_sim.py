# -*- coding: utf-8 -*-
"""T5 多种子复制：不同用户行为种子下的 14 天主动频率/打扰分布。

复用 param_sweep.make_schedule 的行为模型，仅把写死的种子 42 参数化；
其余与 test_campaign.t5_proactive_sim 的 baseline 臂完全一致（60s 粒度）。
输出：reports/multiseed_sim.json
"""
import json
import os
import random
import shutil
import statistics
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from param_sweep import USER_REJECTS, USER_TEXTS_INTENT, USER_TEXTS_PLAIN  # noqa: E402
from src.main import Agent  # noqa: E402
from src.storage import set_setting  # noqa: E402

DAYS = 14
SEEDS = [42, 1, 7, 13, 99, 2026, 3, 11, 27, 100]


def make_schedule(start: float, seed: int):
    """与 param_sweep.make_schedule 同构，种子可变。"""
    rng = random.Random(seed)
    events = {}
    for d in range(DAYS):
        base = start + d * 86400
        if rng.random() < 0.75:
            for slot in (10 + rng.random() * 1.5, 15 + rng.random() * 2.0):
                t = base + slot * 3600
                if rng.random() < 0.15:
                    events[int(t)] = rng.choice(USER_REJECTS)
                elif rng.random() < 0.4:
                    events[int(t)] = rng.choice(USER_TEXTS_INTENT)
                else:
                    events[int(t)] = rng.choice(USER_TEXTS_PLAIN)
    return events


def run_one(seed: int, start: float) -> dict:
    tmp = tempfile.mkdtemp(prefix="ms_")
    try:
        agent = Agent(config_dir=os.path.join(ROOT, "config"),
                      db_path=os.path.join(tmp, "sim.db"))
        schedule = make_schedule(start, seed)
        daily = [0] * DAYS
        annoyed = 0
        total = 0
        events = sorted(schedule.items())
        ei = 0
        for step in range(DAYS * 24 * 60):
            t = start + (step + 1) * 60
            while ei < len(events) and events[ei][0] <= t:
                msg_t, msg = events[ei]
                ei += 1
                agent.on_user_message(msg, now=msg_t)
                set_setting(agent.conn, "last_typing_ts", msg_t - 300)
            r = agent.evaluate_proactive(now=t)
            sent = r.get("sent")
            if sent and sent["level"] >= 2:
                day = int((t - start) // 86400)
                daily[day] += 1
                total += 1
                p = 0.05 + 0.15 * max(0, daily[day] - 2)
                if p > 0.5:
                    annoyed += 1
        return {"seed": seed, "daily": daily, "total": total,
                "pseudo_annoyed": annoyed,
                "days_active": sum(1 for d in daily if d > 0),
                "max_daily": max(daily)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    lt = time.localtime()
    start = time.mktime((lt[0], lt[1], lt[2] - DAYS, 0, 0, 0, 0, 0, -1))
    runs = []
    for s in SEEDS:
        t0 = time.time()
        r = run_one(s, start)
        runs.append(r)
        print(f"seed={s}: L2 总数 {r['total']}（{r['total']/DAYS:.2f}/日）"
              f" 打扰 {r['pseudo_annoyed']} 日峰值 {r['max_daily']}"
              f" 用时 {time.time()-t0:.0f}s", flush=True)
    totals = [r["total"] for r in runs]
    out = {"seeds": SEEDS, "days": DAYS, "runs": runs,
           "total_mean": statistics.fmean(totals),
           "total_sd": statistics.stdev(totals),
           "total_min": min(totals), "total_max": max(totals),
           "ppd_mean": statistics.fmean(totals) / DAYS,
           "annoyed_sum": sum(r["pseudo_annoyed"] for r in runs),
           "budget_hit_runs": sum(1 for r in runs if r["max_daily"] >= 3)}
    path = os.path.join(ROOT, "reports", "multiseed_sim.json")
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"合计: {out['ppd_mean']:.2f}±{out['total_sd']/DAYS:.2f}/日"
          f" 范围 {out['total_min']}-{out['total_max']} 打扰 {out['annoyed_sum']}")
    print("->", path)


if __name__ == "__main__":
    main()
