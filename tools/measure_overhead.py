# -*- coding: utf-8 -*-
"""WithMe 工程层开销测量（mock 渲染计时，不含 LLM 网络与推理）。

mock 渲染为内存模板查找（~0ms），故 wall time ≈ 纯工程层成本：
状态引擎惰性补算 / 记忆写入检索 / 动态 Prompt 重建 / 评分闸门 / SQLite 落库。
输出：reports/overhead.json
"""
import contextlib
import io
import json
import os
import shutil
import statistics
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.main import Agent  # noqa: E402

N_MSG, N_EVOLVE, N_TICK = 200, 200, 100
BASE_INPUTS = [
    "今天上班好累啊，开了三个会", "我打算这个周末去学冲浪", "中午吃了那家新开的面馆",
    "在看一部老电影，节奏很慢但很好看", "项目又要延期了，唉", "想试试学吉他",
    "刚跑完五公里，出了好多汗", "今天天气特别好", "同事说了个冷笑话", "晚上早点睡",
]


def stats(ms):
    s = sorted(ms)
    return {"n": len(s), "p50": round(statistics.median(s), 2),
            "p95": round(s[int(0.95 * (len(s) - 1))], 2),
            "max": round(max(s), 2), "mean": round(statistics.fmean(s), 2)}


def main():
    tmp = tempfile.mkdtemp(prefix="ovh_")
    try:
        agent = Agent(config_dir=os.path.join(ROOT, "config"),
                      db_path=os.path.join(tmp, "agent.db"))
        t = time.mktime((2026, 10, 1, 9, 0, 0, 0, 0, -1))
        msg_ms, tick_ms, ev_ms = [], [], []
        with contextlib.redirect_stdout(io.StringIO()):
            # 1) 消息全链路（感知启发式 + 记忆 + Prompt 重建 + 渲染 mock + 反馈结算）
            for i in range(N_MSG):
                t += 90
                t0 = time.perf_counter()
                agent.on_user_message(BASE_INPUTS[i % len(BASE_INPUTS)], now=t)
                msg_ms.append((time.perf_counter() - t0) * 1000)
            # 2) 状态引擎惰性补算（每小时一格的补算成本）
            for i in range(N_EVOLVE):
                t += 3600
                row = agent.get_row(t)
                t0 = time.perf_counter()
                row = agent.engine.lazy_evolve(row, t)
                ev_ms.append((time.perf_counter() - t0) * 1000)
                agent.save_row(row, t)
            # 3) 主动调度 tick（候选枚举 -> 评分 -> 闸门）
            for i in range(N_TICK):
                t += 360
                t0 = time.perf_counter()
                agent.evaluate_proactive(now=t)
                tick_ms.append((time.perf_counter() - t0) * 1000)
        out = {"note": "mock 渲染计时，不含 LLM 网络与推理；SQLite 落库包含在内",
               "message_full_chain_ms": stats(msg_ms),
               "state_lazy_evolve_ms": stats(ev_ms),
               "proactive_tick_ms": stats(tick_ms)}
        path = os.path.join(ROOT, "reports", "overhead.json")
        json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(json.dumps(out, ensure_ascii=False, indent=1))
        print("->", path)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
