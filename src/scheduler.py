"""调度器：惰性 tick 与主动推送。

不要求后台常驻——run_tick 每次调用先按 Δt 补算状态（lazy_evolve），
因此外部只需要"想起来时"调一次；常驻模式用 run_loop。
"""
import time


def run_tick(agent, now: float = None) -> dict:
    """评估并可能发出一次主动行为，返回 agent.evaluate_proactive 的结果。"""
    return agent.evaluate_proactive(now)


def run_loop(agent, interval_seconds: float = 60.0):
    """常驻循环（演示用）：每 interval 秒 tick 一次。"""
    while True:
        result = run_tick(agent)
        sent = result.get("sent")
        stamp = time.strftime("%H:%M:%S")
        if sent:
            print(f"[{stamp}] 主动 L{sent['level']} {sent['intent']}：{sent['text']}")
        time.sleep(interval_seconds)
