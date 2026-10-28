"""端到端演示：应答 -> 8 小时后主动提醒 -> 拒绝反馈 -> 冷却加倍拦截。

时间线取"今天 12:00（本地）"为基准，避免深夜运行时落入免打扰时段
导致主动环节被拦截（免打扰拦截本身由测试覆盖）。
"""
import os
import sqlite3
import time

from .main import Agent
from .storage import project_root


def _noon_today() -> float:
    lt = time.localtime()
    return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 12, 0, 0, 0, 0, -1))


def run_demo(db_path: str = None, fresh: bool = False):
    db_path = db_path or os.path.join(project_root(), "data", "demo.db")
    if fresh and os.path.exists(db_path):
        for suffix in ("", "-wal", "-shm"):
            try:
                os.remove(db_path + suffix)
            except OSError:
                pass
    agent = Agent(db_path=db_path)

    noon = _noon_today()
    t_user1 = noon - 8 * 3600

    print("=" * 60)
    print("① 8 小时前：用户说了一句话（含未完成意图）")
    r1 = agent.on_user_message("最近在忙项目，不过还想学冲浪来着", now=t_user1)
    print(f"  你：最近在忙项目，不过还想学冲浪来着")
    print(f"  小满：{r1['reply']}")

    print("=" * 60)
    print("② 现在（12:00）：调度 tick——孤独感 + 未完成意图驱动主动")
    r2 = agent.evaluate_proactive(now=noon)
    if r2.get("sent"):
        s = r2["sent"]
        print(f"  〔主动 L{s['level']}·{s['intent']}〕{s['text']}")
        print(f"  主动评分 active_score={r2['active_score']}，闸门通过，log_id={s['log_id']}")
    else:
        print(f"  未主动：{r2.get('reason')} {r2.get('gate', '')}")

    print("=" * 60)
    print("③ 1 分钟后：用户拒绝这次主动")
    r3 = agent.on_user_message("别提这个了，烦", now=noon + 60)
    print(f"  你：别提这个了，烦")
    print(f"  小满：{r3['reply']}")
    if r3["meta"].get("feedback"):
        print(f"  反馈结算：{r3['meta']['feedback']}")

    print("=" * 60)
    print("④ 2 分钟后：再次 tick——同类型冷却已加倍，应被闸门拦截")
    r4 = agent.evaluate_proactive(now=noon + 120)
    if r4.get("sent"):
        print(f"  意外发出：{r4['sent']}")
    else:
        gate = r4.get("gate", {}).get("checks", {})
        blocked = [k for k, v in gate.items()
                   if isinstance(v, dict) and v.get("pass") is False]
        print(f"  已拦截（触发检查项：{blocked or r4.get('reason')}）")
        if "cooldown_level" in gate:
            print(f"  冷却判定：{gate['cooldown_level']}")

    print("=" * 60)
    print("⑤ active_logs 审计（评分与闸门明细都已留痕）")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    for row in conn.execute("SELECT id, level, intent_type, source_type, source_id,"
                            " user_response_latency, user_sentiment_delta"
                            " FROM active_logs ORDER BY id"):
        print(f"  #{row['id']} L{row['level']} {row['intent_type']}"
              f" 来源={row['source_type']}:{row['source_id']}"
              f" 回应延迟={row['user_response_latency']}"
              f" 情绪={row['user_sentiment_delta']}")
    conn.close()
    print("=" * 60)
    print("演示库：", db_path)
