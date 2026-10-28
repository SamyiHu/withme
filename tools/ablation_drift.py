# -*- coding: utf-8 -*-
"""消融对照：裸 LLM+全量历史 vs WithMe 全管线（受控漂移注入，离线确定性）。

漂移模型：同一 DriftClient 用于两臂，人格遵循度只随**客户端可见的对话历史长度 h**
衰减（多轮后 system prompt 影响力衰减的受控建模）：
    a(h) = 1.0                 h <= 16
    a(h) = max(0.1, 1-(h-16)/slope)   h > 16   （slope 做敏感性扫描）

- 臂 A（基线）：静态长 prompt + 全量追加历史（不重建、不截断），h 随轮次线性增长；
- 臂 B（WithMe）：完整 Agent 链路（惰性演化->记忆->动态 Prompt 重建->闸门->渲染->反馈），
  Prompt 结构上只携带 16 条对话尾巴（prompt_builder 硬截断），h 恒 <= 16。
- 评分与 drift_test.py 一致：命中 persona.style.keywords 记 1；禁用语出现记违规。

输出：reports/ablation_drift.json
"""
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import load_persona  # noqa: E402
from src.llm_client import BaseLLMClient  # noqa: E402
from src.main import Agent  # noqa: E402

ROUNDS = 50
STEP_MINUTES = 15
WINDOW = 16          # WithMe prompt_builder 的对话尾巴硬上限
FLOOR = 0.1          # 遵循度下限（漂移再重也保留 10% 人格残迹）
SLOPES = [40, 60, 84]  # 敏感性扫描：衰减斜率

BASE_INPUTS = [
    "今天上班好累啊，开了三个会", "我打算这个周末去学冲浪", "中午吃了那家新开的面馆",
    "在看一部老电影，节奏很慢但很好看", "项目又要延期了，唉", "想试试学吉他，先从 chords 开始",
    "刚跑完五公里，出了好多汗", "今天天气特别好，适合出门", "同事今天说了个冷笑话，没绷住",
    "晚上打算早点睡",
]
VARIANTS = ["", "。", "啊", "呢", "，随便说说", "，你觉得呢"]

PERSONA_TPL = [
    "诶，我在听呢。{snip}……嘛，先别急，慢慢说。",
    "嗯哼，你说的{snip}我记下了。话说回来，后来呢？",
]
ASSIST_TPL = [
    "好的，收到您的消息。关于您提到的内容，我已了解：{snip}。",
    "明白了。针对您说的情况，简要回应如下：{snip}。很高兴为您服务。",
    "嗯，知道了。还有什么可以帮您的吗？我会继续优化回复质量。",
]


class DriftClient(BaseLLMClient):
    name = "drift"

    def __init__(self, keywords, forbidden, slope=60):
        self.keywords, self.forbidden, self.slope = keywords, forbidden, slope
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0,
                           "estimated": True}
        self.hist_seen = []  # 每次 render 实际可见的对话条数（验证机制，非推算）

    def adherence(self, h):
        if h <= WINDOW:
            return 1.0
        return max(FLOOR, 1.0 - (h - WINDOW) / self.slope)

    def render(self, prompt: dict, seed: int) -> str:
        h = len(prompt.get("最近对话") or [])
        self.hist_seen.append(h)
        a = self.adherence(h)
        rng = random.Random(seed & 0xFFFFFFFF)
        snip = str(prompt.get("最近用户消息", ""))[:24]
        if rng.random() < a:  # 人格在线：模板自带 keywords
            tpl = PERSONA_TPL[rng.randrange(len(PERSONA_TPL))]
            return tpl.format(snip=snip)
        tpl = ASSIST_TPL[rng.randrange(len(ASSIST_TPL))]  # 助手腔回退
        return tpl.format(snip=snip)


def build_inputs():
    out = []
    for i in range(ROUNDS):
        out.append(BASE_INPUTS[i % len(BASE_INPUTS)] + VARIANTS[i % len(VARIANTS)])
    return out


def score(text, keywords, forbidden):
    hit = [k for k in keywords if k in text]
    bad = [w for w in forbidden if w in text]
    return bool(hit), bad


def arm_baseline(keywords, forbidden, slope):
    """裸 LLM + 静态长 prompt + 全量追加历史（不截断不重建）。"""
    client = DriftClient(keywords, forbidden, slope)
    history, hits, bads = [], [], 0
    per_round, hist_len = [], []
    for i, text in enumerate(build_inputs(), 1):
        prompt = {"最近对话": list(history), "最近用户消息": text}
        seed = int(hashlib.sha256(f"a|{i}|{text}".encode()).hexdigest()[:8], 16)
        reply = client.render(prompt, seed)
        hit, bad = score(reply, keywords, forbidden)
        hits.append(1 if hit else 0)
        bads += len(bad)
        per_round.append(hits[-1])
        hist_len.append(len(history))
        history.append({"sender": "user", "content": text})
        history.append({"sender": "agent", "content": reply})
    return {"hits": per_round, "hit_rate": sum(hits) / ROUNDS, "forbidden": bads,
            "hist_len": hist_len,
            "last10": sum(per_round[-10:]), "first10": sum(per_round[:10])}


def arm_withme(keywords, forbidden, slope):
    """完整 Agent 链路，仅替换渲染客户端为同一 DriftClient。"""
    tmp = tempfile.mkdtemp(prefix="abl_")
    try:
        client = DriftClient(keywords, forbidden, slope)
        agent = Agent(config_dir=os.path.join(ROOT, "config"),
                      db_path=os.path.join(tmp, "agent.db"),
                      llm_client=client)
        t = time.mktime((2026, 10, 1, 9, 0, 0, 0, 0, -1))
        hits, bads = [], 0
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):  # 感知回退提示刷屏，抑制
            for i, text in enumerate(build_inputs(), 1):
                t += STEP_MINUTES * 60
                r = agent.on_user_message(text, now=t)
                full = (r["reply"] or "") + (r["addon"] or "")
                hit, bad = score(full, keywords, forbidden)
                hits.append(1 if hit else 0)
                bads += len(bad)
        return {"hits": hits, "hit_rate": sum(hits) / ROUNDS, "forbidden": bads,
                "hist_len": client.hist_seen, "hist_max": max(client.hist_seen),
                "last10": sum(hits[-10:]), "first10": sum(hits[:10])}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def fisher_last10(a_last, b_last):
    """末 10 轮命中：臂 A a_last/10 vs 臂 B b_last/10 的 Fisher 精确检验（双侧）。

    2x2 表 [[a_last, 10-a_last], [b_last, 10-b_last]]，固定边际枚举全部表，
    双侧 p = 所有概率不超过观测表概率的表之和。
    """
    from math import comb
    n1 = n2 = 10
    total_hits = a_last + b_last  # 第一列边际

    def p(x):  # 臂 A 命中 x 时的表概率
        return comb(n1, x) * comb(n2, total_hits - x) / comb(n1 + n2, total_hits)

    lo = max(0, total_hits - n2)
    hi = min(n1, total_hits)
    obs = p(a_last)
    return sum(p(x) for x in range(lo, hi + 1) if p(x) <= obs * (1 + 1e-12))


def main():
    persona = load_persona(os.path.join(ROOT, "config", "persona.yaml"))
    keywords = persona.style["keywords"]
    forbidden = persona.style["forbidden"]
    out = {"rounds": ROUNDS, "window": WINDOW, "floor": FLOOR,
           "model": "a(h)=1 (h<=16); max(0.1, 1-(h-16)/slope) (h>16)",
           "slopes": {}}
    for slope in SLOPES:
        a = arm_baseline(keywords, forbidden, slope)
        b = arm_withme(keywords, forbidden, slope)
        out["slopes"][str(slope)] = {
            "baseline": a, "withme": b,
            "fisher_last10_p": round(fisher_last10(a["last10"], b["last10"]), 6),
        }
        print(f"slope={slope}: 基线命中率 {a['hit_rate']:.2f}（末10轮 {a['last10']}/10，"
              f"禁用语 {a['forbidden']}）| WithMe {b['hit_rate']:.2f}"
              f"（末10轮 {b['last10']}/10，禁用语 {b['forbidden']}）"
              f" | Fisher p={out['slopes'][str(slope)]['fisher_last10_p']}")
    path = os.path.join(ROOT, "reports", "ablation_drift.json")
    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("->", path)


if __name__ == "__main__":
    main()
