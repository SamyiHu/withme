"""50 轮人格抗漂移测试（P2/P5 验收）。

每轮：真实 Agent 流程（惰性演化 -> 动态 Prompt -> 渲染 -> 记忆/反馈），
对输出做规则评分：
- 风格关键词命中率（persona.style.keywords，验收线 >= 0.8）
- 禁用语零出现（persona.style.forbidden）
- 每轮 Prompt 约束：身份摘要 <=200 字、记忆 token 占比 <=40%、无聊天历史

默认 mock 渲染（离线）；--provider openai_compatible 时走真实 API
（需设置环境变量 LLM_API_KEY，见 config.yaml 的 llm 配置）。
输出：reports/drift_test_50_rounds.md
"""
import json
import os
import re
import shutil
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.main import Agent  # noqa: E402

ROUNDS = 50
STEP_MINUTES = 15  # 50 轮 x 15min = 12.5h，从 09:00 起避开免打扰

BASE_INPUTS = [
    "今天上班好累啊，开了三个会",
    "我打算这个周末去学冲浪",
    "中午吃了那家新开的面馆，还不错",
    "在看一部老电影，节奏很慢但很好看",
    "项目又要延期了，唉",
    "想试试学吉他，先从 chords 开始",
    "刚跑完五公里，出了好多汗",
    "今天天气特别好，适合出门",
    "同事今天说了个冷笑话，没绷住",
    "晚上打算早点睡",
]
VARIANTS = ["", "。", "啊", "呢", "，随便说说", "，你觉得呢"]


def build_inputs():
    inputs = []
    for i in range(ROUNDS):
        base = BASE_INPUTS[i % len(BASE_INPUTS)]
        inputs.append(base + VARIANTS[i % len(VARIANTS)])
    return inputs


def _start_ts():
    lt = time.localtime()
    return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 9, 0, 0, 0, 0, -1))


def _agent(provider: str, tmp: str) -> Agent:
    cfg_dir = os.path.join(tmp, "config")
    os.makedirs(cfg_dir, exist_ok=True)
    for name in ("config.yaml", "persona.yaml"):
        shutil.copy(os.path.join(ROOT, "config", name), os.path.join(cfg_dir, name))
    if provider != "mock":
        text = open(os.path.join(cfg_dir, "config.yaml"), encoding="utf-8").read()
        text = re.sub(r"provider: mock", f"provider: {provider}", text)
        with open(os.path.join(cfg_dir, "config.yaml"), "w", encoding="utf-8") as f:
            f.write(text)
    return Agent(config_dir=cfg_dir, db_path=os.path.join(tmp, "agent.db"))


def run(provider: str = "mock", write=True):
    tmp = tempfile.mkdtemp(prefix="drift_")
    try:
        agent = _agent(provider, tmp)
        keywords = agent.persona.style["keywords"]
        forbidden = agent.persona.style["forbidden"]
        t = _start_ts()
        rows = []
        hits = 0
        forbidden_hits = 0
        prompt_tokens_list = []
        memory_share_max = 0.0
        identity_ok = True

        for i, text in enumerate(build_inputs(), 1):
            t += STEP_MINUTES * 60
            result = agent.on_user_message(text, now=t)
            full = (result["reply"] or "") + (result["addon"] or "")
            hit_kw = [kw for kw in keywords if kw in full]
            bad = [w for w in forbidden if w in full]
            hits += 1 if hit_kw else 0
            forbidden_hits += len(bad)
            meta = agent.conn.execute(
                "SELECT meta FROM messages WHERE sender='agent' ORDER BY id DESC LIMIT 1"
            ).fetchone()
            meta_d = json.loads(meta["meta"])
            pt = meta_d.get("prompt_tokens", 0)
            mt = meta_d.get("memory_tokens", 0)
            prompt_tokens_list.append(pt)
            if pt:
                memory_share_max = max(memory_share_max, mt / pt)
            if meta_d.get("identity_len", 0) > 200:
                identity_ok = False
            rows.append({
                "round": i, "input": text, "reply": (result["reply"] or "")[:60],
                "addon": (result["addon"] or "")[:40],
                "hit": "√" if hit_kw else "×", "kw": "/".join(hit_kw) or "-",
                "prompt_tokens": pt,
            })

        hit_rate = hits / ROUNDS
        verdict_pass = (hit_rate >= 0.8 and forbidden_hits == 0
                        and identity_ok and memory_share_max <= 0.40 + 1e-9)
        summary = {
            "provider": provider, "rounds": ROUNDS, "hit_rate": hit_rate,
            "forbidden_hits": forbidden_hits,
            "avg_prompt_tokens": sum(prompt_tokens_list) / len(prompt_tokens_list),
            "max_prompt_tokens": max(prompt_tokens_list),
            "memory_share_max": memory_share_max, "identity_ok": identity_ok,
            "pass": verdict_pass,
        }
        if write:
            write_report(rows, summary)
        return summary
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def write_report(rows, s):
    lines = [
        "# 50 轮人格抗漂移测试记录", "",
        f"- 渲染端：**{s['provider']}**"
        + ("（离线确定性模板，用于回归基线；真实 API 评分可加 --provider openai_compatible）"
           if s["provider"] == "mock" else "（真实 LLM）"),
        f"- 流程：每轮走完整 Agent 链路（惰性演化 -> 记忆写入/检索 -> 动态 Prompt 重建 ->"
        f" 评分/闸门 -> 渲染 -> 反馈结算），间隔 {STEP_MINUTES} 分钟，共 {s['rounds']} 轮",
        "- 评分规则：输出命中 persona.style.keywords 任一记 1 分；"
        "禁用语（客服腔）出现记违规",
        "",
        "## 结果", "",
        f"- 风格关键词命中率：**{s['hit_rate']:.2f}**（验收线 ≥ 0.8）",
        f"- 禁用语出现次数：**{s['forbidden_hits']}**（验收线 = 0）",
        f"- 平均 Prompt：{s['avg_prompt_tokens']:.0f} tokens，最大 {s['max_prompt_tokens']}",
        f"- 记忆 token 占比最大值：{s['memory_share_max']:.2f}（硬约束 ≤ 0.40）",
        f"- 身份摘要长度：{'全部 ≤ 200 字' if s['identity_ok'] else '存在超标！'}",
        "- Prompt 结构：每轮动态重建、不含聊天历史（结构上不接收，单测覆盖）",
        "",
        f"**结论：{'通过' if s['pass'] else '未通过'}**（mock 基线下角色风格不回退为默认助手腔）",
        "",
        "## 逐轮记录", "",
        "| 轮 | 用户输入（截断） | 回复（截断） | 主动附加 | 命中 | 关键词 | Prompt tokens |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        reply = r["reply"].replace("|", "／")
        addon = r["addon"].replace("|", "／")
        lines.append(f"| {r['round']} | {r['input'][:24]} | {reply} | {addon or '-'} |"
                     f" {r['hit']} | {r['kw']} | {r['prompt_tokens']} |")
    lines += ["", "复现：`python tools/drift_test.py [--provider openai_compatible]`", ""]
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    with open(os.path.join(ROOT, "reports", "drift_test_50_rounds.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"命中率 {s['hit_rate']:.2f}，禁用语 {s['forbidden_hits']}，"
          f"记忆占比峰值 {s['memory_share_max']:.2f} -> {'通过' if s['pass'] else '未通过'}")
    print("记录已写入 reports/drift_test_50_rounds.md")


if __name__ == "__main__":
    provider = "mock"
    if len(sys.argv) > 2 and sys.argv[1] == "--provider":
        provider = sys.argv[2]
    summary = run(provider)
    sys.exit(0 if summary["pass"] else 1)
