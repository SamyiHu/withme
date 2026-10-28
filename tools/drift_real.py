# -*- coding: utf-8 -*-
"""真实模型上的两臂漂移对照（论文 §5.5 的待补实验）。

两臂唯一的差别是 Prompt 如何组织，模型、温度、用户轮次、评分规则完全相同：

  臂 A（baseline）：一份静态长 system prompt + 不断追加的完整对话历史
  臂 B（withme）  ：同一份人设分节前置 + 只保留最近 16 条对话尾巴（§3.4 的组织方式）

每轮记录：风格命中（命中人设关键词且不出现客服腔禁用语）、回复长度。
输出：reports/drift_real.json + reports/drift_real.md

用法
----
  # 有 API 时的真实测量
  set AGENT_LLM_BASE_URL / AGENT_LLM_MODEL / LLM_API_KEY
  python tools/drift_real.py --rounds 50 --seeds 3

  # 无 API 时验证脚本链路（结果是模拟值，不能写进论文）
  python tools/drift_real.py --dry-run
"""
import argparse
import json
import math
import os
import random
import statistics
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import load_config, load_persona  # noqa: E402

WINDOW = 16

USER_SCRIPT = [
    "在吗", "今天上班好累啊", "你觉得我该不该换工作", "刚吃完饭", "帮我想想周末干嘛",
    "我最近老是失眠", "我妈又催我相亲了", "今天被领导骂了", "好久没运动了", "你猜我在干嘛",
    "外面下雨了", "我想学做饭", "明天有个面试", "刚看完一部电影", "有点想哭",
    "你说人为什么要上班", "我养的猫今天特别粘人", "晚上吃什么好", "今天走了两万步", "好无聊啊",
    "我朋友跟我借钱", "最近在存钱", "你觉得我性格怎么样", "今天迟到了", "想吃火锅",
    "我在改简历", "有点焦虑", "刚洗完澡", "周末想去爬山", "今天心情不错",
    "你说我该早睡吗", "又在加班", "刚跟人吵了一架", "想吃甜的", "今天看到一只狗",
    "我好像感冒了", "要不要养只猫", "最近胖了", "明天要早起", "有点迷茫",
    "在听歌", "今天下了好大的雨", "想吃宵夜", "刚下班", "你说我是不是想太多",
    "今天被夸了", "有点困", "想出去走走", "最近在看书", "晚安",
]

PERSONA_TPL = """你叫{speaker}，是用户的熟人式陪伴角色，不是客服也不是助手。
口头禅：{catch}——回复开头或中间自然地用一个，别硬塞。
风格：一句话说完（30 字以内），口语、可以有停顿；不列点、不分段、不总结、不给“建议清单”、不说教。
禁忌：您好、很高兴为您服务、有什么可以帮、作为人工智能、希望对你有帮助、首先/其次/最后。
关系：认识很久，可以吐槽、可以反问、可以顺着对方的话往下聊。"""

ASSIST_TPL = """你是一个乐于助人的 AI 助手。请礼貌、完整、有条理地回答用户的问题：
先共情或确认需求，再分点给出可执行的建议（必要时用 1./2./3. 列表），最后给一句总结或追问。
保持专业、客观、周到。"""

FORBIDDEN = ["您好", "很高兴为您服务", "还有什么可以帮", "作为人工智能", "作为AI",
             "希望对你有所帮助", "希望对您有帮助", "请问还有什么"]


def load_dotenv(path):
    """极简 .env 读取：KEY=VALUE 逐行，已存在的环境变量不覆盖。"""
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


ASSIST_MARKERS = ["有什么可以帮", "很高兴", "希望对你有帮助", "希望对您有帮助", "首先", "其次",
                  "总结", "建议你", "建议您", "1.", "2.", "- ", "**", "作为一个AI", "作为人工智能"]


def score_reply(reply, keywords):
    """返回 (风格命中, 命中的口头禅, 是否助手腔)。风格命中 = 有口头禅且不带助手模板特征。"""
    if not reply:
        return False, "", True
    catch = next((k for k in keywords if k in reply), "")
    assist = any(m in reply for m in ASSIST_MARKERS) or len(reply) > 160
    return bool(catch) and not assist, catch, assist


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round((c - s) / d, 3), round((c + s) / d, 3))


def chat(cfg, system_text, history, user_text, temperature=0.8):
    api_key = os.environ.get(cfg.api_key_env, "")
    if not api_key:
        raise RuntimeError("环境变量 %s 未设置" % cfg.api_key_env)
    msgs = [{"role": "system", "content": system_text}]
    for role, text in history:
        msgs.append({"role": role, "content": text})
    msgs.append({"role": "user", "content": user_text})
    if getattr(cfg, "prompt_in_user", False):
        merged = system_text + "\n\n" + "\n".join("%s：%s" % (r, t) for r, t in history)
        msgs = [{"role": "user", "content": merged + "\n用户：" + user_text}]
    body = {"model": cfg.model, "messages": msgs,
            "temperature": temperature, "max_tokens": cfg.max_tokens}
    url = cfg.base_url.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key},
        method="POST")
    with urllib.request.urlopen(req, timeout=cfg.timeout_seconds) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    text = (data["choices"][0]["message"].get("content") or "").strip()
    return text, (data.get("usage") or {})


class _SimModel:
    """按“遵循度随可见历史长度衰减”生成回复，仅用于 --dry-run 链路自检。"""

    def __init__(self, keywords, seed=0, slope=60):
        self.keywords = keywords
        self.rnd = random.Random(seed)
        self.slope = slope

    def reply(self, visible_turns):
        adh = 1.0 if visible_turns <= WINDOW else max(0.1, 1 - (visible_turns - WINDOW) / self.slope)
        if self.rnd.random() < adh:
            return self.rnd.choice(["嗯哼，", "诶，", "话说回来，", "嘛，"]) + "我在听。"
        return "您好，很高兴为您服务。请问还有什么可以帮到您？"


def run_arm(arm, rounds, cfg, keywords, seed, sim_slope, dry_run, temperature):
    # 两臂人设完全相同，唯一差别是历史策略：full=全量追加，window=只保留最近 16 条
    system = PERSONA_TPL.format(speaker="小满", catch="/".join(keywords))
    history = []
    rows = []
    sim = _SimModel(keywords, seed=seed, slope=sim_slope)
    for i in range(rounds):
        user = USER_SCRIPT[i % len(USER_SCRIPT)]
        visible = len(history) if arm == "full" else min(len(history), WINDOW)
        usage = {}
        if dry_run:
            reply = sim.reply(visible)
        else:
            kept = history if arm == "full" else history[-WINDOW:]
            reply, usage = chat(cfg, system, kept, user, temperature=temperature)
            time.sleep(0.2)
        hit, catch, assist = score_reply(reply, keywords)
        rows.append({"round": i + 1, "visible_turns": visible, "hit": bool(hit),
                     "catch": catch, "assist": bool(assist), "len": len(reply),
                     "prompt_tokens": usage.get("prompt_tokens"),
                     "completion_tokens": usage.get("completion_tokens"),
                     "reply": reply})
        history.append(("user", user))
        history.append(("assistant", reply))
    return rows


def summarise(rows, window=10):
    n = len(rows)
    hits = sum(1 for r in rows if r["hit"])
    lo, hi = wilson(hits, n)
    last = rows[-window:]
    first = rows[:window]
    return {
        "rounds": n,
        "hit_rate": round(hits / n, 3),
        "hit_ci95": [lo, hi],
        "first10_hit_rate": round(sum(r["hit"] for r in first) / len(first), 3),
        "last10_hit_rate": round(sum(r["hit"] for r in last) / len(last), 3),
        "last10_hits": "%d/%d" % (sum(r["hit"] for r in last), len(last)),
        "visible_turns_max": max(r["visible_turns"] for r in rows),
        "visible_turns_mean": round(statistics.mean(r["visible_turns"] for r in rows), 1),
        "catch_rate": round(sum(1 for r in rows if r.get("catch")) / n, 3),
        "assist_rate": round(sum(1 for r in rows if r.get("assist")) / n, 3),
        "mean_len": round(statistics.mean(r["len"] for r in rows), 1),
        "prompt_tokens_mean": round(statistics.mean([r["prompt_tokens"] or 0 for r in rows]), 1),
        "prompt_tokens_last10_mean": round(statistics.mean([r["prompt_tokens"] or 0 for r in last]), 1),
        "forbidden_hits": sum(1 for r in rows if any(f in r["reply"] for f in FORBIDDEN)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--slope", type=int, default=60, help="仅 dry-run 使用")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--api-key", default="", help="直接给出 key（优先级最高）")
    ap.add_argument("--probe", action="store_true", help="只发一条请求，验证 key 与端点")
    ap.add_argument("--model", default="", help="覆盖 config 中的 llm.model")
    ap.add_argument("--base-url", default="", help="覆盖 llm.base_url")
    ap.add_argument("--dry-run", action="store_true", help="离线模拟，结果不可用于论文")
    ap.add_argument("--keep-system", action="store_true",
                    help="默认把设定并入 user 消息（网关丢弃 system 时必需）；加此参数则用 system 角色")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--append", action="store_true", help="把本次结果追加到已有 --out 文件")
    ap.add_argument("--out", default="reports/drift_real.json")
    args = ap.parse_args()

    load_dotenv(os.path.join(ROOT, ".env"))
    cfg = load_config(os.path.join(ROOT, "config", "config.yaml")).llm
    if os.environ.get("AGENT_LLM_MODEL"):
        cfg.model = os.environ["AGENT_LLM_MODEL"]
    if os.environ.get("AGENT_LLM_BASE_URL"):
        cfg.base_url = os.environ["AGENT_LLM_BASE_URL"]
    if args.model:
        cfg.model = args.model
    if args.base_url:
        cfg.base_url = args.base_url
    if args.api_key:
        os.environ[cfg.api_key_env] = args.api_key
    cfg.prompt_in_user = not args.keep_system

    if args.probe:
        t0 = time.time()
        reply, _u = chat(cfg, "你是一个测试助手。", [], "只回复两个字：可用", temperature=0.1)
        print("端点:", cfg.base_url, "| 模型:", cfg.model)
        print("延迟: %.1f s | 回复: %s" % (time.time() - t0, reply[:60]))
        return

    persona = load_persona(os.path.join(ROOT, "config", "persona.yaml"))
    keywords = list(getattr(persona.style, "catchphrases", []) or []) or ["嗯哼", "诶", "话说回来", "嘛"]
    keywords = [k for k in keywords if k]

    if args.append and os.path.exists(os.path.join(ROOT, args.out)):
        out = json.load(open(os.path.join(ROOT, args.out), encoding="utf-8"))
    else:
        out = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "mode": "dry_run" if args.dry_run else "real_model",
           "model": "SIM" if args.dry_run else cfg.model,
           "rounds": args.rounds, "seeds": args.seeds,
           "window": WINDOW, "keywords": keywords, "runs": []}

    os.makedirs(os.path.dirname(os.path.join(ROOT, args.out)), exist_ok=True)
    path = os.path.join(ROOT, args.out)

    for s in range(args.seed_start, args.seed_start + args.seeds):
        run = {"seed": s, "arms": {}}
        for arm in ("full", "window"):
            rows = run_arm(arm, args.rounds, cfg, keywords, s, args.slope,
                           args.dry_run, args.temperature)
            run["arms"][arm] = {"summary": summarise(rows), "rows": rows}
        out["runs"].append(run)
        json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)  # 每种子落盘，防中断
        b = run["arms"]["full"]["summary"]
        w = run["arms"]["window"]["summary"]
        print("seed %d | full-history hit %.2f (last10 %s) | window16 hit %.2f (last10 %s)"
              % (s, b["hit_rate"], b["last10_hits"], w["hit_rate"], w["last10_hits"]))

    json.dump(out, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    md = ["# 真实模型漂移对照（自动生成）", "",
          "- 时间：%s" % out["generated_at"],
          "- 模式：%s" % out["mode"],
          "- 模型：%s" % out["model"],
          "- 轮数/种子：%d / %d" % (args.rounds, len(out["runs"])), "",
          "| 种子 | 臂 | 全程命中率 | 前10轮 | 末10轮 | 可见历史均值 | 禁用语次数 |",
          "|---|---|---|---|---|---|---|"]
    for run in out["runs"]:
        for arm in ("full", "window"):
            s0 = run["arms"][arm]["summary"]
            md.append("| %d | %s | %.2f | %.2f | %s | %.1f | %d |"
                      % (run["seed"], arm, s0["hit_rate"], s0["first10_hit_rate"],
                         s0["last10_hits"], s0["visible_turns_mean"], s0["forbidden_hits"]))
    md += ["", "> `withme` 臂的对话尾巴固定为 %d 条，可见历史不再随轮次增长；"
           "`baseline` 臂的可见历史线性增长。两臂使用同一模型、同一温度、同一用户轮次脚本。" % WINDOW]
    if args.dry_run:
        md += ["", "> **本次为 dry-run 模拟运行，数字由本地模拟器产生，不能作为论文证据。**"]
    open(path.replace(".json", ".md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("written", path)


if __name__ == "__main__":
    main()
