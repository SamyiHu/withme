"""状态驱动语言（梯队1-1：机制闭环）。

以前"精力疲惫"只是 Prompt 里的建议，模型爱听不听。现在：
极端状态 -> 推导可机检的硬约束（LEN/EXCL/LAUGH）-> 渲染后逐条校验
-> 违规打回重写一次 -> 仍违规机械兜底（裁句/换标点）。

约束只在极端状态生效（低精力、情绪低落、亢奋），中常态零约束、零额外成本。
约束、校验、重试、兜底全链路写进消息 meta，可审计。
"""
import re

LAUGH_RE = re.compile(r"哈{2,}|233+|笑死|xswl", re.I)
SENTENCE_END = "。！？!?\n"


DEFAULT_BASELINE = {"energy": 0.6, "valence": 0.35, "arousal": 0.4}


def derive(sv: dict, baseline: dict = None) -> list:
    """从状态向量推导硬约束。返回形如 ["LEN<=60", "EXCL<=0"] 的可机检规则。

    低落判定用相对基线（比角色常态低 0.20 即"蔫了"）——valence 是慢变量，
    绝对阈值 -0.15 在 0.35 基线下几乎不可达，相对值才与状态动力学匹配。
    """
    base = baseline or DEFAULT_BASELINE
    energy, valence, arousal = sv["energy"], sv["valence"], sv["arousal"]
    very_low = energy < 0.25
    low = energy < 0.35
    sad = valence < base["valence"] - 0.20
    hyped = valence > base["valence"] + 0.15 and arousal > base["arousal"] + 0.2
    out = []
    if very_low:
        out.append("LEN<=40")
    elif low:
        out.append("LEN<=60")
    if low or sad:
        out.append("EXCL<=0")
    if hyped:
        out.append("EXCL<=4")
        out.append("LAUGH<=2")
    if sad:
        out.append("LAUGH<=0")
    return out


def to_prompt(constraints: list) -> str:
    parts = []
    for c in constraints:
        kind, n = c.split("<=")
        if kind == "LEN":
            parts.append(f"回复总长不超过{n}字（不含空白），一到两句说完")
        elif kind == "EXCL":
            parts.append("不使用任何感叹号（！和!都不行）" if n == "0"
                         else f"感叹号至多{n}个")
        elif kind == "LAUGH":
            parts.append("不使用哈哈/233/笑死这类笑声词" if n == "0"
                         else f"笑声词至多{n}处")
    return "；".join(parts) + "。这些约束机器会逐条检查，违反会被打回重写。"


def _len(text: str) -> int:
    return len("".join(text.split()))


def check(text: str, constraints: list) -> list:
    """返回违规描述列表；空列表 = 全部通过。"""
    violations = []
    for c in constraints:
        kind, n = c.split("<=")
        n = int(n)
        if kind == "LEN" and _len(text) > n:
            violations.append(f"长度超限({_len(text)}>{n}字)")
        elif kind == "EXCL":
            cnt = text.count("！") + text.count("!")
            if cnt > n:
                violations.append(f"感叹号超限({cnt}>{n})")
        elif kind == "LAUGH":
            cnt = len(LAUGH_RE.findall(text))
            if cnt > n:
                violations.append(f"笑声词超限({cnt}>{n})")
    return violations


def rewrite_instruction(violations: list) -> str:
    return ("上一版回复违反了约束：" + "；".join(violations) +
            "。请重写一版严格遵守全部约束，只输出重写后的回复，不要解释。")


def mechanical_fix(text: str, constraints: list) -> str:
    """重试仍违规时的最后手段：机械修改（裁到句界/换掉多余标点/删多余笑声词）。"""
    for c in constraints:
        kind, n = c.split("<=")
        n = int(n)
        if kind == "LEN" and _len(text) > n:
            compact = text
            # 句界优先（。！？!?），逗号/分号次之：微信口语没句号，
            # 只认句界会把整条砍光（实测'挤成一坨'被切成'挤成一'）
            for bounds in (SENTENCE_END, "，,；;、"):
                best = -1
                for i, ch in enumerate(compact):
                    if ch in bounds and _len(compact[:i + 1]) <= n:
                        best = i + 1
                if best > 0:
                    text = compact[:best].rstrip(bounds)
                    break
            else:
                text = compact[:n].rstrip() + "……"
        elif kind == "EXCL":
            chars = list(text)
            kept = 0
            for i, ch in enumerate(chars):
                if ch in "！!":
                    if kept < n:
                        kept += 1
                    else:
                        chars[i] = "。"
            text = "".join(chars)
        elif kind == "LAUGH":
            hits = list(LAUGH_RE.finditer(text))
            surplus = len(hits) - n
            if surplus > 0:
                for m in hits[:surplus]:
                    text = text.replace(m.group(0), "", 1)
    return text


def toolong_backstop(text: str, limit: int = 18) -> str:
    """审核门"重写后仍超长"的最后兜底：截到 limit 字内（真人中位 9.5 字）。

    边界=换行/句末标点/逗号/空格，取前 limit 字内最后一个（位置>=3，免得切出
    "行哦，"这种残头）；实在没有边界就显式加省略号，绝不无声切断在词中间。

    实测病灶（2026-10-29，真实病例）：旧逻辑只认 pos>=8 的标点、不认换行——
    "行哦，那我用下功夫瞅瞅\n\n先说个粗略的…" 的唯一逗号在 pos2 够不到门槛，
    被一刀切成 18 字硬 slice："…瞅瞅\n\n先说个粗略"，词中间断掉、空行留在
    正中，对方读起来就是"消息被截断了"。"""
    text = text or ""
    if len(text) <= limit:
        return text
    head = text[:limit]
    best = -1
    for i, ch in enumerate(head):
        if i >= 3 and ch in "\n。！？；，,!? ":
            best = i
    return (head[:best] if best >= 0 else head + "…").strip("，。！？；,!? \n\t")
