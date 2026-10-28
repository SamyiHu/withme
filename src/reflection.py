"""反思层（成长核心）：定期把近期经历沉淀为洞察记忆，并在有界预算内微调人格。

成长不是放飞：吸引子中心相对角色卡的总漂移有上限（超出即被裁剪），
聊天习惯在角色卡之外最多新增 N 条，每一步调整都写 reflections 表留痕，
可一键 reset 回角色卡。收敛性不受影响——回拉目标只是缓慢移动，仍有界。
"""
from .perception import extract_json
from .state_dims import DIMS

REFLECTION_TASK = """你是陪伴型 AI 的自我反思模块。基于近期真实经历（不是想象），
回答两个问题：这段时间我了解到什么？我的性格参数该朝哪个方向微调？

只输出一个 JSON 对象，禁止任何其他文字：
{
 "insights": ["洞察，1-3条；关于用户的近况/关系动态/我的行为效果，要具体、可被日后引用", ...],
 "new_rule": "如果有充分证据，提出一条新的聊天习惯规则（一句话）；证据不足就给空字符串",
 "trait_shift": {"维度名": -0.05~0.05 的调整量, ...}
}

trait_shift 可用维度与含义（调整要克制，方向要有经历依据）：
- energy 精力水平 / social_need 社交需求 / curiosity 好奇心 / care 关心强度
- expression 表达欲 / arousal 唤醒度 / valence 情绪基线
依据示例：主动多次被拒 -> care/expression 下调；分享记忆屡获正面回应 -> expression 上调。

"gate_shift"：{"L2_threshold": -0.05~+0.05}。主动被拒率明显偏高 -> 上调（更保守），
主动屡获正面回应 -> 下调（更愿意搭话）；没有明确依据就不给这个字段。"""


def build_task(evidence: dict) -> str:
    parts = [REFLECTION_TASK, "【近期经历证据】"]
    if evidence.get("themes"):
        parts.append("最近对话主题（截断）：\n- " + "\n- ".join(evidence["themes"][:6]))
    stats = evidence.get("active_stats") or {}
    parts.append(
        f"主动行为：共 {stats.get('total', 0)} 次；被拒 {stats.get('rejected', 0)} 次；"
        f"平均回应情绪 {stats.get('avg_sentiment', 0):+.2f}（-1~1）")
    rel = evidence.get("relation") or {}
    parts.append(f"关系现状：亲近 {rel.get('affinity', 0):.2f} / 信任 {rel.get('trust', 0):.2f} /"
                 f" 熟识 {rel.get('familiarity', 0):.2f}")
    if evidence.get("open_intents"):
        parts.append("用户还没做完的事：" + "；".join(evidence["open_intents"][:3]))
    if evidence.get("memories"):
        parts.append("期间记下的事实（截断）：\n- " + "\n- ".join(evidence["memories"][:6]))
    drift = evidence.get("drift") or {}
    parts.append(f"当前人格漂移（相对角色卡，有界）：{drift}")
    parts.append("【任务】输出上述 JSON。洞察必须能在证据里找到出处，禁止编造。")
    return "\n\n".join(parts)


def parse_output(raw: str, cfg) -> dict:
    obj = extract_json(raw)
    if obj is None:
        return None
    insights = [str(x).strip()[:90] for x in (obj.get("insights") or [])
                if str(x).strip()][:cfg.max_insights]
    new_rule = str(obj.get("new_rule") or "").strip()[:100]
    shifts = {}
    raw_shift = obj.get("trait_shift")
    if isinstance(raw_shift, dict):
        for dim, val in raw_shift.items():
            if dim in DIMS:
                try:
                    step = max(-cfg.drift_step, min(cfg.drift_step, float(val)))
                    if abs(step) > 1e-6:
                        shifts[dim] = step
                except (TypeError, ValueError):
                    continue
    gate_shift = None
    raw_gate = obj.get("gate_shift")
    if isinstance(raw_gate, dict) and "L2_threshold" in raw_gate:
        try:
            gate_shift = max(-cfg.drift_step, min(cfg.drift_step,
                                                  float(raw_gate["L2_threshold"])))
        except (TypeError, ValueError):
            gate_shift = None
    return {"insights": insights, "new_rule": new_rule, "trait_shift": shifts,
            "gate_shift": gate_shift}
