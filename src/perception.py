"""洞察层：模型对用户消息的轻量推理（计划原则 0 允许的"必要的轻量推理"）。

每次用户消息先做一次廉价的结构化分析（意图/情绪/话题/取消与完成），
结果喂给外部指标——关系更新、意图闭环、记忆显著度、状态脉冲、主动评分上下文。

边界：模型只提供"感知"，不决定"是否主动/主动什么"——决策权仍在闸门（原则 B/D）。
分析失败或 mock 模式时回退到关键词启发式，链路永不中断。
"""
import json
import re

ANALYSIS_HEADER = """分析用户发给陪伴型 AI 的最新一条消息，只输出一个 JSON 对象，禁止输出任何其他文字。

会给你一些上下文（最近对话、对方提过还没做完的事），用来消解指代：
"那件事""上次说的""先放放吧"这类说法，必须结合上下文还原成具体话题。

字段定义：
- sentiment: 情绪倾向，-1.0 到 1.0（负=烦躁/低落/拒绝，正=开心/感激/兴奋）
- emotion: 一个词，如 平静/疲惫/兴奋/烦躁/低落/感激/好奇
- intent_type: 从这些里选一个：闲聊|倾诉|分享|提问|求助建议|情绪安抚|提出计划|推迟计划|取消计划|完成计划|拒绝
- topics: 消息涉及的话题关键词列表（1-3 个，用具体词，禁止用"那件事"这类代词）
- urgency: 0.0 到 1.0，用户是否需要及时/认真回应
- cancel_topics: 若用户在彻底放弃某个计划，写出该计划的具体关键词；否则 []
- defer_topics: 若用户只是推迟/改期某个计划（以后还会做），写出该计划的具体关键词；否则 []
- done_topics: 若用户完成了某件之前提到的事，写出具体关键词；否则 []
- hand_action: 若这句话隐含可执行的动手请求，从【可用动作】里选并给出参数，
  形如 {"name":"reminder.set","params":{...}}；一句话含多个步骤时给数组（最多3个，按顺序）；
  没有就 null。
  唱歌/翻唱分两个动作：
  - **song.sing**（自己编歌唱）：对方说"唱首歌/唱一段"没指定具体歌曲时用。
    lyrics：对方把歌词发给你时原样填入（≤12 行）；没给才自己写 2-4 句短句。
    style 必须来自用户；对方没说曲风就留空让回复先问。
  - **song.cover**（翻唱现成歌曲）：对方说「翻唱xxx/唱《歌名》/用你的声音唱xxx」
    指的是现成的歌时用。title 填歌名，artist 填歌手（可选），
    duration 可选（翻一段），start 可选（跳前奏）。
    对方给了歌词但指的是现成歌的词 → 也是 song.cover。
  voice / duration 对两个动作都适用。
  不要用文字念歌词代替唱歌。
  【上一轮动作】是系统刚执行过的动作及其结果：仅当对方明确要求重做该动作
  （"再画一张/重试/再发一次/还是不行"）时才重新发起（参数按对方新说法调整，
  如画图被拦就改用描述性说法重画）。对方在纠正别的事（称呼/辈分/说法对错）
  或只是闲聊时，hand_action 给 null——不要因为上一轮动作的存在就重发
- reply_plan: 像人一样先想清楚再开口——结合对方情绪和上下文给出
  {"what": "接哪个点/回什么内容（一句话）", "how": "怎么说：语气+长短+要不要延展",
   "mood": "你此刻该有的情绪反应，从 惊讶/好奇/无语/心疼/兴奋/乐/认真/平静 里挑最贴的"}
  情绪别只会乐：内容离谱就惊讶或无语，对方难受就认真或心疼，新鲜事就好奇"""


def build_task(text: str, context: dict = None) -> str:
    parts = [ANALYSIS_HEADER]
    context = context or {}
    if context.get("dialogue"):
        parts.append("【最近对话】（从早到晚）\n" + "\n".join(context["dialogue"][-4:]))
    if context.get("open_intents"):
        lines = [f"- {c}" for c in context["open_intents"][:3]]
        parts.append("【对方提过、还没做完的事】\n" + "\n".join(lines))
    if context.get("hands"):
        lines = [f"- {m['name']}({', '.join(m['params'])}): {m['desc']}"
                 for m in context["hands"]]
        parts.append("【可用动作】（hand_action 只能从这里选）\n" + "\n".join(lines))
    if context.get("recent_hand"):
        parts.append("【上一轮动作】" + str(context["recent_hand"])[:200])
    parts.append(f"【用户最新消息】{text[:200]}")
    return "\n\n".join(parts)


def extract_json(raw: str):
    """从模型输出里抠出 JSON 对象或数组；``` 代码块、前后闲话都能容错。"""
    if not raw:
        return None
    raw = raw.replace("```json", "```")
    if "```" in raw:
        for part in (p.strip() for p in raw.split("```")):
            if part.startswith("{") or part.startswith("["):
                raw = part
                break
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = raw.find(opener), raw.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                continue
    return None


def _num(value, lo, hi, default):
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return default


def _clamp_output(obj: dict, text: str, valid_hands=()) -> dict:
    intent_type = str(obj.get("intent_type", ""))
    valid = ("闲聊", "倾诉", "分享", "提问", "求助建议", "情绪安抚",
             "提出计划", "推迟计划", "取消计划", "完成计划", "拒绝")
    if intent_type not in valid:
        intent_type = "闲聊"
    topics = obj.get("topics")
    if not isinstance(topics, list):
        topics = []
    cancel = obj.get("cancel_topics")
    done = obj.get("done_topics")
    defer = obj.get("defer_topics")
    hand = obj.get("hand_action")
    hand_out = None
    items = hand if isinstance(hand, list) else ([hand] if isinstance(hand, dict) else [])
    actions = []
    for act in items[:3]:
        if isinstance(act, dict) and act.get("name") in valid_hands:
            params = act.get("params") if isinstance(act.get("params"), dict) else {}
            actions.append({"name": str(act["name"])[:24],
                            "params": {str(k)[:16]: (v if isinstance(v, (str, int, float))
                                                     else str(v)[:60])
                                       for k, v in list(params.items())[:8]}})
    if actions:
        hand_out = actions if isinstance(hand, list) else actions[0]
    plan = obj.get("reply_plan")
    plan_out = None
    if isinstance(plan, dict) and (plan.get("what") or plan.get("how") or plan.get("mood")):
        plan_out = {"what": str(plan.get("what", ""))[:80],
                    "how": str(plan.get("how", ""))[:80],
                    "mood": str(plan.get("mood", ""))[:12]}
    return {
        "sentiment": _num(obj.get("sentiment"), -1.0, 1.0, 0.0),
        "emotion": str(obj.get("emotion", "平静"))[:8],
        "intent_type": intent_type,
        "topics": [str(t)[:12] for t in topics[:3]],
        "urgency": _num(obj.get("urgency"), 0.0, 1.0, 0.3),
        "cancel_topics": [str(t)[:12] for t in (cancel if isinstance(cancel, list) else [])[:3]],
        "defer_topics": [str(t)[:12] for t in (defer if isinstance(defer, list) else [])[:3]],
        "done_topics": [str(t)[:12] for t in (done if isinstance(done, list) else [])[:3]],
        "hand_action": hand_out,
        "reply_plan": plan_out,
        "text_head": text[:16],
    }


def perceive(client, text: str, seed: int, perception_cfg, context: dict = None) -> dict:
    """调用 LLM 做消息洞察（带上下文消解指代）；mock/失败/解析失败一律返回 None。"""
    if client is None or client.name == "mock" or not perception_cfg.enabled:
        return None
    try:
        raw = client.render_raw(
            build_task(text, context), seed,
            temperature=perception_cfg.temperature,
            max_tokens=perception_cfg.max_tokens)
        obj = extract_json(raw)
        if obj is None:
            print(f"[perception] JSON 解析失败，回退启发式。原始输出：{raw[:120]!r}")
            return None
        return _clamp_output(obj, text, valid_hands={
            m["name"] for m in (context or {}).get("hands", [])})
    except Exception as e:  # noqa: BLE001  洞察失败绝不阻塞主链路
        print(f"[perception] 调用失败，回退启发式：{e!r}")
        return None


# 长度决策信号：intent_type 到"该不该详细说"的映射（模型要决策：有必要详细说
# 的情况下就详细说，没必要就继续短说）。感知为 None（mock/失败回退）时默认
# False——与既有闲聊短说行为一致，感知链路异常不改变默认人设。
DETAIL_INTENTS = ("提问", "求助建议")

# 明确的"要内容"请求：讲故事/继续/展开/解释——意图分类常把它们归到
# 提出计划/闲聊，只按 intent_type 判会漏。漏判的代价是审核按闲聊 36 字兜底：
# 2026-10-30 实测原稿 850 字的平行世界故事被砍成"那我一口气给你讲完" 9 个字，
# 用户看到的就是"老是不讲"。
_DETAIL_REQ = re.compile(
    r"继续|接着|然后呢|还有呢|往下|展开|详细|一口气|讲完|说完|多讲|多说|从头|"
    r"讲个|讲一下|讲讲|讲故事|说个|说说|写个|写篇|编个|编一个|故事|来龙去脉|"
    r"原理|区别|对比|分析|解释|介绍|科普|梳理|捋|"
    # 选品/推荐：要的是型号和参数，短回等于没答。2026-11-01 实测"那推荐几款吧，
    # 给出京东连接"被当闲聊，36 字兜底只留下"…型号：小米4"，后半句全没了。
    r"推荐|几款|哪款|怎么选|选型|型号|参数|链接|连接")


def needs_detail(perc: dict) -> bool:
    """从感知结果派生"这条回复该不该详细"。

    判定：提问/求助建议要详细（技术问答、要方法要链接）；正文里明确要求
    "继续/讲完/展开/编个故事"的也要详细（需求本身就是要内容，短回等于没答）。
    分享/提出计划/闲聊/情绪安抚默认短说——群里日常分享琐事时给它
    "必要信息全给到"的指令，它会写成两段式讲课，和插嘴的短句形成反差
    （2026-10-29 实测：@ 回复 60~120 字两段式 vs 插嘴 5~15 字，"看着怪怪的"）。
    urgency 高的求助即使意图识别偏了也放行（宁可详细不可漏事）。"""
    if not perc:
        return False
    if perc.get("intent_type") in DETAIL_INTENTS:
        return True
    if float(perc.get("urgency", 0) or 0) >= 0.7 and perc.get("topics"):
        return True
    head = str(perc.get("text_head") or "")
    return bool(head and _DETAIL_REQ.search(head))
