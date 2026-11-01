"""审核 Agent（前额叶）：不说话，只判断和修正。与互动 Agent（杏仁核，
负责情绪化表达）共同组成"一个人"。

双 Agent 模式（v2，2026-10-28 真正启用）：
- 互动 Agent（杏仁核）：把选定的意图渲染成一句要发到微信的话。
- 审核 Agent（本模块）：独立的第二次调用 + 独立提示词，只做判断——
  这句话会翻车吗？翻车就给一句改写要求，交回互动 Agent 重写。
  它不是正则补丁：规则检查只是快路径，命中/高风险时由 LLM 审稿员终审。

分工（用户架构）：
- 输入侧（复用既有器官，收拢到本模块的门下）：
  意图猜测 = perception；工具调用 = perception.hand_action + hands；
  风格化建议 = reply_plan + style_ctrl + state_style 约束
- 输出侧（本模块）：互动 Agent 每条回复出门前过审——
  能力否认检测（"我发不了语音"这类，实测 deepseek 世界观里 AI 不出声，
  提示词说服不了，必须出门拦下打回重写）、AI 自曝、客服腔、答非所问、
  群聊站队、微信口语度、主动消息空转。

失效回忆曲线：cap_fail 计数在 agent 行里持久化——否认一次 +1，
干净一轮 -1，>0 时 prompt 常驻能力提醒（回忆注入），归零后撤除。
"""
import re
import time

# 否认短语 × 能力词，出现在同一句话（60 字窗口）内即判定。
# 2026-10-29 增补"编造型"变体：模型会用"语音键坏了/只剩打字版"这类
# 拟人借口假装没能力（鱼群真实病例："语音键坏了，嘴替功能只剩打字版"——
# 规则漏判、chime 路径也没进 LLM 审稿，裸奔发出）。
_DENY = (r"(?:发不出|发不了|不能发|没法发|没办法发|做不到|没.{0,4}能力|"
         r"无法|不会说|说不了|只能打字|只剩?[^。！？!?\n]{0,4}打字|"
         r"(?:键|功能|通道|模块)[^。！？!?\n]{0,3}(?:坏|失灵|失效)|"
         r"没这个功能|不支持|看不了|听不了|查不了)")
_CAP = (r"(?:语音|声音|说话|音频|搜索|联网|查不到.{0,6}网|画图|画一张|"
        r"看图|看不到图|识别图|表情|图|视频|评论)")
DENIAL_RE = re.compile(
    r"{0}[^。！？!?\n]{{0,40}}{1}|{1}[^。！？!?\n]{{0,40}}{0}".format(_DENY, _CAP))

# AI 挡箭牌：人格号可以承认自己是 AI，但不能拿 AI 身份当挡箭牌、加一段设定说明。
# 2026-10-29 修正：原规则连"坦承自己是 AI"一起拦，与角色卡 identity_summary /
# boundaries 的"坦承自己是 AI，不假装人类"自相矛盾，模型每次开口都要二选一。
AI_SELF_RE = re.compile(
    r"(?:作为(?:一个|一名)?(?:AI|人工智能|大模型|语言模型|智能助手|助手|聊天机器人)|"
    r"我(?:只|不过|只不过)?是(?:一个|一名)?(?:AI|人工智能|大模型|语言模型)?"
    r"(?:助手|智能助手|聊天机器人|机器人助手)|"
    r"我的训练(?:数据|语料)|我是由[^，。]{0,10}(?:开发|训练|制造)的)")

# 客服腔：自称服务/工具，与"群友"人设冲突
SERVICE_RE = re.compile(
    r"(?:请问还有什么|还有什么(?:可以|能)帮|很高兴为您服务|有什么可以帮(?:到)?您|"
    r"作为一个助手|本助手|如需(?:帮助|协助)请|希望以上[^。]{0,8}(?:有帮助|帮到你)|"
    r"您可以(?:随时)?(?:联系|告诉)我)")

_OPENER_RE = re.compile(r"^\s*(诶[，,]?|嗯[，,]?哼|行[，,吧 ]|好家伙|绝了|乐[，,]?|"
                        r"哈哈[，,]?|家人们)")

# emoji 用量：真人微信一条最多挂一个，且极少用"尴尬笑"那一挂
# 内部控制标记（[skip]/[跳过]）：泄露到正文里对方会看到，属于硬病灶
_SKIP_MARK_RE = re.compile(r"[\[【]\s*(?:skip|跳过)\s*[\]】]", re.I)

# 干货形态：链接/代码/命令/列点——正经内容，不按闲聊长度砍（旧正则，保留）。
_GOODS_RE = re.compile(
    r"https?://|www\.|```|`[^`\n]{2,}`|\b(?:pip|npm|git|cd|ls|python)\b|^[ \t]*[-*•]",
    re.M)
# 商品/型号推荐也是一种"干货"：中文顿号并列的型号清单、或"京东/CADR + 型号"
# 这种可执行信息，和链接/代码同级（2026-11-01 真实病例：群里问"推荐几款"，
# 回复"京东搜：352 X86C 或者米家Ultra，CADR都在600上下，预算低点就小米5S…"
# 被 36 字兜底切成"…CADR都在600上下"，后半句的型号全没了，读起来像消息断了）。
_SHOP_RE = re.compile(r"京东|淘宝|天猫|拼多多|旗舰店|型号|CADR|参数|搜[：:]|购物车")
_MODEL_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"\d{2,4}\s?[A-Za-z]{1,4}\d{1,4}[A-Za-z]{0,4}|"      # 352 X86C
    r"\d{1,4}[A-Za-z]{1,6}|"                              # 小米5S / 米家4lite
    r"[A-Za-z]{2,}\s?\d{1,4}[A-Za-z]{0,4}|"               # X86C / lite4
    r"(?:Lite|Pro|Max|Ultra|Plus|Mini)"                   # 米家Ultra
    r")(?![A-Za-z0-9])", re.I)


def is_substantive(text: str) -> bool:
    """这条回复是不是在给具体信息（链接/代码/列点/型号清单）。

    是的话长度按内容走：闲聊 36 字上限不适用，截断会把"给的东西"砍成残句。
    """
    s = (text or "").strip()
    if not s:
        return False
    if _GOODS_RE.search(s):
        return True
    # 一句里两个顿号以上 = 在列东西（"小米4 Pro、米家全效、352 X86C"）
    if s.count("、") >= 2:
        return True
    # 型号 + 购买语境：即使不并列，也是能照着去买的可执行信息
    return bool(_SHOP_RE.search(s) and _MODEL_RE.search(s))


_FLAT_Q_RE = re.compile(
    r"(?:啥|什么|哪|怎么|为什么|谁|多少|几点|哪儿)[^。！？!?\n]{0,14}$"
    r"|([\u4e00-\u9fff])不\1$")   # 正反问只认 A不A（关不关/是不是/要不要），别误伤"不急"

_EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B50\u2764]")
_BANNED_EMOJI = set("😂😅🤣😭💧🫠🙃")

# 硬病灶：真正会翻车的问题——规则命中这些才请 LLM 审稿员终审。
# 风格小毛病（tic_*/too_long/emoji/control_leak）走机械修复，不花 LLM。
RISK_ISSUES = ("capability_denial", "ai_self", "service_talk")

# 熟人 loose 模式只放松表达层：这些是"像不像微信短句"的审美约束，不是安全问题。
# control_leak 不在此列——内部标记泄露到正文是硬 bug；empty/能力否认也照旧拦。
SOFT_ISSUES = ("too_long", "tic_period", "tic_opener", "tic_twoline",
               "emoji_spam", "flat_question", "only_question", "question_tail")

# 规则病灶 -> 给互动 Agent 的改写要求（一句一条，拼起来当 correction）
_FIX_TEXT = {
    "capability_denial": "你上一版否认了自己真实拥有的能力（见能力清单），重写时按清单执行"
                         "（对方要语音就以「[语音]」开头），绝不出现'发不了/做不到/只能打字'",
    "ai_self": "可以承认自己是 AI，但别拿 AI 身份当挡箭牌、别加一段设定说明；"
               "直接给反应，把这话正常说完",
    "service_talk": "别用客服腔（'还有什么可以帮您''很高兴为您服务'），你是群友不是客服",
    "tic_period": "把句号全删掉——微信打字不说完整书面句，短句之间用换行或直接断开",
    "tic_opener": "开头直接从内容说起，删掉 诶/行/好家伙/绝了/乐 这类起手词，"
                  "也别拿'哈哈'垫在句尾",
    "tic_twoline": "写成一行，别搞两段式",
    "emoji_spam": "别挂 😂😅🤣 这类尴尬笑表情；一条最多一个表情，句尾别把表情当标点",
    "control_leak": "别把 [skip]/[跳过] 这类内部标记写进正文：不是对你说的就只回 [skip]，要说话就只说正文",
    "flat_question": "这是问句，句末要带问号（该带'吗/呢/吧'就带上）：'我错过啥了' -> '我错过啥了吗？'",
    "too_long": "太长了，压到两句以内（总长不超过 36 字）：只留最要紧的那半句，其余别说",
    "only_question": "别只回一个问句：先接住对方那句、给个态度，再决定要不要问",
    "question_tail": "结尾别用问句收：把最后那句问话换成你自己的判断或吐槽",
}


def mechanical_fix(reply: str, issues: list) -> str:
    """AI 腔病灶的机械修正（确定性，不再多花一次 LLM）：删句号、删起手词。"""
    text = reply or ""
    if "tic_period" in issues:
        # 句号在微信里确实不用，但直接删会把两句黏死（实测"随口一喷方案是卡在…"）。
        # 句中的句号换成逗号，句尾的删掉。
        text = re.sub(r"。[ \t]*\n", "\n", text)
        text = re.sub(r"。+(?=[^\s])", "，", text)
        text = re.sub(r"。+\s*$", "", text)
        text = text.replace("，\n", "\n")
    if "tic_opener" in issues:
        text = _OPENER_RE.sub("", text, count=1).lstrip("，, ")
    if "tic_twoline" in issues:
        import re as _re
        text = _re.sub(chr(10) + "+", "，", text)
    if "flat_question" in issues:
        s = text.rstrip("。,.，、 ")
        if s and not s.endswith(("？", "?")):
            text = s + "？"
    if "control_leak" in issues:
        text = _SKIP_MARK_RE.sub("", text).strip()
    if "emoji_spam" in issues:
        text = "".join(ch for ch in text if ch not in _BANNED_EMOJI)
        seen = 0
        out = []
        for ch in text:
            if _EMOJI_RE.match(ch):
                seen += 1
                if seen > 1:
                    continue
            out.append(ch)
        text = "".join(out).replace("  ", " ").strip()

    return text


def rule_issues(reply: str, context: dict = None) -> list:
    """规则快路径：确定性可判的病灶（零成本、零延迟）。"""
    text = reply or ""
    issues = []
    if not text.strip():
        issues.append("empty")
        return issues
    if DENIAL_RE.search(text):
        issues.append("capability_denial")
    if AI_SELF_RE.search(text):
        issues.append("ai_self")
    if SERVICE_RE.search(text):
        issues.append("service_talk")
    # AI 腔病灶（实测 60 条：31 条句中句号、30 条固定起手词）
    if "。" in text:
        issues.append("tic_period")
    if _OPENER_RE.match(text):
        issues.append("tic_opener")
    if chr(10) in text and len(text) <= 45:
        issues.append("tic_twoline")   # 短回复搞两段式（实测 64-80% 中招）
    _stripped = text.strip()
    # 离线语料（LCCC 真实对话）实测：人类回复中位 10.5 字，模型中位 19 字。
    # 阈值与人格卡"最多两句、不超 24 字"对齐。
    # 两层豁免（长度决策权在感知层，规则只做闲聊兜底）：
    # 1) needs_detail：感知已判"该详细说"（提问/求助）——
    #    实测病灶：362 字含 GitHub 链接与列点的技术分析被 18 字兜底腰斩成
    #    "pip 能装，读写都行，渲染别指望"，有用信息全灭；
    # 2) 干货形态：回复自带链接/代码/命令/列点/型号清单 = 正经内容
    #    （型号清单见 is_substantive：顿号并列 / 京东·CADR + 型号）。
    _detail_ok = bool((context or {}).get("needs_detail")) or is_substantive(_stripped)
    # 阈值 36 = 人格卡"最多两句、不超 36 字"；旧值 18 比人设还严，专砍包袱：
    # 2026-10-29 实测两连"资格"惨案——完整俏皮话"怎么，我连抽的资格都没有？
    # 是奖品太差看不上我…"被砍成前半句，读起来像在闹情绪。
    if (len(_stripped) > 36 and not _stripped.endswith(("？", "?"))
            and not _detail_ok):
        issues.append("too_long")
    if ("？" not in text and "?" not in text and _FLAT_Q_RE.search(_stripped)
            and not _stripped.endswith(("没什么", "没啥", "不什么", "怎么着"))):
        issues.append("flat_question")   # 疑问句没问号：语气会被读成陈述
    _user_msg = str((context or {}).get("user_message") or "")
    if (_stripped.endswith(("？", "?")) and "，" in _stripped
            and len(_stripped) >= 12 and _user_msg
            and "？" not in _user_msg and "?" not in _user_msg):
        issues.append("question_tail")   # 陈述+问句结尾：真人很少这么收尾（语料 0%）
    if (_stripped.endswith(("？", "?")) and "，" not in _stripped and "。" not in _stripped
            and len(_stripped) <= 20 and _user_msg
            and "？" not in _user_msg and "?" not in _user_msg):
        issues.append("only_question")   # 光反问、不给态度：像把话推回去
    if _SKIP_MARK_RE.search(text):
        issues.append("control_leak")
    _emojis = _EMOJI_RE.findall(text)
    if _emojis and (len(_emojis) > 1 or any(e in _BANNED_EMOJI for e in _emojis)):
        issues.append("emoji_spam")    # 表情刷屏/尴尬笑（实测 24h 有 26 条带 emoji）
    if (context or {}).get("loose"):
        # 熟人局：长度/句号/起手词/问句结尾/表情这些表达层约束全部放行。
        # 硬病灶（能力否认/AI 自曝/客服腔/空回复/内部标记泄露）照旧。
        issues = [i for i in issues if i not in SOFT_ISSUES]
    return issues


def build_correction(issues: list, extra: str = "") -> str:
    parts = [_FIX_TEXT[i] for i in issues if i in _FIX_TEXT]
    if extra:
        parts.append(str(extra).strip())
    if not parts:
        return ""
    return "；".join(parts) + "。只输出重写后的回复。"


# ---------------- LLM 审稿员（真正的第二个 Agent） ----------------

REVIEW_CHECKLIST = (
    "你是这个微信人格号的「审核 Agent」（前额叶）：你不说话，只做判断。\n"
    "互动 Agent 刚写好一句准备发到微信的话，你在出门前拦毛病。只判下面几类，别管文笔偏好：\n"
    "1 capability_denial：否认自己真实具备的能力（语音/搜索/看图/发图/表情），"
    "如'我发不了语音''只能打字'\n"
    "2 ai_self：自曝是 AI/模型/助手，或声明自己没有情感\n"
    "3 service_talk：客服腔（'还有什么可以帮您''很高兴为您服务'）\n"
    "4 off_topic：答非所问，或认错人（把 A 说的当成 B 说的）\n"
    "5 take_side：群聊里替谁站队去攻击别人、跟着挑拨起哄（可以有态度，不能当枪；"
    "朋友间玩梗式互损如'杂鱼/菜鸡'不算，真辱骂、家人隐私、外貌攻击才拦）\n"
    "6 unnatural：明显不像微信口语（长段书面语、分点罗列、总结陈词、一句话塞满解释）\n"
    "7 empty_proactive：主动开口却只有寒暄/空问句，没有任何具体由头\n"
)

_REVIEW_JSON_HINT = (
    "\n只输出一个 JSON 对象，禁止输出其他任何文字：\n"
    '通过：{"pass": true, "issues": [], "fix": ""}\n'
    '不通过：{"pass": false, "issues": ["capability_denial"], '
    '"fix": "给作者的一句话改写要求，别重写全文"}'
)


def build_review_prompt(text: str, context: dict = None) -> str:
    ctx = context or {}
    purpose = str(ctx.get("purpose") or "respond")
    purpose_cn = {"respond": "回对方一句话", "proactive": "自己主动开口（没人问我）",
                  "chime": "群里自己插一句", "reminder": "提醒到点",
                  "group_reply": "在群里回一句"}.get(purpose, purpose)
    lines = [REVIEW_CHECKLIST, "【这条消息的用途】" + purpose_cn]
    if ctx.get("group"):
        lines.append("【场景】微信群聊（旁边还有别人在看）")
    if ctx.get("loose"):
        lines.append("【熟人 loose 模式】这个群是熟人局：允许 1-3 句、跑题、吐槽、抽象联想、"
                     "回扣旧梗，不要因为篇幅、口语度、问句结尾判 unnatural 或 off_topic；"
                     "仍只拦能力否认、AI 自曝、客服腔、认错人、站队攻击。")
    user_msg = str(ctx.get("user_message") or "").strip()
    if user_msg:
        lines.append("【对方刚说的】" + user_msg[:600])
    recent = ctx.get("recent") or []
    if recent:
        lines.append("【群里最近几句】" + " / ".join(str(x)[:120] for x in recent[-6:]))
    lines.append("【要发出去的话】" + text[:600])
    lines.append("【判断纪律】上下文不够时只判话术问题（自曝/客服腔/不像微信口语），"
                 "不要凭一句半句判 off_topic 或 take_side；拿不准就放行。")
    lines.append(_REVIEW_JSON_HINT)
    return chr(10).join(lines)


def parse_review(raw: str):
    """从审稿员输出里取 JSON；拿不到就返回 None（当作没审，交回规则）。"""
    if not raw or not raw.strip():
        return None
    try:
        from .perception import extract_json
        data = extract_json(raw)
    except Exception:  # noqa: BLE001  解析失败=没审
        data = None
    if not isinstance(data, dict):
        return None
    issues = data.get("issues") or []
    if isinstance(issues, str):
        issues = [issues]
    issues = [str(i).strip() for i in issues if str(i).strip()]
    passed = data.get("pass")
    if passed is None:            # 模型只给了 issues，就用 issues 反推
        passed = not issues
    return {"pass": bool(passed), "issues": issues,
            "fix": str(data.get("fix") or "").strip()[:300]}


class ReviewAgent:
    """审核 Agent：规则快路径 + LLM 审稿员。mode=off|rule|llm。"""

    def __init__(self, cfg, client=None):
        self.cfg = cfg
        self.client = client
        self.today = ""
        self.calls_today = 0

    # ---- 配额（防审稿把预算烧穿）----
    def _llm_allowed(self, purpose: str, rule_hits: list) -> bool:
        if self.cfg.mode != "llm" or self.client is None:
            return False
        if self.calls_today >= int(self.cfg.max_llm_per_day or 0) > 0:
            return False
        # 白名单语义：显式列出的用途每条都审；**留空 = 不按用途放行**，只按风险触发。
        # （旧实现 `or ["respond"]` 把留空回退成"respond 每条都审"——与配置注释
        # 相反，且是延迟大头：审核 LLM 单次实测 p50 2.4~7.7s。）
        purposes = list(self.cfg.llm_purposes or [])
        if purpose in purposes:
            return True
        # 风险触发只认"硬病灶"：能力否认/AI 自曝/客服腔——真正会翻车的。
        # 风格小毛病（句号/起手词/两段式/超长/emoji）走机械修复，不值得花一次 LLM。
        return bool(self.cfg.risk_triggers
                    and any(i in RISK_ISSUES for i in (rule_hits or [])))

    def review(self, text: str, context: dict = None, allow_llm: bool = True) -> dict:
        """出门过审。返回 {ok, issues, correction, mode, llm_used, llm_failed, latency_ms}。"""
        ctx = dict(context or {})
        purpose = str(ctx.get("purpose") or "respond")
        started = time.time()
        if self.cfg.mode == "off":
            return {"ok": True, "issues": [], "correction": None, "mode": "off",
                    "llm_used": False, "llm_failed": False, "latency_ms": 0}
        issues = rule_issues(text, ctx)
        llm_used = False
        llm_failed = False
        extra_fix = ""
        if allow_llm and self._llm_allowed(purpose, issues):
            day = time.strftime("%Y-%m-%d")
            if day != self.today:
                self.today, self.calls_today = day, 0
            self.calls_today += 1
            llm_used = True
            try:
                raw = self.client.render_raw(
                    build_review_prompt(text, ctx), 0,
                    temperature=float(self.cfg.temperature),
                    max_tokens=int(self.cfg.max_tokens),
                    timeout=float(self.cfg.timeout_seconds))
            except TypeError:      # 老客户端没有 timeout 形参
                try:
                    raw = self.client.render_raw(
                        build_review_prompt(text, ctx), 0,
                        temperature=float(self.cfg.temperature),
                        max_tokens=int(self.cfg.max_tokens))
                except Exception:  # noqa: BLE001  审稿调用失败=不阻断
                    raw, llm_failed = "", True
            except Exception:  # noqa: BLE001  审稿调用失败=不阻断
                raw, llm_failed = "", True
            verdict = parse_review(raw)
            if verdict is None:
                llm_failed = llm_failed or not (raw or "").strip()
            else:
                for name in verdict["issues"]:
                    if name not in issues:
                        issues.append(name)
                extra_fix = verdict["fix"]
        if ctx.get("loose"):
            issues = [i for i in issues if i not in SOFT_ISSUES]
        return {"ok": not issues, "issues": issues,
                "correction": build_correction(issues, extra_fix) or None,
                "mode": self.cfg.mode, "llm_used": llm_used,
                "llm_failed": llm_failed,
                "latency_ms": int((time.time() - started) * 1000)}


def review_output(reply: str, capabilities_hint: list = None) -> dict:
    """兼容旧接口：纯规则过审。返回 {ok, issues, correction}。"""
    issues = rule_issues(reply)
    return {"ok": not issues, "issues": issues,
            "correction": build_correction(issues) or None}
