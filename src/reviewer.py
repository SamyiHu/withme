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

# 否认短语 × 能力词，出现在同一句话（60 字窗口）内即判定
_DENY = (r"(?:发不出|发不了|不能发|没法发|没办法发|做不到|没.{0,4}能力|"
         r"无法|不会说|说不了|只能打字|没这个功能|不支持|看不了|听不了|查不了)")
_CAP = (r"(?:语音|声音|说话|音频|搜索|联网|查不到.{0,6}网|画图|画一张|"
        r"看图|看不到图|识别图|表情|图|视频|评论)")
DENIAL_RE = re.compile(
    r"{0}[^。！？!?\n]{{0,40}}{1}|{1}[^。！？!?\n]{{0,40}}{0}".format(_DENY, _CAP))

# AI 自曝：人格号不能承认自己是模型/助手（等同"机"格分裂）
AI_SELF_RE = re.compile(
    r"(?:我是(?:一个|一名)?(?:AI|人工智能|大模型|语言模型|机器人助手|智能助手|聊天机器人)|"
    r"作为(?:一个|一名)?(?:AI|人工智能|语言模型|智能助手)|"
    r"我的训练(?:数据|语料)|我是由[^，。]{0,10}(?:开发|训练|制造)的|"
    r"我(?:并)?(?:没有|不具备)(?:真实)?(?:的)?(?:情感|意识|感情|情绪))")

# 客服腔：自称服务/工具，与"群友"人设冲突
SERVICE_RE = re.compile(
    r"(?:请问还有什么|还有什么(?:可以|能)帮|很高兴为您服务|有什么可以帮(?:到)?您|"
    r"作为一个助手|本助手|如需(?:帮助|协助)请|希望以上[^。]{0,8}(?:有帮助|帮到你)|"
    r"您可以(?:随时)?(?:联系|告诉)我)")

_OPENER_RE = re.compile(r"^\s*(诶[，,]?|嗯[，,]?哼|行[，,吧 ]|好家伙|绝了|乐[，,]?|"
                        r"哈哈[，,]?|家人们)")

# 规则病灶 -> 给互动 Agent 的改写要求（一句一条，拼起来当 correction）
_FIX_TEXT = {
    "capability_denial": "你上一版否认了自己真实拥有的能力（见能力清单），重写时按清单执行"
                         "（对方要语音就以「[语音]」开头），绝不出现'发不了/做不到/只能打字'",
    "ai_self": "别自曝是 AI/模型/助手，也别声明自己没有情感——你是人，直接给反应",
    "service_talk": "别用客服腔（'还有什么可以帮您''很高兴为您服务'），你是群友不是客服",
    "tic_period": "把句号全删掉——微信打字不说完整书面句，短句之间用换行或直接断开",
    "tic_opener": "开头直接从内容说起，删掉 诶/行/好家伙/绝了/乐 这类起手词，"
                  "也别拿'哈哈'垫在句尾",
    "tic_twoline": "写成一行，别搞两段式",
}


def mechanical_fix(reply: str, issues: list) -> str:
    """AI 腔病灶的机械修正（确定性，不再多花一次 LLM）：删句号、删起手词。"""
    text = reply or ""
    if "tic_period" in issues:
        text = text.replace("。", "")
    if "tic_opener" in issues:
        text = _OPENER_RE.sub("", text, count=1).lstrip("，, ")
    if "tic_twoline" in issues:
        import re as _re
        text = _re.sub(chr(10) + "+", "，", text)

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
    "5 take_side：群聊里替谁站队去攻击别人、跟着挑拨起哄（可以有态度，不能当枪）\n"
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
    user_msg = str(ctx.get("user_message") or "").strip()
    if user_msg:
        lines.append("【对方刚说的】" + user_msg[:200])
    recent = ctx.get("recent") or []
    if recent:
        lines.append("【群里最近几句】" + " / ".join(str(x)[:60] for x in recent[-4:]))
    lines.append("【要发出去的话】" + text[:600])
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
        purposes = self.cfg.llm_purposes or ["respond"]
        if purpose in purposes:
            return True
        # 不在名单里：只有规则已经抓到问题、且开了风险触发，才请审稿员终审
        return bool(self.cfg.risk_triggers and rule_hits)

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
