"""聊天里的参数配置：斜杠命令 -> 校验 -> 复用 Agent / 桥现成的落点。

几条硬约束（都是踩过的坑）：
- 只认主人；群里还要 @（普通文本里的 "/" 不误吃，别人发的命令直接丢）
- 命令不进对话历史/记忆，由桥直接回执——否则人格记忆里混进一堆 /set
- 每个可调项在 PARAMS 里声明类型/范围/作用域，非法值不动配置
- 写操作记 config_audit（在会话自己的库里），/undo 回滚上一笔
- Agent 级改动走 apply_settings/apply_config（按会话库持久化）；
  桥级改动写进程内 cfg + BridgeState.overrides（重启后自动恢复）

取值优先级：config.yaml 是默认值，命令改的是运行期覆盖；/undo 回到上一笔。
"""
import json
import re
import time

from .storage import get_setting


class CmdError(Exception):
    """参数不合法：消息直接回给主人，配置保持原样。"""


class Param:
    """一个可调项：名字、作用域、类型/范围、群内是否可用、读写闭包。"""

    def __init__(self, name, scope, kind, help_text, getter, setter,
                 lo=None, hi=None, options=(), group_ok=True, field=None):
        self.name = name
        self.scope = scope          # agent=本会话库 | bridge=桥进程全局
        self.kind = kind            # float|int|bool|enum|text|quiet|chat
        self.help = help_text
        self.getter = getter
        self.setter = setter
        self.lo = lo
        self.hi = hi
        self.options = tuple(options)
        self.group_ok = group_ok
        self.field = field          # 桥级：cfg.wechat 上的属性名


class Ctx:
    """一次命令的上下文。agent 可能为 None（单用户模式没有本地 Agent）。"""

    __slots__ = ("agent", "bridge", "peer", "is_group", "is_owner", "now")

    def __init__(self, agent, bridge, peer, is_group=False, is_owner=True, now=None):
        self.agent = agent
        self.bridge = bridge
        self.peer = str(peer or "")
        self.is_group = bool(is_group)
        self.is_owner = bool(is_owner)
        self.now = time.time() if now is None else now


# ---------------- 取值/校验/展示 ----------------

_TRUE = ("on", "1", "true", "yes", "开", "是")
_FALSE = ("off", "0", "false", "no", "关", "否")


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "开" if v else "关"
    if isinstance(v, float):
        return ("%.3f" % v).rstrip("0").rstrip(".")
    if isinstance(v, (list, tuple)):
        return ",".join(str(x) for x in v) if v else "off"
    if v is None:
        return "未设"
    return str(v)


def _parse_bool(raw) -> bool:
    if isinstance(raw, bool):
        return raw
    s = str(raw).strip().lower()
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    raise CmdError("只能是 on / off")


def _coerce(p: Param, raw):
    """把命令里的字符串（或 /undo 回放的类型值）变成参数要的类型。"""
    if p.kind == "bool":
        return _parse_bool(raw)
    if p.kind == "chat":
        return _parse_bool(raw)
    if p.kind in ("float", "int"):
        try:
            n = float(str(raw).strip())
        except (TypeError, ValueError):
            raise CmdError(f"{p.name} 要数字")
        if p.kind == "int":
            n = int(round(n))
        if p.lo is not None and not (p.lo <= n <= p.hi):
            raise CmdError(f"{p.name} 只能 {_fmt(float(p.lo))}–{_fmt(float(p.hi))}")
        return n
    if p.kind == "enum":
        s = str(raw).strip().lower()
        if s not in p.options:
            raise CmdError(f"{p.name} 只能 {' / '.join(p.options)}")
        return s
    if p.kind == "text":
        s = str(raw).strip()
        if not s:
            raise CmdError(f"{p.name} 不能为空")
        if len(s) > 64:
            raise CmdError(f"{p.name} 太长了（>64 字）")
        return s
    if p.kind == "quiet":
        if isinstance(raw, (list, tuple)):
            spans = [str(x).strip() for x in raw if str(x).strip()]
        else:
            s = str(raw).strip().lower()
            if s in ("off", "none", "关", "清空"):
                return []
            spans = [x.strip() for x in s.replace("，", ",").split(",") if x.strip()]
        from .config import _parse_quiet_span
        for sp in spans:
            if not re.match(r"^\d{2}:\d{2}-\d{2}:\d{2}$", sp):
                raise CmdError("安静时段格式是 23:00-08:00（或 off）")
            try:
                _parse_quiet_span(sp)
            except Exception:  # noqa: BLE001
                raise CmdError(f"时间不合法：{sp}")
        return spans
    return raw


# ---------------- Agent 级参数 ----------------

def _level_on(agent, level: str) -> bool:
    disabled = get_setting(agent.conn, "proactive_disabled_levels", {}) or {}
    return bool(agent.persona.active_permissions.get(level, False)) and not disabled.get(level, False)


def _agent_num(name, section, key, lo, hi, help_text, cast=float, group_ok=True):
    return Param(
        name, "agent", "int" if cast is int else "float", help_text,
        lambda c: getattr(getattr(c.agent.config, section), key),
        lambda c, v: c.agent.apply_config({section: {key: v}}),
        lo=lo, hi=hi, group_ok=group_ok)


def _threshold(name, level):
    return Param(
        name, "agent", "float", f"{level} 阈值（0–1，越小越爱开口）",
        lambda c: c.agent.config.gate.thresholds[level],
        lambda c, v: c.agent.apply_config({"gate": {"thresholds": {level: v}}}),
        lo=0.0, hi=1.0)


def _agent_bool(name, help_text, getter, setter, group_ok=True):
    return Param(name, "agent", "bool", help_text, getter, setter, group_ok=group_ok)


# ---------------- 桥级参数（cfg.wechat + BridgeState.overrides） ----------------

def _set_bridge(c, field, value):
    setattr(c.bridge.cfg.wechat, field, value)
    st = getattr(c.bridge, "state", None)
    if st is not None:
        st.overrides[field] = value
        st.save()


def _bridge_param(name, field, kind, help_text, lo=None, hi=None, options=(),
                  group_ok=True):
    return Param(
        name, "bridge", kind, help_text,
        lambda c: getattr(c.bridge.cfg.wechat, field),
        lambda c, v: _set_bridge(c, field, v),
        lo=lo, hi=hi, options=options, group_ok=group_ok, field=field)


def _chat_enabled(c) -> bool:
    w = c.bridge.cfg.wechat
    peer = str(c.peer)
    disabled = [str(x) for x in (getattr(w, "disabled_chats", None) or [])]
    allow = [str(x) for x in (getattr(w, "enabled_chats", None) or [])]
    if peer in disabled:
        return False
    return not allow or peer in allow


def _set_chat(c, value):
    """本会话开关。off -> 只进停用名单（disabled 优先），不动白名单：
    白名单是"只开这些"，删掉唯一一项会变成"其余全开"，副作用太大。
    on  -> 出停用名单；白名单生效时把自己补进去。"""
    w = c.bridge.cfg.wechat
    peer = str(c.peer)
    disabled = [str(x) for x in (getattr(w, "disabled_chats", None) or [])]
    allow = [str(x) for x in (getattr(w, "enabled_chats", None) or [])]
    if value:
        disabled = [x for x in disabled if x != peer]
        if allow and peer not in allow:
            allow.append(peer)
    else:
        if peer not in disabled:
            disabled.append(peer)
    w.disabled_chats = disabled
    w.enabled_chats = allow
    st = getattr(c.bridge, "state", None)
    if st is not None:
        st.overrides["disabled_chats"] = list(disabled)
        st.overrides["enabled_chats"] = list(allow)
        st.save()


# ---------------- 注册表 ----------------

PARAMS = []
_BY_NAME = {}


def _reg(*params):
    for p in params:
        PARAMS.append(p)
        _BY_NAME[p.name] = p


_reg(_threshold("th1", "L1"), _threshold("th2", "L2"), _threshold("th3", "L3"))
_reg(
    _agent_num("tick", "tick", "fast_seconds", 15, 3600,
               "主动评估间隔（秒）"),
    _agent_num("budget", "cost", "daily_token_budget", 0, 100000000,
               "每日 token 预算", cast=int),
    _agent_num("window", "dialogue", "window", 4, 64, "对话窗口条数", cast=int),
    _agent_num("compress", "dialogue", "compress_trigger", 8, 500,
               "压缩触发条数", cast=int),
)
_reg(
    _agent_bool("proactive", "主动行为总开关",
                lambda c: bool(get_setting(c.agent.conn, "proactive_enabled", True)),
                lambda c, v: c.agent.set_proactive(v)),
    _agent_bool("perception", "感知层开关",
                lambda c: c.agent.config.perception.enabled,
                lambda c, v: c.agent.apply_settings({"perception": {"enabled": v}})),
    _agent_bool("reflection", "反思开关",
                lambda c: c.agent.config.reflection.enabled,
                lambda c, v: c.agent.apply_config({"reflection": {"enabled": v}})),
)
for _lvl in ("L1", "L2", "L3"):
    _reg(Param(
        _lvl.lower(), "agent", "bool", f"主动 {_lvl} 开关",
        (lambda lv: lambda c: _level_on(c.agent, lv))(_lvl),
        (lambda lv: lambda c, v: c.agent.set_level_enabled(lv, v))(_lvl)))
_reg(
    Param("roast", "agent", "enum", "玩梗互损尺度 off/mild/on",
          lambda c: (c.agent.persona.style.get("roast_level") or "off"),
          lambda c, v: c.agent.apply_settings({"persona_style": {"roast_level": v}}),
          options=("off", "mild", "on")),
    Param("review", "agent", "enum", "审核模式 off/rule/llm",
          lambda c: c.agent.config.reviewer.mode,
          lambda c, v: c.agent.apply_config({"reviewer": {"mode": v}}),
          options=("off", "rule", "llm")),
    Param("model", "agent", "text", "当前模型（换模型只说名字）",
          lambda c: c.agent.config.llm.model,
          lambda c, v: c.agent.apply_settings({"llm": {"model": v}}),
          group_ok=False),
    Param("quiet", "agent", "quiet", "安静时段 23:00-08:00（off=清空）",
          lambda c: list(c.agent.persona.quiet_hours or []),
          lambda c, v: c.agent.apply_settings({"quiet_hours": v})),
)
_reg(
    _bridge_param("chime", "group_chime", "bool", "群插嘴开关"),
    _bridge_param("chimed", "group_chime_daily_max", "int", "每群每天插嘴上限",
                  lo=0, hi=100),
    _bridge_param("chimecd", "group_chime_cooldown_seconds", "float",
                  "插嘴冷却（秒）", lo=0, hi=3600),
    _bridge_param("afterreply", "group_chime_after_reply_seconds", "float",
                  "刚说过话后的静默（秒）", lo=0, hi=600),
    _bridge_param("mention", "group_mention_only", "bool", "群里仅@响应"),
    _bridge_param("grouppro", "groups_proactive", "bool", "群主动搭话"),
    _bridge_param("voice", "voice_reply", "enum", "语音回复 mirror/always/never",
                  options=("mirror", "always", "never")),
    Param("chat", "bridge", "chat", "会话开关（/chat [名字] on|off，/chat list 看全部）",
          _chat_enabled, _set_chat, field="disabled_chats"),
)


# ---------------- 解析 ----------------

CMDS = ("help", "status", "set", "get", "on", "off", "quiet", "model",
        "chat", "audit", "undo")

ALIASES = {"帮助": "help", "状态": "status", "改": "set", "设置": "set",
           "看": "get", "开": "on", "关": "off", "撤销": "undo",
           "记录": "audit", "命令": "help"}


def parse(text):
    """"/set th2 0.3" -> ("set", ["th2", "0.3"])；不认识的一律返回 None。

    只认白名单里的命令名：普通聊天里出现的 "1/2"、"and/or"、路径都不算命令。
    """
    t = str(text or "").strip()
    if not t:
        return None
    t = t.replace("／", "/").replace("\u3000", " ")
    sep5 = chr(0x2005)
    while t.startswith("@"):
        cut = t.find(sep5)
        if 1 <= cut <= 32:
            t = t[cut + 1:].lstrip(sep5 + " ")
        elif 1 <= t.find(" ") <= 32:
            t = t[t.find(" ") + 1:].lstrip()
        else:
            return None
    if not t.startswith("/"):
        return None
    parts = t[1:].split()
    if not parts:
        return None
    name = parts[0].lower()
    name = ALIASES.get(name, name)
    if name not in CMDS:
        return None
    return name, parts[1:]


# ---------------- 审计 ----------------

def _audit(ctx, p: Param, old, new, source: str = "chat", room: str = None):
    a = ctx.agent
    if a is None or not hasattr(a, "conn"):
        return
    try:
        with a._lock:
            a.conn.execute(
                "INSERT INTO config_audit(role_id, room, scope, name, old_value,"
                " new_value, source, ok, created_at) VALUES(?,?,?,?,?,?,?,1,?)",
                (a.role_id, str(room if room is not None else ctx.peer), p.scope, p.name,
                 json.dumps(old, ensure_ascii=False),
                 json.dumps(new, ensure_ascii=False), source, ctx.now))
            a.conn.commit()
    except Exception:  # noqa: BLE001  审计失败不能挡住改配置
        pass


def _last_audit(ctx):
    a = ctx.agent
    if a is None or not hasattr(a, "conn"):
        return None
    with a._lock:
        return a.conn.execute(
            "SELECT * FROM config_audit WHERE role_id=? AND ok=1"
            " ORDER BY id DESC LIMIT 1", (a.role_id,)).fetchone()


# ---------------- 各命令 ----------------

def _need_agent(ctx):
    if ctx.agent is None:
        raise CmdError("会话级参数要在多用户模式下改（当前单用户，去控制台或改 config.yaml）")


def _param(name: str) -> Param:
    p = _BY_NAME.get(str(name or "").strip().lower())
    if p is None:
        raise CmdError(f"没有 {name} 这个项，/help 看清单")
    return p


def _apply(ctx, p: Param, raw):
    if not p.group_ok and ctx.is_group:
        raise CmdError(f"{p.name} 在私聊里改")
    if p.scope == "agent":
        _need_agent(ctx)
    old = p.getter(ctx)
    value = _coerce(p, raw)
    p.setter(ctx, value)
    new = p.getter(ctx)
    _audit(ctx, p, old, new)
    return f"{p.name} {_fmt(old)} → {_fmt(new)} ✓"


def _do_set(ctx, args):
    if len(args) < 2:
        raise CmdError("用法：/set <项> <值>，例：/set th2 0.3")
    return _apply(ctx, _param(args[0]), " ".join(args[1:]))


def _do_switch(ctx, args, value: bool):
    if len(args) != 1:
        raise CmdError(f"用法：/{'on' if value else 'off'} <开关>，例：/off chime")
    p = _param(args[0])
    if p.kind != "bool":
        raise CmdError(f"{p.name} 不是开关，用 /set {p.name} <值>")
    return _apply(ctx, p, value)


def _do_get(ctx, args):
    if len(args) != 1:
        raise CmdError("用法：/get <项>")
    p = _param(args[0])
    if p.scope == "agent":
        _need_agent(ctx)
    return f"{p.name} = {_fmt(p.getter(ctx))}（{p.help}）"


def _do_quit(ctx, args):
    if args:
        return _apply(ctx, _BY_NAME["quiet"], " ".join(args))
    qh = list(ctx.agent.persona.quiet_hours or []) if ctx.agent is not None else []
    return f"安静时段：{_fmt(qh)}（/quiet 23:00-08:00 或 /quiet off）"


def _do_model(ctx, args):
    p = _BY_NAME["model"]
    if not args:
        _need_agent(ctx)
        return f"模型：{p.getter(ctx)}（/model <名字> 换）"
    return _apply(ctx, p, " ".join(args))


def _sub_ctx(ctx, peer):
    """把上下文换到另一个会话（/chat 群名 on 用）。agent 不变——开关是桥级的。"""
    peer = str(peer)
    return Ctx(ctx.agent, ctx.bridge, peer, is_group=peer.endswith("@chatroom"),
               is_owner=True, now=ctx.now)


def _chat_label(ctx, peer):
    peer = str(peer)
    if peer == ctx.peer:
        return "本会话"
    fn = getattr(ctx.bridge, "chat_label", None)
    if fn is not None:
        try:
            name = fn(peer)
            if name:
                return str(name)
        except Exception:  # noqa: BLE001
            pass
    return peer


def _resolve_target(ctx, token):
    """"AI技术冲了" / 房间号 -> room id。桥提供 resolve_chat 时按群名/昵称找。"""
    fn = getattr(ctx.bridge, "resolve_chat", None)
    if fn is not None:
        try:
            wid = fn(token)
        except Exception:  # noqa: BLE001
            wid = None
        if wid:
            return str(wid)
    t = str(token or "").strip()
    if t.endswith("@chatroom") or t.startswith("wxid_"):
        return t
    return None


def _do_chat(ctx, args):
    """开关会话。不带名字=当前会话；带名字=指定会话（在私聊里也能开/关某个群）。"""
    p = _BY_NAME["chat"]
    if not args:
        return f"本会话：{'已启用' if p.getter(ctx) else '已停用'}（/chat on|off）"
    if len(args) == 1 and args[0].lower() in ("list", "列表", "all"):
        return _do_chat_list(ctx)
    if len(args) == 1:
        sub = ctx
        value = _coerce(p, args[0])
    else:
        target = _resolve_target(ctx, args[0])
        if not target:
            raise CmdError(f"没找到「{args[0]}」这个会话；/chat list 看已知的，"
                           f"或直接用房间号")
        sub = _sub_ctx(ctx, target)
        value = _coerce(p, " ".join(args[1:]))
    old = p.getter(sub)
    p.setter(sub, value)
    new = p.getter(sub)
    _audit(ctx, p, old, new, room=sub.peer)
    label = _chat_label(ctx, sub.peer)
    if old == new:
        return f"{label} 已经{'开' if new else '关'}了"
    return f"{label} {'开' if new else '关'} ✓"


def _do_chat_list(ctx):
    peers = []
    pool = getattr(ctx.bridge, "pool", None)
    if pool is not None:
        peers = [str(x) for x in getattr(pool, "agents", {}) or {}]
    if ctx.peer not in peers:
        peers.insert(0, ctx.peer)
    if not peers:
        return f"本会话：{'已启用' if _BY_NAME['chat'].getter(ctx) else '已停用'}"
    parts = []
    for wid in peers[:12]:
        sub = _sub_ctx(ctx, wid)
        parts.append(f"{_chat_label(ctx, wid)} "
                     f"{'开' if _BY_NAME['chat'].getter(sub) else '关'}")
    return "会话：" + " · ".join(parts)


def _do_help(ctx, args):
    if args:
        p = _param(args[0])
        cur = ""
        if p.scope == "agent" and ctx.agent is None:
            cur = "（单用户模式下不可读）"
        else:
            cur = f"，当前 {_fmt(p.getter(ctx))}"
        return f"{p.name}：{p.help}{cur}"
    return (
        "命令（主人专属，群聊私聊都能用）\n"
        "/status 当前状态\n"
        "/set <项> <值> · /on|/off <开关>\n"
        "/get <项> · /audit · /undo · /help <项>\n"
        "常用：th1/th2/th3 阈值0-1 · tick 秒 · roast off|mild|on\n"
        "review off|rule|llm · model <名>（私聊）· l1/l2/l3 主动等级\n"
        "chime on|off · chimed 日上限 · chimecd 冷却秒 · afterreply 秒\n"
        "mention 仅@ · grouppro 群主动 · voice mirror|always|never\n"
        "chat [名字] on|off · /chat list · quiet 23:00-08:00|off\n"
        "全部项：" + " ".join(p.name for p in PARAMS))


def _do_status(ctx):
    w = ctx.bridge.cfg.wechat
    lines = []
    a = ctx.agent
    if a is not None:
        lv = " ".join(f"L{i}{'✓' if _level_on(a, f'L{i}') else '✗'}" for i in (1, 2, 3))
        lines.append(f"模型 {a.config.llm.model} · 感知 {'开' if a.config.perception.enabled else '关'}")
        lines.append(f"主动 {'开' if get_setting(a.conn, 'proactive_enabled', True) else '关'}"
                     f"（{lv}） · tick {int(a.config.tick.fast_seconds)}s")
        th = a.config.gate.thresholds
        lines.append(f"阈值 {_fmt(th.get('L1'))}/{_fmt(th.get('L2'))}/{_fmt(th.get('L3'))}"
                     f" · roast {a.persona.style.get('roast_level') or 'off'}")
        lines.append(f"安静 {_fmt(list(a.persona.quiet_hours or []))}")
    else:
        lines.append("会话级参数：单用户模式不生效")
    lines.append(f"插嘴 {'开' if getattr(w, 'group_chime', True) else '关'}"
                 f"（冷却{int(getattr(w, 'group_chime_cooldown_seconds', 0))}s/"
                 f"日{int(getattr(w, 'group_chime_daily_max', 0))}）"
                 f" · 仅@ {'开' if getattr(w, 'group_mention_only', True) else '关'}")
    lines.append(f"群主动 {'开' if getattr(w, 'groups_proactive', True) else '关'}"
                 f" · 语音 {getattr(w, 'voice_reply', 'mirror')}")
    lines.append(f"本会话 {'已启用' if _chat_enabled(ctx) else '已停用'}")
    return "\n".join(lines)


def _do_audit(ctx, args):
    _need_agent(ctx)
    try:
        n = max(1, min(20, int(args[0]))) if args else 5
    except (TypeError, ValueError):
        n = 5
    with ctx.agent._lock:
        rows = ctx.agent.conn.execute(
            "SELECT * FROM config_audit WHERE role_id=? ORDER BY id DESC LIMIT ?",
            (ctx.agent.role_id, n)).fetchall()
    if not rows:
        return "还没有改过参数"
    out = []
    for r in rows:
        try:
            old = json.loads(r["old_value"]) if r["old_value"] is not None else None
            new = json.loads(r["new_value"]) if r["new_value"] is not None else None
        except (TypeError, ValueError):
            old = new = None
        ts = time.strftime("%m-%d %H:%M", time.localtime(float(r["created_at"] or 0)))
        mark = "↩" if (r["source"] or "") == "undo" else ""
        out.append(f"{ts} {r['name']} {_fmt(old)}→{_fmt(new)}{mark}")
    return "\n".join(out)


def _do_undo(ctx, args):
    _need_agent(ctx)
    row = _last_audit(ctx)
    if row is None:
        return "没有可撤销的改动"
    p = _BY_NAME.get(row["name"])
    if p is None:
        raise CmdError(f"{row['name']} 这个项已经不存在了")
    try:
        old = json.loads(row["old_value"]) if row["old_value"] is not None else None
    except (TypeError, ValueError):
        old = None
    # chat 的作用对象是审计里记的 room（可能是 /chat 群名 指到的另一个会话）
    sub = ctx
    if p.name == "chat" and row["room"]:
        sub = _sub_ctx(ctx, row["room"])
    cur = p.getter(sub)
    p.setter(sub, _coerce(p, old))
    _audit(ctx, p, cur, p.getter(sub), source="undo", room=sub.peer)
    return f"已撤销：{_chat_label(ctx, sub.peer) if p.name == 'chat' else p.name}" \
           f" {_fmt(cur)} → {_fmt(p.getter(sub))}"


def _run(cmd, args, ctx):
    if cmd == "help":
        return _do_help(ctx, args)
    if cmd == "status":
        return _do_status(ctx)
    if cmd == "set":
        return _do_set(ctx, args)
    if cmd == "get":
        return _do_get(ctx, args)
    if cmd == "on":
        return _do_switch(ctx, args, True)
    if cmd == "off":
        return _do_switch(ctx, args, False)
    if cmd == "quiet":
        return _do_quit(ctx, args)
    if cmd == "model":
        return _do_model(ctx, args)
    if cmd == "chat":
        return _do_chat(ctx, args)
    if cmd == "audit":
        return _do_audit(ctx, args)
    if cmd == "undo":
        return _do_undo(ctx, args)
    return None


def handle(text, ctx) -> str:
    """执行一条命令并返回回执；不是命令返回空串（调用方据此决定是否继续走聊天）。"""
    parsed = parse(text)
    if parsed is None:
        return ""
    if not ctx.is_owner:
        return ""
    cmd, args = parsed
    try:
        return _run(cmd, args, ctx) or ""
    except CmdError as e:
        return f"没改成：{e}"
    except ValueError as e:      # apply_* 的校验错误已经是人话
        return f"没改成：{e}"
    except Exception as e:       # noqa: BLE001  命令失败不能拖垮桥
        return f"命令出错：{type(e).__name__}: {e}"


# ---------------- 桥级覆盖的启动恢复 ----------------

def apply_bridge_overrides(cfg, overrides) -> list:
    """把上次命令改过的桥级参数写回 cfg（config.yaml 仍是默认值）。"""
    applied = []
    if not isinstance(overrides, dict):
        return applied
    for p in PARAMS:
        if p.scope != "bridge" or p.kind == "chat" or not p.field:
            continue
        if p.field not in overrides:
            continue
        try:
            value = _coerce(p, overrides[p.field])
        except CmdError:
            continue
        setattr(cfg.wechat, p.field, value)
        applied.append(p.name)
    for field in ("disabled_chats", "enabled_chats"):
        if isinstance(overrides.get(field), list):
            setattr(cfg.wechat, field, [str(x) for x in overrides[field]])
            applied.append(field)
    return applied
