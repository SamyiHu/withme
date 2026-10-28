"""config.yaml / persona.yaml 的加载、默认值合并与校验。

设计：文件 -> dict（yamlite）-> from_dict（校验+默认值）-> dataclass。
校验失败一次性收集全部错误再抛 ConfigError，方便一次改对。
"""
import re
from dataclasses import dataclass, field

from .state_dims import DIMS, RANGES
from . import yamlite

LEVELS = ["L0", "L1", "L2", "L3"]


class ConfigError(Exception):
    pass


# ---------------- config.yaml ----------------

@dataclass
class LLMConfig:
    provider: str = "mock"          # mock | openai_compatible
    api_key_env: str = "LLM_API_KEY"
    base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    model: str = "glm-4-flash"
    temperature_surface: float = 0.8
    timeout_seconds: int = 60
    max_tokens: int = 16384  # 思考型模型思考链吃预算，直接给满（帽子不计费）
    vision_model: str = ""  # 视觉模型（如 cn:glm-4.6v）；空 = 不启用看图
    # 某些网关/桥接会丢弃 system 角色消息：置 true 时把整段设定并入 user 消息
    prompt_in_user: bool = False


@dataclass
class TickConfig:
    fast_seconds: int = 60
    medium_seconds: int = 3600
    slow_seconds: int = 86400


@dataclass
class ChaosConfig:
    epsilon_base: float = 0.05
    seed_tau: int = 300
    noise_smoothing: float = 0.8


@dataclass
class ConvergenceConfig:
    alpha: float = 0.15
    max_pull: float = 0.2
    distance_function: str = "mad"  # mad | rms
    decay_tau_seconds: float = 1800.0


@dataclass
class MotivationConfig:
    weights: dict = field(default_factory=lambda: {
        "social": 0.35, "curiosity": 0.15, "care": 0.25, "expression": 0.15, "disturb": 0.10})
    loneliness_rate_per_hour: float = 0.04
    loneliness_cap: float = 0.95


@dataclass
class GateConfig:
    thresholds: dict = field(default_factory=lambda: {"L1": 0.12, "L2": 0.32, "L3": 0.55})
    scene_whitelist: list = field(default_factory=lambda: ["chat", "idle"])


@dataclass
class MemoryConfig:
    provider: str = "sqlite"   # sqlite | mem0（mem0 需 pip install mem0ai fastembed）
    retrieval_top_k: int = 5
    min_salience: float = 0.3
    # 检索排序的时间衰减半衰期（天）：未被触碰的记忆权重减半，下限 0.35；0=关闭
    salience_halflife_days: float = 14.0
    # mem0 后端的攒批提炼：候选消息攒够 batch_min_items 且距上次 >= batch_min_minutes
    # 才做一次 LLM 提炼（比逐条 infer 省 token，prompt 带已有记忆防重复）；
    # 连续失败 batch_max_fail 次自动停用（maybe_extract(force=True) 可越过）。
    batch_extract: bool = True
    batch_min_items: int = 8
    batch_min_minutes: float = 10.0
    batch_max_fail: int = 2


@dataclass
class DialogueConfig:
    """对话上下文压缩（滚动摘要）：老消息折叠成摘要，近窗保留原文。

    原则 C 只约束"不塞完整聊天记录"，本段决定塞多少、超了怎么办：
    超出 window 的消息不直接丢，攒到 compress_trigger 条时后台压成一段
    滚动摘要（合并旧摘要），与近窗原文一起进 prompt。"""
    window: int = 16              # 近窗原文条数（prompt 最近对话段）
    compress_trigger: int = 30    # 未压缩消息积压达到该条数触发压缩
    summary_max_chars: int = 900  # 摘要长度上限（超出由提炼提示词自行控制，这里硬截）


@dataclass
class PerceptionConfig:
    """洞察层：让 LLM 对每条用户消息做轻量意图/情绪分析，喂给外部指标。"""
    enabled: bool = True
    max_tokens: int = 16384  # 思考型模型的思考链吃输出预算，直接给满
    temperature: float = 0.3


@dataclass
class ReflectionConfig:
    """反思层（成长核心）：定期把经历沉淀为洞察，并在有界预算内微调人格。"""
    enabled: bool = True
    min_interval_hours: float = 12.0   # 两次反思的最小间隔
    min_new_messages: int = 6          # 触发反思需要的最小新消息数（证据门槛）
    max_drift: float = 0.15            # 吸引子中心相对角色卡的总漂移上限
    drift_step: float = 0.05           # 单次反思可调整的步长上限
    max_extra_rules: int = 2           # 聊天习惯在角色卡之外最多新增条数
    max_insights: int = 3              # 单次最多沉淀的洞察条数


@dataclass
class CostConfig:
    """成本护栏：日 token 预算，超出自动降级（关反思/洞察/主动附加，核心聊天不降）。"""
    daily_token_budget: int = 600_000   # 0 = 不设限


@dataclass
class StateStyleConfig:
    """状态驱动语言（机制闭环）：极端状态产出可机检硬约束，渲染后校验。"""
    enabled: bool = True


@dataclass
class HandsConfig:
    """手部（行动层）：感知提议动作，引擎校验执行，结果回填回复。"""
    enabled: bool = True
    notify_desktop: bool = True        # 到点提醒弹 Windows 通知（失败降级为日志）
    allow_shell: bool = False          # shell.run 默认关闭（有风险，需显式开启）
    plugins_enabled: bool = True       # 加载 plugins/ 目录的社区动作插件
    web_max_bytes: int = 200_000       # web.fetch 抓取上限
    web_timeout: int = 15


@dataclass
class ASRConfig:
    """语音识别：任何 OpenAI 兼容 /audio/transcriptions 端点（默认智谱 GLM-ASR）。"""
    base_url: str = ""            # 空 = 复用 llm.base_url
    api_key_env: str = "LLM_API_KEY"
    model: str = "glm-asr-2512"
    timeout_seconds: int = 60


@dataclass
class ReviewerConfig:
    """审核 Agent（双 Agent 模式的第二个人，前额叶）：只判断、不说话。

    mode:
      off  - 关闭（只剩互动 Agent 一条路）
      rule - 只有确定性规则（能力否认/AI 自曝/客服腔/AI 腔），零额外成本
      llm  - 规则快路径 + 独立 LLM 审稿员终审（真双 Agent，默认）
    llm_purposes 里的用途每条都请审稿员；其余用途只在规则抓到问题时终审。
    审稿调用失败不阻断发送（宁可发出去，也不能因为审核挂了失联）。
    """
    mode: str = "llm"
    llm_purposes: list = field(default_factory=lambda: ["respond", "proactive", "chime"])
    risk_triggers: bool = True       # 规则命中时也请审稿员终审
    max_llm_per_day: int = 400       # 审稿 LLM 调用日上限（0=不限），防预算被审稿烧穿
    max_tokens: int = 260            # 审稿只要一个短 JSON，别让它写作文
    temperature: float = 0.1
    timeout_seconds: float = 20.0    # 审稿超时：宁可放行也不卡住回复


@dataclass
class SearchConfig:
    """联网搜索（web.search 手部）：智谱 Web Search Pro。
    实时信息/新梗/事实核查走这里；与 ASR 共用一把智谱 key。"""
    enabled: bool = True
    base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    api_key_env: str = "BIGMODEL_API_KEY"
    model: str = "web-search-pro"
    timeout_seconds: int = 20
    max_results: int = 4
    chain: list = field(default_factory=lambda: ["bigmodel", "cush"])
    cush_exe: str = ""  # 可选：Cush 万能搜（MCP）可执行文件路径；空=跳过该兜底源
    cush_timeout_seconds: float = 50.0


@dataclass
class ImageGenConfig:
    """画图（图片发送能力之一）：CogView 或任何 OpenAI 兼容 /images/generations 端点。"""
    enabled: bool = True
    base_url: str = ""            # 空 = 复用 llm.base_url
    api_key_env: str = "LLM_API_KEY"
    model: str = "cogview-3-flash"
    size: str = "1024x1024"
    timeout_seconds: int = 60


@dataclass
class WxhookConfig:
    """wxhook 通道（github.com/mengyunqwq/wechat-hook，微信 4.1.10.27，version.dll 代理）。
    该通道支持真语音气泡（/SendVoiceMsg SILK bytes 模式）。"""
    base_url: str = "http://127.0.0.1:30001"   # hook 的本地 HTTP 服务
    notify_port: int = 18998                    # hook 推送新消息通知的本地端口
    data_root: str = ""                         # 微信数据目录（msg/attach 所在）；空=由 PRAGMA 推断
    fallback_poll_seconds: float = 60.0         # 兜底轮询间隔（0=关）。hook 的 SQL 不带连接锁，
                                                # 高频查库会与客户端写库竞争（2026-10-27 实测致库损坏），
                                                # 收消息以通知推送为主，轮询只是兜底，默认 60s


@dataclass
class WechatConfig:
    """微信桥（src/wechat_bridge.py）：把 REST API 接到个人微信。
    桥是独立进程，本段只是它的参数；server 未启动时桥会一直重试连接。
    channel：wcf（wcferry/微信3.9.12.51，语音以MP3文件卡片发出）
           | wxhook（wechat-hook/微信4.1.10.27，真语音气泡）"""
    channel: str = "wcf"
    multi_user: bool = False      # 多用户模式：每个私聊用户一份独立小满（独立库），桥内置调度
    allowed_wxids: list = field(default_factory=list)  # 空=不限（所有私聊都响应）
    max_users: int = 20           # 并发用户上限（防陌生人刷爆内存/Token）
    server_url: str = "http://127.0.0.1:8787"
    api_token_env: str = "AGENT_API_TOKEN"
    owner_wxid: str = ""          # 空 = 第一个私聊的人自动绑定为唯一主人并持久化
    poll_seconds: float = 2.0     # /pending 增量轮询间隔（回复与主动消息共用此通道）
    voice_reply: str = "mirror"   # mirror（你发语音她回语音）| always | never
    voice_max_chars: int = 220    # 回复超过该长度只发文字（微信语音上限 60s 的保守值）
    batch_quiet_seconds: float = 4.0  # 连发合并静默窗：同一个人最后一条后静默 N 秒才算说完
    groups_enabled: bool = False      # 群聊响应（wxhook 通道；仅被@或主人发言时回，免刷屏）
    group_mention_only: bool = True   # 群里只响应"@小满"（atuserlist 命中自己）与主人；False=所有消息都回
    groups_proactive: bool = True     # 群主动搭话：频率/时机由引擎闸门把关（冷却/预算/免打扰），发送时 @ 最近说话人
    group_chime: bool = True          # 群智能插嘴：群友聊天攒够几条后，AI 自己判断要不要接一句（未被@时）
    group_chime_cooldown_seconds: float = 300.0  # 两次插嘴最小间隔（秒/群）
    group_chime_daily_max: int = 12   # 每群每天最多自发插嘴次数（防刷屏硬顶）
    group_self_aliases: list = field(default_factory=list)  # 群里没@但这样叫也算叫它（群昵称/人设名会自动识别）
    media_dir: str = "data/wechat_media"   # 相对仓库根；语音/图片中转目录
    wxhook: WxhookConfig = field(default_factory=WxhookConfig)


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    tick: TickConfig = field(default_factory=TickConfig)
    chaos: ChaosConfig = field(default_factory=ChaosConfig)
    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)
    motivation: MotivationConfig = field(default_factory=MotivationConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    dialogue: DialogueConfig = field(default_factory=DialogueConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    reflection: ReflectionConfig = field(default_factory=ReflectionConfig)
    hands: HandsConfig = field(default_factory=HandsConfig)
    state_style: StateStyleConfig = field(default_factory=StateStyleConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    reviewer: ReviewerConfig = field(default_factory=ReviewerConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    web_search: SearchConfig = field(default_factory=SearchConfig)
    image_gen: ImageGenConfig = field(default_factory=ImageGenConfig)
    wechat: WechatConfig = field(default_factory=WechatConfig)


# ---------------- persona.yaml ----------------

@dataclass
class AttractorEntry:
    state_name: str
    center: float
    tolerance: float
    weight: float


@dataclass
class Persona:
    role_id: str
    identity_summary: str
    core_invariants: list
    attractor: list          # [AttractorEntry]
    baseline_state: dict
    style: dict              # tone / catchphrases / forbidden / keywords
    boundaries: list
    active_permissions: dict # {"L0": bool, ...}
    cooldown_seconds: dict   # {"L1": 300, ...}
    daily_budget: dict       # {"L1": 20, ...}
    quiet_hours: list        # ["23:00-08:00"]
    creator: dict = field(default_factory=dict)  # 创造者关系（不可被反思/群聊改写）

    def attractor_map(self) -> dict:
        return {e.state_name: e for e in self.attractor}


# ---------------- 加载 ----------------

def load_config(path: str) -> AppConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yamlite.loads(f.read())
    return config_from_dict(data)


def load_persona(path: str) -> Persona:
    with open(path, "r", encoding="utf-8") as f:
        data = yamlite.loads(f.read())
    return persona_from_dict(data)


def config_from_dict(data: dict) -> AppConfig:
    if not isinstance(data, dict):
        raise ConfigError("config 根节点必须是映射")
    errors = []
    cfg = AppConfig()
    try:
        _fill_dataclass(cfg.llm, data.get("llm", {}), errors, "llm")
        _fill_dataclass(cfg.tick, data.get("tick", {}), errors, "tick")
        _fill_dataclass(cfg.chaos, data.get("chaos", {}), errors, "chaos")
        _fill_dataclass(cfg.convergence, data.get("convergence", {}), errors, "convergence")
        _fill_dataclass(cfg.gate, data.get("gate", {}), errors, "gate")
        _fill_dataclass(cfg.memory, data.get("memory", {}), errors, "memory")
        _fill_dataclass(cfg.dialogue, data.get("dialogue", {}), errors, "dialogue")
        _fill_dataclass(cfg.perception, data.get("perception", {}), errors, "perception")
        _fill_dataclass(cfg.reflection, data.get("reflection", {}), errors, "reflection")
        _fill_dataclass(cfg.hands, data.get("hands", {}), errors, "hands")
        _fill_dataclass(cfg.state_style, data.get("state_style", {}), errors, "state_style")
        _fill_dataclass(cfg.cost, data.get("cost", {}), errors, "cost")
        _fill_dataclass(cfg.reviewer, data.get("reviewer", {}), errors, "reviewer")
        _fill_dataclass(cfg.asr, data.get("asr", {}), errors, "asr")
        _fill_dataclass(cfg.web_search, data.get("web_search", {}), errors, "web_search")
        _fill_dataclass(cfg.image_gen, data.get("image_gen", {}), errors, "image_gen")
        _fill_dataclass(cfg.wechat, {k: v for k, v in data.get("wechat", {}).items()
                                     if k != "wxhook"}, errors, "wechat")
        _fill_dataclass(cfg.wechat.wxhook, (data.get("wechat", {}) or {}).get("wxhook", {}),
                        errors, "wechat.wxhook")
        m = data.get("motivation", {})
        _fill_dataclass(cfg.motivation, {k: v for k, v in m.items() if k != "weights"}, errors, "motivation")
        if "weights" in m:
            cfg.motivation.weights = dict(m["weights"])
    except ConfigError:
        raise
    # 语义校验
    if cfg.llm.provider not in ("mock", "openai_compatible"):
        errors.append(f"llm.provider 必须是 mock 或 openai_compatible，当前 {cfg.llm.provider}")
    if cfg.reviewer.mode not in ("off", "rule", "llm"):
        errors.append("reviewer.mode 只支持 off / rule / llm，当前 " + str(cfg.reviewer.mode))
    if cfg.convergence.distance_function not in ("mad", "rms"):
        errors.append("convergence.distance_function 只支持 mad / rms")
    w = cfg.motivation.weights
    for k in ("social", "curiosity", "care", "expression", "disturb"):
        if k not in w:
            errors.append(f"motivation.weights 缺少 {k}")
    if not all(0.0 <= float(v) <= 1.0 for v in w.values()):
        errors.append("motivation.weights 各权重须在 [0,1]")
    if cfg.chaos.epsilon_base <= 0 or cfg.chaos.noise_smoothing < 0 or cfg.chaos.noise_smoothing > 1:
        errors.append("chaos.epsilon_base 须 >0；noise_smoothing 须在 [0,1]")
    if cfg.convergence.alpha <= 0 or cfg.convergence.max_pull <= 0:
        errors.append("convergence.alpha / max_pull 须 >0")
    if cfg.memory.retrieval_top_k < 1:
        errors.append("memory.retrieval_top_k 须 >=1")
    if cfg.memory.salience_halflife_days < 0:
        errors.append("memory.salience_halflife_days 须 >=0（0=关闭衰减）")
    if cfg.memory.batch_min_items < 2 or cfg.memory.batch_min_minutes < 1:
        errors.append("memory.batch_min_items 须 >=2；batch_min_minutes 须 >=1")
    if cfg.memory.batch_max_fail < 1:
        errors.append("memory.batch_max_fail 须 >=1")
    if not (6 <= cfg.dialogue.window <= 60):
        errors.append("dialogue.window 须在 [6, 60]")
    if not (cfg.dialogue.window + 4 <= cfg.dialogue.compress_trigger <= 200):
        errors.append("dialogue.compress_trigger 须 >= window+4 且 <=200")
    if not (200 <= cfg.dialogue.summary_max_chars <= 4000):
        errors.append("dialogue.summary_max_chars 须在 [200, 4000]")
    if cfg.wechat.voice_reply not in ("mirror", "always", "never"):
        errors.append("wechat.voice_reply 只支持 mirror / always / never")
    if cfg.wechat.channel not in ("wcf", "wxhook"):
        errors.append("wechat.channel 只支持 wcf / wxhook")
    if not (1 <= cfg.wechat.wxhook.notify_port <= 65535):
        errors.append("wechat.wxhook.notify_port 须在 [1, 65535]")
    if not (0 <= cfg.wechat.wxhook.fallback_poll_seconds <= 3600):
        errors.append("wechat.wxhook.fallback_poll_seconds 须在 [0, 3600]")
    if not all(isinstance(x, str) and x for x in cfg.wechat.allowed_wxids):
        errors.append("wechat.allowed_wxids 须为非空字符串列表")
    if not (1 <= cfg.wechat.max_users <= 200):
        errors.append("wechat.max_users 须在 [1, 200]")
    if cfg.wechat.poll_seconds <= 0 or cfg.wechat.poll_seconds > 60:
        errors.append("wechat.poll_seconds 须在 (0, 60]")
    if not (1 <= cfg.wechat.voice_max_chars <= 1000):
        errors.append("wechat.voice_max_chars 须在 [1, 1000]")
    if not (cfg.wechat.batch_quiet_seconds >= 0.5 and cfg.wechat.batch_quiet_seconds <= 15):
        errors.append("wechat.batch_quiet_seconds 须在 [0.5, 15]")
    if not (60 <= cfg.wechat.group_chime_cooldown_seconds <= 86400):
        errors.append("wechat.group_chime_cooldown_seconds 须在 [60, 86400]")
    if not (1 <= cfg.wechat.group_chime_daily_max <= 50):
        errors.append("wechat.group_chime_daily_max 须在 [1, 50]")
    if not (cfg.wechat.media_dir == "" or re.match(r"^[A-Za-z0-9_./\\-]+$", cfg.wechat.media_dir)):
        errors.append("wechat.media_dir 含非法字符")
    _raise(errors)
    return cfg


def persona_from_dict(data: dict) -> Persona:
    if not isinstance(data, dict):
        raise ConfigError("persona 根节点必须是映射")
    errors = []
    required = ["role_id", "identity_summary", "core_invariants", "attractor",
                "baseline_state", "style", "boundaries", "active_permissions",
                "cooldown_seconds", "daily_budget", "quiet_hours"]
    for k in required:
        if k not in data:
            errors.append(f"persona 缺少字段 {k}")
    _raise(errors)

    identity = data["identity_summary"]
    if not isinstance(identity, str) or not identity:
        errors.append("identity_summary 必须是非空字符串")
    elif len(identity) > 200:
        errors.append(f"identity_summary 超过 200 字（当前 {len(identity)} 字）——动态 Prompt 每轮都带它")

    inv = data["core_invariants"]
    if not (isinstance(inv, list) and 3 <= len(inv) <= 5):
        errors.append("core_invariants 必须是 3-5 条")

    attractor = []
    seen = set()
    for i, e in enumerate(data["attractor"]):
        if not isinstance(e, dict) or not {"state_name", "center", "tolerance", "weight"} <= set(e):
            errors.append(f"attractor[{i}] 缺字段（需 state_name/center/tolerance/weight）")
            continue
        name = e["state_name"]
        if name not in DIMS:
            errors.append(f"attractor[{i}].state_name `{name}` 不在状态维度 {DIMS} 中")
            continue
        if name in seen:
            errors.append(f"attractor[{i}] 重复维度 {name}")
        seen.add(name)
        lo, hi = RANGES[name]
        center = e["center"]
        if not isinstance(center, (int, float)) or not (lo <= center <= hi):
            errors.append(f"attractor[{name}].center={center} 超出取值范围 [{lo},{hi}]")
        if not isinstance(e["tolerance"], (int, float)) or e["tolerance"] <= 0:
            errors.append(f"attractor[{name}].tolerance 必须 >0")
        attractor.append(AttractorEntry(name, float(center), float(e["tolerance"]), float(e["weight"])))
    for d in DIMS:
        if d not in seen:
            errors.append(f"attractor 缺少维度 {d}")

    baseline = data["baseline_state"]
    if not isinstance(baseline, dict) or set(baseline) != set(DIMS):
        errors.append(f"baseline_state 必须恰好包含全部维度 {DIMS}")
    else:
        for d in DIMS:
            lo, hi = RANGES[d]
            if not isinstance(baseline[d], (int, float)) or not (lo <= baseline[d] <= hi):
                errors.append(f"baseline_state.{d}={baseline[d]} 超出 [{lo},{hi}]")

    style = data["style"]
    if not isinstance(style, dict) or "catchphrases" not in style:
        errors.append("style 必须包含 catchphrases")
    if isinstance(style, dict) and not style.get("keywords"):
        errors.append("style.keywords 不能为空（抗漂移测试依据它评分）")

    perms = data["active_permissions"]
    if not isinstance(perms, dict) or set(perms) < {"L0", "L1", "L2", "L3"}:
        errors.append("active_permissions 必须包含 L0-L3 四级布尔值")
    for lvl in LEVELS:
        if lvl in perms and not isinstance(perms[lvl], bool):
            errors.append(f"active_permissions.{lvl} 必须是布尔值")
    if perms.get("L0") is False:
        errors.append("L0（沉默）不允许关闭")

    for field_name in ("cooldown_seconds", "daily_budget"):
        block = data[field_name]
        if not isinstance(block, dict) or set(block) < {"L1", "L2", "L3"}:
            errors.append(f"{field_name} 必须包含 L1/L2/L3")
    if not isinstance(data["quiet_hours"], list):
        errors.append("quiet_hours 必须是列表")
    for span in data["quiet_hours"]:
        if not re.match(r"^\d{2}:\d{2}-\d{2}:\d{2}$", str(span)):
            errors.append(f"quiet_hours 项 `{span}` 须为 HH:MM-HH:MM")
        else:
            _parse_quiet_span(str(span))  # 抛异常则格式实际不合法

    _raise(errors)
    return Persona(
        role_id=str(data["role_id"]),
        identity_summary=identity,
        core_invariants=[str(x) for x in inv],
        attractor=attractor,
        baseline_state={d: float(baseline[d]) for d in DIMS},
        style=style,
        boundaries=[str(x) for x in data["boundaries"]],
        active_permissions=perms,
        cooldown_seconds={k: int(v) for k, v in data["cooldown_seconds"].items()},
        daily_budget={k: int(v) for k, v in data["daily_budget"].items()},
        quiet_hours=[str(x) for x in data["quiet_hours"]],
        creator=dict(data.get("creator") or {}),
    )


def parse_quiet_hours(spans: list, local_hour_minute: tuple) -> bool:
    """给定 (hour, minute) 判断是否处于免打扰时段。"""
    minutes = local_hour_minute[0] * 60 + local_hour_minute[1]
    for span in spans:
        start, end = _parse_quiet_span(span)
        if start <= end:
            if start <= minutes < end:
                return True
        else:  # 跨午夜，如 23:00-08:00
            if minutes >= start or minutes < end:
                return True
    return False


def _parse_quiet_span(span: str):
    a, b = span.split("-")
    ah, am = (int(x) for x in a.split(":"))
    bh, bm = (int(x) for x in b.split(":"))
    if not (0 <= ah < 24 and 0 <= bh < 24 and 0 <= am < 60 and 0 <= bm < 60):
        raise ConfigError(f"quiet_hours 项 `{span}` 时间不合法")
    return ah * 60 + am, bh * 60 + bm


def _fill_dataclass(target, source: dict, errors, prefix: str):
    for k, v in source.items():
        if not hasattr(target, k):
            errors.append(f"{prefix}.{k} 不是已知配置项")
            continue
        cur = getattr(target, k)
        try:
            if isinstance(cur, bool):
                setattr(target, k, bool(v))
            elif isinstance(cur, int) and not isinstance(cur, bool):
                setattr(target, k, int(v))
            elif isinstance(cur, float):
                setattr(target, k, float(v))
            else:
                setattr(target, k, v)
        except (TypeError, ValueError):
            errors.append(f"{prefix}.{k} 类型不合法：{v!r}")


def _raise(errors):
    if errors:
        raise ConfigError("配置校验失败：\n  - " + "\n  - ".join(errors))
