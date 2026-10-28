# WithMe — API-only 拟人化主动智能体（混沌-收敛人格引擎 + 反思成长）

不训练、不微调任何模型。人格、记忆、状态、主动性、权限、收敛逻辑全部由外部工程实现；
LLM API 只负责"把已选定的意图渲染成自然语言"。混沌负责"活"，收敛负责"是"：

- **混沌**：哈希种子 → 平滑有界噪声，只扰动内部状态层（7 维），不直接决定说什么；
- **收敛**：人格吸引子（角色卡定义中心/容差/权重）+ 每轮动态重建 Prompt，保证长期不漂移；
- **主动**：候选意图（必须挂记忆/意图来源）→ 评分 → L0-L3 闸门 → 才交给 LLM 渲染；
- **成长**：反思层定期把经历沉淀为洞察记忆，并在**有界预算内**微调人格（见下文）。
- **审核（双 Agent）**：互动 Agent 写完一句话，审核 Agent（独立调用 + 独立提示词）出门过审——
  能力否认 / AI 自曝 / 客服腔 / 答非所问 / 群聊站队 / 微信口语度 / 主动消息空转；
  不过审就打回重写一次，主动与插嘴被审死则宁可沉默；每条留痕在 `review_logs`。

零第三方依赖，纯 Python 标准库（3.9+，CI 覆盖 3.9–3.13）。默认 mock 渲染，**无 API key 即可完整运行**。


## 仓库内容说明

本仓库包含**人格化引擎 + 基础 IO + 网页聊天/控制台**：

- `src/`：Agent 引擎、记忆、人物卡、主动闸门、微信桥
- `src/web/`：网页聊天客户端（`/`）与后台控制台（`/console`）
- `config/` `db/` `tests/` `docs/` `plugins/` `deploy/`
- 运行数据（`data/` `logs/`）、密钥（`.env`，模板见 `.env.example`）与本机启动脚本不入库（见 `.gitignore`）

架构与开源范围：[ARCHITECTURE.md](docs/ARCHITECTURE.md) · [OPEN_SOURCE.md](docs/OPEN_SOURCE.md)。
终端客户端：`python tools/chat_client.py`。

## 快速开始

```bash
python -m src.main serve                # ← 日常使用入口：HTTP API + 网页客户端 + 内置调度
#   浏览器打开 http://127.0.0.1:8787/ 即可和小满聊天；主动消息由内置线程按闸门推送

python -m src.main demo                 # 端到端演示：应答 → 8h 后主动提醒 → 拒绝 → 冷却拦截
python -m src.main chat                 # 终端对话（不经 HTTP）
python -m src.main tick                 # 手动调度 tick（不跑 serve 时的 cron 入口）
python -m src.main status               # 状态向量 / 距吸引子距离 / 关系 / 今日主动
python -m src.main set-proactive off    # 关闭所有主动行为（set-level L2 off 只关 L2）
```

调度器是**惰性**的：`serve` 模式内置每 60s 一次的调度线程；不跑 serve 时把
`python -m src.main tick` 挂到系统计划任务/cron 即可，tick 会按 Δt 补算状态。

## 部署与 I/O

### 本机使用

```bash
python -m src.main serve --port 8787          # 默认绑定 127.0.0.1，仅本机可访问
```

浏览器打开 `http://127.0.0.1:8787/` 就是聊天界面（零构建单文件网页，15 秒轮询一次
主动消息）。想当"桌面小组件"用：浏览器装个 PWA/固定标签页，或用任何 webview 壳包一层。

### 服务器部署（Linux + systemd）

```bash
sudo useradd -m withme && sudo mkdir -p /opt/withme
# 把整个仓库拷到 /opt/withme
sudo cp deploy/withme.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now withme
journalctl -u withme -f          # 看日志（主动行为会实时打印）
```

对外暴露：`serve --host 0.0.0.0` + 环境变量 `AGENT_API_TOKEN=随机长串`，前面挂
nginx/caddy 加 HTTPS。令牌启用后所有 API 需带 `Authorization: Bearer <token>`
（网页端右上角填一次即可）。

### 接真实 LLM（部署时不改文件）

```bash
export AGENT_LLM_PROVIDER=openai_compatible
export AGENT_LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4   # 任意 OpenAI 兼容端点
export AGENT_LLM_MODEL=glm-4-flash
export LLM_API_KEY=sk-xxx
# 某些网关会丢弃 system 角色消息（模型收不到角色设定、退化成通用助手），
# 此时加：export AGENT_LLM_PROMPT_IN_USER=1   # 设定整段并入 user 消息
python -m src.main serve
```

### HTTP API 一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 网页聊天客户端 |
| GET | `/health` | 存活检查（含渲染端类型） |
| GET | `/status` | 状态向量 / 关系 / 今日主动 / 开关 |
| GET | `/console` | 后台控制台（参数可视化 + 可调） |
| GET | `/console-data` | 控制台数据（状态/人物卡/记忆/参数/群 Agent） |
| GET | `/trajectory?limit=120` | 会话轨迹：每个会话一根真实时间轴（私聊 + 每个群），消息/模型/工具泳道 + 审核结果 |
| GET | `/messages?limit=50` | 最近消息（双向，含主动，恢复历史用） |
| GET | `/pending?after_id=N` | id > N 的 agent 消息增量轮询（客户端拉主动消息） |
| POST | `/chat` `{"text": …, "images": [{"b64":…,"mime":…}]}` | 发消息（可带图片），返回 `{reply, addon, meta.message_id}` |
| POST | `/tick` | 手动触发一次调度评估 |
| POST | `/config` | 写入运行时参数（tick/chaos/convergence/gate/motivation/dialogue/reflection/reviewer/persona 权限等） |
| POST | `/proactive` `{"enabled": bool}` | 主动总开关 |
| POST | `/level` `{"level":"L2","enabled":bool}` | 分级开关 |

任何插件/脚本/第三方壳（浏览器插件、桌面 widget、微信机器人框架）都能直接调这套
REST API 接入。终端客户端：`python tools/chat_client.py --url http://127.0.0.1:8787`。

并发与线程：HTTP 层多线程收请求，Agent 内部 RLock 串行化全部状态读写（有并发测试覆盖）；
主动消息同时写入 messages 表（带 `meta.proactive` 溯源信息），所以 `/pending`、
`/messages` 和网页端都能看到。

## 微信桥（个人微信接入：文字/图片/语音·双通道）

`python -m src.main wechat` 把小满接到**个人微信**，通道二选一（`wechat.channel`）：

- **wxhook**（默认，微信 4.1.10.27 + [wechat-hook](https://github.com/mengyunqwq/wechat-hook)，
  **预编译 version.dll 开箱即用**）：**真语音气泡**——edge-tts → 24kHz PCM → SILK
  （pysilk-mod/pilk/silk-wasm 三选一，全免编译）→ 调微信自己的语音消息工厂（type 34）。
- **wcf**（备选，微信 3.9.12.51 + wcferry）：语音回复 = TTS MP3 文件卡片
  （3.9 客户端无发送语音功能，接口层面补不了）。

文字双向、图片双向（收图走视觉理解，发图含 CogView 画图）、语音收（GLM-ASR 听写，
复用现有 key）两通道一致；主动消息通过 `/pending` 增量轮询投递，与回复共用水位线去重。

```bash
pip install pysilk-mod zstandard edge-tts imageio-ffmpeg   # wxhook 通道（默认）
# pip install wcferry                                     # 或 wcf 通道
python -m src.main serve                                   # 先起服务端
python -m src.main wechat --probe                          # 按通道连通性自检
python -m src.main wechat                                  # 起桥
```

一次性准备（锁版本微信安装、version.dll 获取、风险说明）、配置项与常见问题见
**[docs/wechat_setup.md](docs/wechat_setup.md)**。相关配置段：`config.yaml`
的 `wechat`（含 `wxhook` 子段）/ `asr` / `image_gen`。

## 接入真实 LLM（改配置文件的方式）

编辑 `config/config.yaml`：

```yaml
llm:
  provider: openai_compatible          # 从 mock 切换
  api_key_env: LLM_API_KEY             # 从这个环境变量读 key
  base_url: "https://open.bigmodel.cn/api/paas/v4"   # 任意 OpenAI 兼容端点
  model: glm-4-flash
```

```bash
export LLM_API_KEY=sk-xxx    # Windows: set LLM_API_KEY=sk-xxx
python -m src.main chat
```

LLM 客户端可替换：`src/llm_client.py` 的 `BaseLLMClient` 只有 `render(prompt, seed)`
一个方法，实现新 provider 后在 `create_client` 注册即可。

## 常驻运行与设置面板

**开机自启（Windows，无需管理员权限）**：

```bash
# 先把当前终端的环境变量备好（桥接地址/模型/key），再执行：
python -m src.main install      # 生成 deploy/withme_env.bat 等启动三件套，
                                # 优先注册计划任务，无权限则回退到用户启动文件夹
python -m src.main uninstall    # 移除自启
```

启动链：登录 → `withme_autostart.vbs`（无窗口）→ `deploy/start_withme.bat`
（加载 `deploy/withme_env.bat` 环境变量）→ `pythonw -m src.main serve`。
**改桥接地址/模型/Key**：编辑 `deploy/withme_env.bat` 或直接用网页设置面板。

**设置面板**：网页右上角 ⚙。运行时热改、立即生效、持久化到 SQLite（重启不丢）：

| 分区 | 可调项 |
|---|---|
| 主动行为 | 总开关、L1/L2/L3 分级开关、免打扰时段 |
| AI 接入 | provider / API 地址 / 模型 / 设定并入 user / API Key |
| 记忆后端 | sqlite ↔ mem0 热切换 |
| 洞察层 | 意图揣摩开关 |
| 人设风格 | 口头禅、语气、聊天习惯（每轮 Prompt 即时生效） |
| 缓存命中 | 各用途 DeepSeek 前缀缓存命中率实时展示 |

对应 API：`GET /settings`、`PUT /settings`（部分更新）。

## DeepSeek 前缀缓存优化

DeepSeek 系模型按**请求前缀**自动命中上下文缓存（命中部分按缓存价计费）。
本项目针对它做了三件事：

1. **Prompt 分节按变化频率升序**：永不变（身份/不变量/风格/输出格式）→
   少变（权限/未完成意图）→ **追加式稳定的最近对话（16 条内只增不改）** →
   每轮必变（关系/记忆/当前状态/任务沉底）。有钉子测试防止顺序回退。
2. **确定性序列化**：同输入必产出逐字节相同的 Prompt 文本。
3. **命中遥测**：`GET /status` 的 `llm_usage` 按用途（respond/perceive/addon/proactive）
   报告 `cache_hit_tokens` 与命中率；设置 `AGENT_DEBUG_PROMPTS=<文件路径>`
   可把每次请求的前缀与 MD5 写日志做比对。

实测（经中转桥）：respond 稳定命中人设前缀 ~384 tokens/次、perceive ~128。
**上限在桥接侧**：相同前缀直连探测显示命中恒为 128 的倍数且封顶 ~384-512，
伴随写入延迟与疑似上游账号轮换——若桥接能把请求粘到单一上游账号
（session affinity），稳定前缀（人设+意图+对话历史 ≈1200+ tokens）可整体命中。

## 路线图完成情况（2026-10-23 全量落地）

| 项 | 状态 | 落点 |
|---|---|---|
| 状态驱动语言闭环 | ✅ | `src/state_style.py` + meta.style 审计 |
| 时间感/自传体记忆 | ✅ | 相关记忆带相对时间（今天/上周/N周前），初识锚点，`known_days`，回调规则带时间感 |
| 场景感知（打字勿扰） | ✅ | `POST /presence {typing}` -> 90 秒内主动评分打扰成本 +0.25 |
| LLM-as-judge 评估 | ✅ | `python tools/judge_eval.py` 三维打分出报告；实测 人味5.00/一致4.80/分寸4.60 |
| 反思调行为参数 | ✅ | 反思可提议 `gate_shift`（L2 阈值 ±0.05/步、±0.15 封顶），随拒绝率自适应，reset 一并回滚 |
| 多步手部任务 | ✅ | 感知可提议动作序列（≤3，按序执行，结果合并审计） |
| 成本护栏 | ✅ | `cost.daily_token_budget`（默认 60 万 token/日），耗尽自动降级（反思/洞察/主动附加），核心聊天不降，次日恢复 |
| VL 看画面 | ✅ | `bilibili.watch` 附封面视觉描述；视觉回落对话模型（deepseek-v4.1-flash 原生看图），`AGENT_VLM_MODEL` 可指定 |
| 语音 TTS | ✅ | `POST /tts {text}` -> mp3（edge-tts，`pip install edge-tts`，`AGENT_TTS_VOICE` 可换声） |
| 语音 ASR | ✅ | `src/asr_client.py`：OpenAI 兼容 `/audio/transcriptions`（默认 GLM-ASR，复用 LLM key），微信桥语音听写用 |
| 图片生成 | ✅ | 手部动作 `image.draw`（CogView-3-Flash，`image_gen.enabled` 可关），产物随回复发送 |
| 消息通道 | ✅ | REST API 即边界，任何 bot 框架直连（对接指南：[docs/api.md](docs/api.md)，`/api/v1` 前缀 + 增量轮询 + 错误契约）；**微信桥**（个人微信：文字/图片/语音，`python -m src.main wechat`，双通道 wcf/wxhook，wxhook 支持真语音气泡，见 [docs/wechat_setup.md](docs/wechat_setup.md)） |
| 多角色 | ✅ | `AGENT_CONFIG_DIR` + `AGENT_DB_PATH` 环境变量，一进程一角色，多实例并行 |

未做：桥接 session affinity（桥接侧配置）。

## 插件系统（社区扩展点）

**动作插件**：往 `plugins/` 丢一个 `.py`，重启即生效——感知层自动看到新动作并学会提议，
执行/审计/结果回填全走现成管线。最小模板只有 `DESC + ACTIONS` 两个变量，
handler 拿 `(params, ctx)`，ctx 提供网络抓取、桌面通知、按插件隔离的 KV 存储，
**拿不到**数据库与 Agent 内部。范例：`plugins/weather.py`（Open-Meteo 免 key 开箱即用）。
坏插件不拖垮主程序；`settings` 表 `plugins_disabled` 可禁用。详见 `plugins/README.md`。

**通道插件**：不需要进进程——REST API 就是边界，任何语言直接调（见 API 一览）。

## 手部（行动层）：让她能做事，而不只是说话

延续闸门哲学：**LLM 永远不直接执行任何东西**。感知层最多"提议"一个动作
（`hand_action`，只能从可用动作清单里选），引擎校验参数与权限后执行，
结果回填回复 Prompt 的【手部动作】节——所以她说"记上了"时是真的记上了。

| 动作 | 说明 |
|---|---|
| `reminder.set/list/cancel` | 定时提醒；中文时间解析（"40分钟后/明天早上9点/晚上8点半/周一8点"）|
| `note.add` | 随手记，入记忆库可检索 |
| `web.fetch` | 抓网页正文（剥离脚本标签、限 200KB、仅 http(s)）——给她"眼睛" |
| `bilibili.watch` | "看"B站视频：元信息+弹幕热词+高赞评论合成观后素材（匿名公开端点，含 b23.tv 短链解析）|
| `bilibili.comments` | 读B站评论区：热评按点赞排序摘录 |
| `desktop.notify` | Windows 桌面通知（无依赖 PowerShell 气泡） |
| `shell.run` | 本地命令；**默认禁用**（`hands.allow_shell: true` 显式开启） |

到点提醒是用户自己约的闹钟：调度 tick 检查 `reminders` 表，**直发不走过评分闸门**
（active_logs 留痕标记 `user_requested`），同时弹桌面通知、走 `/pending` 推送；
渲染失败也保证模板直发，绝不吞提醒。全部执行写 `hands_log` 表可审计。

已知边界（诚实声明）：感知提议依赖 LLM 输出 JSON——推理型模型思考链会吃输出预算，
`perception.max_tokens` 需留足（默认 800）；感知失败回退词表的那一轮不会有手部动作，
模型可能口头答应而未执行。审计：`meta.hand` 为 null 且 `/reminders` 无新行即为
"假执行"，重说一遍即可。

## 状态驱动语言（机制闭环）

以前"精力疲惫"只是 Prompt 里的建议；现在是硬约束闭环：
极端状态 -> 推导**可机检约束**（低精力=LEN<=40/60+禁叹号；低落[相对基线-0.20]=禁叹号+禁笑声；
亢奋=放开叹号笑声）-> 渲染后逐条校验 -> 违规**打回重写一次**（指令点名违规项）->
仍违规**机械兜底**（裁到句界/换标点/删笑声词）。约束、校验、重试、兜底、当时状态快照
全写进消息 meta（`meta.style`）可审计。中常态零约束零开销（单次渲染）。

实测：连续被拒后 valence 跌破线，一句"在吗"换来 41 字、无叹号无笑声、
"我就不多问，不想聊也没事"的克制回应；而用户带来强正面情绪时 valence 被拉回，
约束自动解除——她跟着你乐，语义正确。

## 反思层（成长核心）

人格不再写死：`maybe_reflect` 在证据足够时（默认 ≥12h 间隔且 ≥6 条新消息）触发一次反思——
LLM 阅读真实证据（近期对话主题、主动行为的拒绝率与回应情绪、关系现状、未完成意图），
产出三类结果，全部**有界、留痕、可回滚**：

| 产出 | 机制与边界 |
|---|---|
| 洞察（insights） | 存为 `reflection` 记忆（可检索，进入后续 Prompt 的相关记忆） |
| 新聊天习惯 | 角色卡之外最多 +2 条（`max_extra_rules`），须有证据依据 |
| 人格微调（trait_shift） | 吸引子中心漂移：单步 ≤0.05、累计 ≤0.15（相对角色卡），**收敛性不受影响**（回拉目标只是缓慢有界移动） |

每次反思写 `reflections` 表（证据/产出/应用三维留痕）；`GET /status` 的 `growth`
显示当前漂移与成长规则；一键回滚：`PUT /settings {"growth":{"reset":true}}`。
手动触发：`python -m src.main reflect` 或 `POST /reflect`。
证据不足时它会诚实地什么都不改——实测：关系数据平淡时模型自己回答"暂无依据"。

## 记忆后端：SQLite（默认）或 Mem0（语义记忆）

- **SQLite（默认，零依赖）**：显著度 × 中文二元组关键词检索 × **时间衰减**；意图生命周期、
  溯源审计、混沌张力哈希都在这层。
- **Mem0（`memory.provider: mem0` 或环境变量 `AGENT_MEMORY_PROVIDER=mem0`）**：
  开源记忆层——LLM 从对话抽取结构化事实（自动去重/合并/带日期），
  本地中文向量（fastembed `BAAI/bge-small-zh-v1.5`，ONNX 离线推理）做语义检索，
  qdrant 本地盘持久化。**"我养的小动物叫什么名字"能命中"猫叫年糕"**——
  零关键词重叠。

混合分工：**事实走 mem0，意图生命周期仍走 SQLite**（open/done/expired/推迟是闸门语义，
mem0 没有）；mem0 的 UUID 经 `mem_aliases` 表映射成短数字别名，`[M12]` 溯源格式不变；
mem0 抽取/检索失败自动回退 SQLite 关键词路径，聊天不中断。

**写入与遗忘（2026-10 吸收 wxhook 社区实践）**：
- **三层字面去重**（归一化相等 / 包含 / 编辑距离≥0.8，`src/memory.py`）：同一件事
  换个说法不再重复入库，重复陈述会刷新既有记忆（touch）而非新增行；阈值保守，
  宁可漏合并也不错杀，语义级合并交给 mem0 向量侧。
- **检索时间衰减**（`memory.salience_halflife_days`，默认 14 天，0=关闭）：未被触碰的
  记忆排序权重每半衰期减半、下限 0.35——低价值旧记忆不再永久占据 top_k，而老而强相关的
  记忆（如过敏史）仍能被高相关度拉回前排。
- **mem0 攒批提炼**（`memory.batch_extract`，默认开）：候选消息攒够
  `batch_min_items`（8）条且距上次 ≥`batch_min_minutes`（10 分钟）才做一次 LLM 提炼
  （比逐条 infer 省约一个数量级的 token），提炼 prompt 自带最近 24 条已记事实防
  "换说法重复"；失败保险：空返回重试、失败保留缓冲、连续 `batch_max_fail`（2）次
  自动停用（`maybe_extract(force=True)` 可手动唤醒）。

启用（需 `pip install mem0ai fastembed`；国内加 HF 镜像）：

```bash
pip install mem0ai fastembed
export HF_ENDPOINT=https://hf-mirror.com        # 首次下载向量模型用
export AGENT_MEMORY_PROVIDER=mem0
python -m src.main serve
python tools/backfill_mem0.py                   # 可选：存量记忆回填进 mem0
```

成本：攒批提炼把抽取调用摊薄到约 8 条消息一次（闲聊自动跳过）；检索纯本地零成本。
相关文件：`src/memory.py`（去重/衰减）、`src/memory_mem0.py`（混合后端/批量提炼/桥接适配）、
`tools/backfill_mem0.py`。

## 洞察层（模型揣摩意图，指标外部评判）

每条用户消息在渲染回复前，先让 LLM 做一次廉价的结构化分析（~400 token 上限，实际约 130）。
分析时携带**最近对话与未完成意图作为上下文**，用于消解"那件事/上次说的"这类指代：
`sentiment / emotion / intent_type（闲聊|倾诉|分享|提问|求助建议|情绪安抚|提出计划|推迟计划|取消计划|完成计划|拒绝）/ topics / urgency / cancel_topics / defer_topics / done_topics`。

分析结果喂给外部指标，**决策权仍留在闸门**（感知不决定"是否主动"）：

| 感知结果 | 影响的指标 |
|---|---|
| sentiment / intent_type | 关系更新与反馈结算（比关键词词表准，含蓄拒绝能识别） |
| cancel_topics / defer_topics / done_topics | 意图闭环三分法——彻底取消关闭意图、**推迟保留意图只记备注**（"缓一缓等发工资"之后仍会主动问起）、完成则关闭；含蓄说法（"那件事先算了"）靠上下文消解指代 |
| intent_type=提出计划 / urgency | 记忆类型与显著度（认真的事记得牢，闲聊记得轻） |
| sentiment | 状态脉冲：用户情绪小幅拨动 valence |
| 提问/求助且 urgency 高 | 主动评分的信息缺口（悬而未决的问题提升主动关心优先级） |

兜底与审计：分析失败/超时/解析不出 JSON 一律回退关键词启发式，主链路不中断；
每次感知结果写入 messages 表 meta 与 /chat 返回的 `meta.perception`，可审计。
关闭：`perception.enabled: false`（config.yaml）。成本：每条消息多一次小调用，约增 1-3 秒延迟。

## 主动行为分级与闸门

| 级别 | 行为 | 触发条件（全部满足） |
|---|---|---|
| L0 | 沉默 | 评分不足 / 闸门未过时的默认态 |
| L1 | 微反应 | 阈值 0.12 + 300s 冷却（免打扰时段放行） |
| L2 | 轻主动（提醒意图/分享记忆/轻问/关心） | 阈值 0.35 + **4h 冷却** + 预算 3/天 + 免打扰拦截 + 来源可溯源 |
| L3 | 强主动（换话题/提议活动） | 默认 `active_permissions.L3: false` 关闭 |

闸门检查顺序（全部留痕进 `active_logs.gate_result`）：用户开关 → 角色卡权限 →
评分阈值 → 免打扰时段 → 场景白名单 → 来源存在性 → 冷却（分级 × 拒绝倍率）→ 每日预算。

**每个 L2+ 意图必须挂来源**（未完成意图 / 记忆条目），无来源的主动在候选枚举阶段
就不会出现——不存在纯随机发起话题的路径。

## 审计

主动行为双写落库：

- SQLite 表 `active_logs`：level、intent_type、source_type/source_id（溯源）、
  rendered_text、score_detail（评分明细 JSON）、gate_result（闸门判定明细 JSON）、
  user_response_latency、user_sentiment_delta（反馈结算）
- `logs/active_logs.jsonl`：同内容的追加式流水

```bash
python -c "import sqlite3,json;[print(json.dumps(json.loads(r[0]),ensure_ascii=False)[:200]) for r in sqlite3.connect('data/agent.db').execute('SELECT gate_result FROM active_logs ORDER BY id DESC LIMIT 3')]"
```

记忆溯源：主动分享记忆时输出带 `[M<id>]` 标记，`SELECT * FROM memories WHERE id=<id>`
可回查原文；检索不到的记忆不会进入 Prompt，也就不可能被引用。

## 配置

- `config/config.yaml` — 引擎参数（tick / chaos / convergence / motivation / gate 阈值 / memory）。
  **默认值已按 P5 参数扫描校准**（L2 阈值 0.35）；脱敏模板见 `config/config.example.yaml`。
- `config/persona.yaml` — 角色卡：身份摘要（≤200 字，硬校验）、3-5 条核心不变量、
  吸引子（每维 center/tolerance/weight）、基线、风格（口头禅/禁忌/评分关键词）、
  边界、主动权限、冷却（L2=14400s）、预算（L2=3/天）、免打扰时段。
- 冷却/预算/免打扰以 persona 为准（角色级产品策略）；闸门阈值在 config（引擎级）。

## 测试与评估

```bash
python tests/run_all.py                    # 全量回归（含 HTTP 并发），写 reports/test_report.md
python tools/drift_test.py                 # 50 轮抗漂移：风格命中率 0.94（线 0.8）
python tools/drift_test.py --provider openai_compatible   # 真实 API 评分
python tools/param_sweep.py                # 参数扫描（~4 分钟），写调参报告 + CSV
python tools/ablation_engine.py            # 机制消融：弛豫 vs 回拉（离线）
python tools/drift_real.py --rounds 50 --seeds 3   # 真实模型两臂漂移对照（需 LLM_API_KEY）
```

测试覆盖计划的全部验收点：种子可复现、噪声有界、状态收敛、闸门（免打扰/冷却/预算/
权限/开关/来源）、记忆溯源（反幻觉）、Prompt 重建约束（无聊天历史/记忆 ≤40%/身份 ≤200 字）、
反馈闭环（拒绝加倍冷却、正面提升限幅）、惰性补算精确一致。

## 目录结构

```
config/          config.yaml + persona.yaml（角色卡；脱敏模板 *.example.yaml 随仓库提供）
db/schema.sql    SQLite schema（storage.py 经 PRAGMA user_version 迁移）
src/
  state_engine.py      状态演化：多尺度惰性 tick + 事件脉冲 + 昼夜/孤独调制
  chaos.py             哈希种子 → 平滑噪声（幅度 ≤ epsilon_base）
  persona_attractor.py 吸引子距离 / 回拉 / 一致性
  memory.py            SQLite 记忆（四类）+ 检索 + 张力哈希
  memory_mem0.py       Mem0 混合后端（事实语义记忆 + 桥接 LLM/本地向量适配）
  relation.py          affinity/trust/familiarity（限幅更新）
  motivation.py        主动评分（5.4）+ 候选意图枚举与评分（5.5/5.6）
  gate.py              L0-L3 闸门（阈值/冷却/预算/免打扰/权限/开关/来源）
  prompt_builder.py    每轮动态重建 Prompt（记忆 ≤40% token，无聊天历史）
  llm_client.py        mock / OpenAI 兼容客户端（可替换）
  feedback.py          回复延迟/情绪/拒绝结算 → 冷却倍率 + 关系 + 下轮 Prompt
  scheduler.py         惰性 tick / 常驻循环
  server.py            HTTP API + 网页客户端 + 内置调度线程（serve 子命令）
  web/index.html       零构建聊天网页（轮询主动消息，支持令牌）
  main.py              Agent 编排 + CLI
  demo.py              端到端演示
tests/           回归测试套件（含 HTTP I/O 面）+ run_all.py
tools/           param_sweep.py（调参）+ drift_test.py（抗漂移）+ chat_client.py（终端客户端）
                 ablation_engine.py（机制消融）+ drift_real.py（真实模型漂移）
deploy/          withme.service（systemd 部署示例）
data/            agent.db（运行时生成）
logs/            active_logs.jsonl
reports/         测试报告 / 调参报告 / 抗漂移记录 / sweep CSV（运行时生成，不入库）
```

## 边界与已知限制

- 不实现真实意识/情感，不冒充真人；角色自述为 AI（核心不变量之一）。
- 记忆默认为 SQLite 关键词检索（mem0 后端见上节）；SQLite 模式非向量，
  转述改写可能检索不到。
- 单角色单库：一个 `agent.db` 对应一个 role_id；多角色请分库。
- mock 渲染是确定性模板基线（用于回归与离线演示），语气生动度请接真实 API。
- 打扰率等指标来自文档化的合成用户模型（见调参报告），上线后以真实
  `user_sentiment_delta` 复核。
