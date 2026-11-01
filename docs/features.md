# 功能与机制详解

主页（README）只留概览；本文是机制细节的唯一权威文档。分层与数据模型见
[ARCHITECTURE.md](ARCHITECTURE.md)，HTTP 接口见 [api.md](api.md)。

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

反馈闭环：回复延迟 / 情绪 / 拒绝会结算成冷却倍率与关系更新（拒绝加倍冷却、
正面提升限幅），影响下一轮 Prompt 与主动资格。

## 手部（行动层）

LLM 永远不直接执行任何东西：感知层最多"提议"一个动作（只能从可用动作清单里选），
引擎校验参数与权限后执行，结果回填回复 Prompt——她说"记上了"时是真的记上了。

| 动作 | 说明 |
|---|---|
| `reminder.set/list/cancel` | 定时提醒；中文时间解析（"40分钟后/明天早上9点/晚上8点半/周一8点"） |
| `note.add` | 随手记，入记忆库可检索 |
| `web.fetch` | 抓网页正文（剥脚本标签、限 200KB、仅 http(s)） |
| `web.search` | 联网搜索（智谱 Web Search Pro，实时信息/新梗/事实核查），搜索结果带原文链接 |
| `shopping.link` | 要购买链接时按型号生成京东/淘宝搜索直达链接（不编造商品页 ID） |
| `bilibili.watch` | "看"B 站视频：元信息+弹幕热词+高赞评论+封面视觉描述合成观后素材 |
| `bilibili.comments` | 读 B 站评论区：热评按点赞排序摘录 |
| `desktop.notify` | Windows 桌面通知（无依赖 PowerShell 气泡） |
| `image.draw` | 画图（CogView-3-Flash 免费档，产物随回复发送） |
| `song.sing` | 点唱：本地 ACE-Step 1.5 生成 55 秒歌声，后台完成后以微信语音气泡发出 |
| `meme.generate` | 表情包生成（需可选依赖 meme_generator，未装则明确报缺） |
| `shell.run` | 本地命令；**默认禁用**（`hands.allow_shell: true` 显式开启） |

到点提醒是用户自己约的闹钟：调度 tick 检查 `reminders` 表，**直发不走过评分闸门**
（active_logs 留痕标记 `user_requested`），渲染失败也保证模板直发，绝不吞提醒。
全部执行写 `hands_log` 表可审计。

### 唱歌（本地 ACE-Step 1.5）

点唱不是用 TTS 念歌词，而是本地音乐模型真的唱出来：

- **触发**：感知层把"唱首歌/点歌/唱一段"提议成 `song.sing`，并写 2–4 句原创中文短句
  （不照搬现成歌词；点具体歌名时按那个情绪写原创词）
- **曲风不写死**：`style` 必须来自用户；对方没说曲风时动作不生成，返回"先问想听什么风格"，
  由回复问一句。只有对方说"随便/都行/你定"时，才由模型挑一个具体曲风
- **时长可指定**：`duration` 来自对话（"唱30秒/来一分钟的"），统一解析成秒并夹在
  [10, `max_duration_seconds`=300]；默认 55 秒。超过 `max_send_seconds`（气泡上限）时
  自动只发高音质文件，不再砍时长
- **歌词可送入**：对方把歌词发过来就**原样**进歌（≤12 行、`max_lyrics_chars` 字上限，
  长歌词记得同时给 duration）；对方没给才由模型写原创短句
- **音色后端可调**：`song_gen.voice`（默认 `女声`）拼进 caption，可改 `男声/低沉男声/
  清亮少女音`；用户在对话里指定音色时，`voice` 参数覆盖后端默认
- **生成**：后台线程调本地 `acestep.cpp`（Vulkan 后端）异步 API：
  `POST /lm` 出歌词/音频码 → `POST /synth` 渲染 mp3；不阻塞这一轮回复
- **投递**：桥收到 `song_sender` 回调，按 `send_mode` 发送：
  `bubble`=mp3→24kHz PCM→SILK 语音气泡；`file`=320kbps mp3 文件卡片（48kHz 立体声原样）；
  `both`=气泡 + 文件（默认）。失败回落另一种/一句人话，不假装唱了
  （投递走多用户模式 `wechat.multi_user: true`，单用户 REST 模式只落盘并在日志里给出路径）
- **音质**：`mp3_bitrate: 320`；微信语音气泡本身是 24kHz 单声道（SILK 上限），
  要听 48kHz 立体声请用文件卡片。模型可选 `acestep-v15-turbo-Q8_0`（快，8 步）/
  `acestep-v15-sft-Q8_0`（50 步，更细）/ `*-xl-*`（4B，质量最高、显存/时间更大）
- **状态**：服务没启动时动作直接返回"唱歌服务没开"，模型会直说这次唱不了

启动本地服务（独立进程，模型放仓库外即可）：

```bash
cd <acestep.cpp 目录>          # 预编译包：serveurperso.com/temp/acestep.cpp-win64
build/Release/ace-server.exe --models models --device Vulkan0 --keep-loaded --port 8085
```

配置见 `config.example.yaml` 的 `song_gen` 段；`provider: mock` 用于离线测试
（生成 1 秒正弦波，不联网）。AMD 显卡走 Vulkan 不需要 ROCm；NVIDIA 可改用
`--device CUDA0`。已知边界：首次请求要把 1.7B LM + 2B DiT 载入显存（约 5–10 秒），
之后每首约 10–20 秒（RX 9070 XT 实测 11 秒/55 秒歌）。

### 翻唱（cover）怎么处理

点具体歌名（"唱《晴天》"）走真实翻唱，不是 ACE-Step 重写：

1. **搜歌/下载**：`songsearch` 先按网易云搜原曲，VIP 失败回退酷我；原唱是男声时
   优先找明确标注「女声/女生/女版」的翻唱（B站用 yt-dlp 下音轨、酷我直链），源音频
   本身就在她的音区，不用移调。找不到女声版才用原唱
2. **分离**：常驻 worker（`C:/path/to/AI/seedvc/code/vc_worker.py`）用 Demucs
   `htdemucs_ft` 在 GPU 上分出人声/伴奏（30 秒约 7 秒）
3. **主唱/和声拆分**：只在 Demucs 的人声轨上跑 UVR `UVR-MDX-NET-Karaoke-2`
   （`C:/path/to/AI/uvr-venv`，模型在 `C:/path/to/AI/uvr-models`），分出主唱与和声。
   不能直接在整首歌上跑该模型：开头吉他/伴奏会漏进“主唱”轨，被 RVC 当人声唱
4. **换色**：只把主唱送进她音色训练出的 **RVC v2 48k 模型**
   （`C:/path/to/AI/rvc/assets/weights/xiaoyi.pth`）。音高优先按原唱性别定：
   感知层知道歌手性别时传 `gender`——男调以她舒适音区（25% 分位，约 253Hz）为目标，
   升 +2~+7（低音男声才到 +7）；女调保持原调 0。不知道性别时用 Praat 估原唱人声中位
   基频，对比她的音区自动判断（女调≈0、偏高男声 +1~+4、男低音 +4~+7）
5. **混音**：换色后的主唱 + Demucs 伴奏 + 原版和声（0.8 音量）320kbps 混音，
   回传语音气泡 / 文件卡片。人声移了 N 半音时，伴奏和和声会用 rubberband 同步移
   N 半音（保持速度），否则人声和伴奏调性对不上

RVC 的工程细节（AMD ROCm 踩坑记录）：

- 训练数据用 edge-tts 小忆（`zh-CN-XiaoyiNeural`）生成 250 句，先训 30 轮、再续到 100 轮；
  训练脚本 `C:/path/to/AI/rvc/train`，无需 fairseq
- ROCm/MIOpen 对长序列 GEMM 会栈溢出（`0xC00000FD`），推理必须 **5 秒分块 + 0.5 秒交叉淡化**；
  54 秒翻唱空闲时约 45 秒（demucs 7s + 主唱分离 16s + 换色 20s）
- RVC 每块输出比输入短约 0.4%，不校正会累计成节奏漂移（12 块约 0.2 秒）。每块转换后
  重采样回源块时长，最后再与源音频严格等长，混音才不会越走越偏
- 必须用 **fp32**（`RVC_FP32=1`）且不能设 `HSA_OVERRIDE_GFX_VERSION`，否则 MIOpen 选错
  kernel 崩溃；fp16 在 9070 XT 上不可用
- worker 启动命令见 `deploy/voice_worker.bat`；没起 worker 时自动退回 Seed-VC 零样本换色

ACE-Step 的 `cover-nofsq` 是另一种东西（让模型重唱，不是换音色），没有接进这条链路。


已知边界（诚实声明）：感知提议依赖 LLM 输出 JSON——推理型模型思考链会吃输出预算，
`perception.max_tokens` 需留足（默认 800）；感知失败回退词表的那一轮不会有手部动作，
模型可能口头答应而未执行。审计：`meta.hand` 为 null 且 `/reminders` 无新行即为
"假执行"，重说一遍即可。

## 状态驱动语言

"精力疲惫"不只是 Prompt 里的建议，而是硬约束闭环：极端状态 → 推导**可机检约束**
（低精力=长度上限+禁叹号；低落[相对基线 −0.20]=禁叹号+禁笑声；亢奋=放开）→
渲染后逐条校验 → 违规**打回重写一次**（指令点名违规项）→ 仍违规**机械兜底**
（裁到句界/换标点/删笑声词）。约束、校验、重试、兜底、当时状态快照全写进
消息 meta（`meta.style`）可审计。中常态零约束零开销（单次渲染）。

## 反思层（成长核心）

`maybe_reflect` 在证据足够时（默认 ≥12h 间隔且 ≥6 条新消息）触发一次反思——
LLM 阅读真实证据（近期对话主题、主动行为的拒绝率与回应情绪、关系现状、未完成意图），
产出三类结果，全部**有界、留痕、可回滚**：

| 产出 | 机制与边界 |
|---|---|
| 洞察（insights） | 存为 `reflection` 记忆（可检索，进入后续 Prompt 的相关记忆） |
| 新聊天习惯 | 角色卡之外最多 +3 条（`max_extra_rules`，可调），满了淘汰最旧一条（FIFO），须有证据依据 |
| 成长外环（docs/growth.md） | 行为证据估计出目标 P*，中心偏移 p 缓慢向它收敛：步长随证据指数衰减（会饱和）、每维包络 ±`growth_envelope`（默认 0.25）、偏离超阈值才保留可塑性；核心不变量/边界不参与 |

成长外环的关键性质（固定数据集离线验收，`tests/test_growth.py`）：

- **收敛**：同一批证据反复喂，步长单调下降到 <1e-6，p 停在 P* 附近；
- **饱和**：收敛后继续喂同样的数据不再移动——成长有尽头，不会无限漂；
- **跟随**：用户行为反转（热络→冷淡）后目标翻转，p 在包络内重新收敛；
- **有界**：任意极端证据 p 不超过包络，且不把中心推出维度取值范围；
- **可见**：控制台「成长」页——收敛曲线（纵轴自动缩放）、当前 vs 出厂/目标表、
  append-only 成长时间线（`growth_events`，回滚也留痕）、一键回滚按钮；
- **进提示词**：收敛后的画像以稳定段【长期成长】进入每轮 Prompt（≤220 字，
  证据 ≥40 条起生效）；原始日志只进 SQLite，不占 Prompt。

每次反思写 `reflections` 表与 `growth_events` 账本；`GET /status` 的 `growth`
显示当前 p / 目标 / 证据量；一键回滚：`PUT /settings {"growth":{"reset":true}}`
（控制台成长页也有按钮）。手动触发：`POST /reflect`。
证据不足时它会诚实地什么都不改。

## 记忆后端

- **SQLite（默认，零依赖）**：显著度 × 中文二元组关键词检索 × **时间衰减**；
  意图生命周期、溯源审计、混沌张力哈希都在这层。
- **Mem0（`AGENT_MEMORY_PROVIDER=mem0`）**：LLM 从对话抽取结构化事实
  （自动去重/合并/带日期），本地中文向量（fastembed `BAAI/bge-small-zh-v1.5`，
  ONNX 离线推理）做语义检索，qdrant 本地盘持久化——"我养的小动物叫什么名字"
  能命中"猫叫年糕"，零关键词重叠。

混合分工：**事实走 mem0，意图生命周期仍走 SQLite**（open/done/expired 是闸门语义，
mem0 没有）；mem0 抽取/检索失败自动回退 SQLite 关键词路径，聊天不中断。

**写入与遗忘**：

- **三层字面去重**（归一化相等 / 包含 / 编辑距离 ≥0.8）：同一件事换个说法不重复入库，
  重复陈述刷新既有记忆（touch）而非新增行；阈值保守，语义级合并交给 mem0 向量侧。
- **检索时间衰减**（`memory.salience_halflife_days`，默认 14 天，0=关闭）：未被触碰的
  记忆排序权重每半衰期减半、下限 0.35——低价值旧记忆不再永久占据 top_k，而老而强相关
  的记忆（如过敏史）仍能被高相关度拉回前排。
- **mem0 攒批提炼**（默认开）：候选攒够 8 条且距上次 ≥10 分钟才做一次 LLM 提炼
  （比逐条省约一个数量级 token）；空返回重试、失败保留缓冲、连续 2 次失败自动停用。

启用 Mem0（可选依赖）：`pip install mem0ai fastembed`，国内加
`export HF_ENDPOINT=https://hf-mirror.com`；存量记忆回填：`python tools/backfill_mem0.py`。

## 洞察层（感知）

每条用户消息在渲染回复前，先让 LLM 做一次廉价的结构化分析（~400 token 上限，实际约 130），
携带最近对话与未完成意图作为上下文，用于消解"那件事/上次说的"这类指代：
`sentiment / emotion / intent_type（闲聊|倾诉|分享|提问|求助建议|情绪安抚|提出计划|推迟计划|取消计划|完成计划|拒绝）/ topics / urgency / cancel_topics / defer_topics / done_topics`。

分析结果喂给外部指标，**决策权仍留在闸门**（感知不决定"是否主动"）：

| 感知结果 | 影响的指标 |
|---|---|
| sentiment / intent_type | 关系更新与反馈结算（含蓄拒绝能识别） |
| cancel/defer/done_topics | 意图闭环三分法：取消关闭、推迟保留（之后仍会主动问起）、完成关闭 |
| intent_type=提出计划 / urgency | 记忆类型与显著度（认真的事记得牢） |
| sentiment | 状态脉冲：用户情绪小幅拨动 valence |
| 提问/求助且 urgency 高 | 主动评分的信息缺口 |

兜底：分析失败/超时/解析不出 JSON 一律回退关键词启发式，主链路不中断；结果写
messages 表 meta 可审计。关闭：`perception.enabled: false`。成本：每条消息约 +1-3 秒。

## DeepSeek 前缀缓存优化

DeepSeek 系模型按**请求前缀**自动命中上下文缓存（命中部分按缓存价计费）。三件事：

1. **Prompt 分节按变化频率升序**：永不变（身份/不变量/风格/输出格式）→ 少变
   （权限/未完成意图）→ 追加式稳定的最近对话（16 条内只增不改）→ 每轮必变
   （关系/记忆/状态/任务沉底）。有钉子测试防止顺序回退。
2. **确定性序列化**：同输入必产出逐字节相同的 Prompt 文本。
3. **命中遥测**：`GET /status` 的 `llm_usage` 按用途报告 `cache_hit_tokens` 与命中率；
   `AGENT_DEBUG_PROMPTS=<路径>` 可把每次请求前缀与 MD5 写日志比对。

实测（经中转桥）：respond 稳定命中人设前缀 ~384 tokens/次。上限在桥接侧；
若桥接能把请求粘到单一上游账号（session affinity），稳定前缀可整体命中。

## 插件系统

**动作插件**：往 `plugins/` 丢一个 `.py`，重启即生效——感知层自动看到新动作并学会提议，
执行/审计/结果回填全走现成管线。最小模板只有 `DESC + ACTIONS` 两个变量，handler 拿
`(params, ctx)`，ctx 提供网络抓取、桌面通知、按插件隔离的 KV 存储，**拿不到**数据库与
Agent 内部。范例：`plugins/weather.py`（Open-Meteo 免 key）。坏插件不拖垮主程序；
`settings` 表 `plugins_disabled` 可禁用。详见 [plugins/README.md](../plugins/README.md)。

**通道插件**：不需要进进程——REST API 就是边界（[api.md](api.md)）。

## 审计

主动行为双写落库：SQLite 表 `active_logs`（level、intent_type、来源溯源、评分明细、
闸门判定明细、反馈结算）+ `logs/active_logs.jsonl` 追加式流水。记忆溯源：主动分享
记忆时输出带 `[M<id>]` 标记，`SELECT * FROM memories WHERE id=<id>` 可回查原文；
检索不到的记忆不会进入 Prompt，也就不可能被引用（反幻觉设计）。

## 配置参考

- `config/config.yaml` — 引擎参数（tick / chaos / convergence / motivation / gate 阈值 /
  memory / dialogue / reviewer / 微信桥）。默认值按两阶段 240 组参数扫描校准
  （L2 阈值 0.35、冷却 14400s、预算 3/天）；脱敏模板 `config.example.yaml`。
- `config/persona.yaml` — 角色卡：身份摘要（≤200 字硬校验）、核心不变量、吸引子
  （每维 center/tolerance/weight）、基线、风格（口头禅/禁忌/聊天习惯）、边界、
  主动权限、冷却/预算/免打扰。冷却预算以 persona 为准（角色级产品策略），闸门阈值
  在 config（引擎级）。
- 环境变量覆盖（不改文件）：`AGENT_LLM_*`、`AGENT_API_TOKEN`、`AGENT_MEMORY_PROVIDER`、
  `AGENT_CONFIG_DIR` / `AGENT_DB_PATH`（多角色：一进程一角色）。

## 常驻运行与设置面板

**Windows 开机自启**（无需管理员）：`python -m src.main install` 生成启动三件套并注册
计划任务（无权限回退启动文件夹）；`uninstall` 移除。

**设置面板**：网页右上角 ⚙，热改立即生效、持久化到 SQLite：主动开关/分级/免打扰、
AI 接入（provider/地址/模型/Key）、记忆后端热切换、洞察开关、人设风格、缓存命中率
实时展示。对应 `GET/PUT /settings`。

**服务器**：`deploy/withme.service`（systemd）；对外暴露用 `--host 0.0.0.0` +
`AGENT_API_TOKEN` + nginx/caddy HTTPS。

## 聊天命令（远程调参）

微信里直接发斜杠命令改参数，不用开电脑：只认主人（owner wxid），群聊私聊都能用
（群里加 @ 更明确，但不强制——微信的 @ 元数据在兜底补收时可能缺失）；命令不进
对话历史、不进记忆、不过审核，由桥直接回执。解析器只认白名单命令名，普通文本里的
`/`、分数、路径不会被误吃；非主人发的命令静默丢弃。

```text
/status                  当前模型/开关/阈值/插嘴配额/本会话状态
/set th2 0.3             闸门 L2 阈值（th1/th2/th3，0–1）
/off chime               群插嘴开关（on|off）；chimecd 冷却秒、chimed 日上限
/set roast mild          玩梗尺度 off|mild|on
/quiet 23:00-08:00       安静时段（off 清空）
/set model <名字>        换模型（私聊限定）
/chat off                停用当前会话；/chat <群名> on|off 可在别处按名字开关
/chat list               列出已知会话的开关状态
/get th2 · /audit · /undo · /help [项]
```

作用域分两层：`th2`、`tick`、`roast`、`review`、`l*` 等会话级参数写到该会话自己的
库（群与私聊天然隔离，重启从 settings 恢复）；`chime`、`voice`、`chat` 等桥级参数
改桥进程内存并写 `data/wechat_bridge_state.json` 的 overrides，重启自动覆盖
`config.yaml`（配置文件仍是默认值）。每次成功修改写 `config_audit` 表，`/undo` 回滚
上一笔；非法值只回提示、不动配置。单用户模式没有本地 Agent，只支持桥级命令。

## 测试与评估

```bash
python tests/run_all.py           # 全量回归（含 HTTP 并发），写 reports/test_report.md
python tools/drift_test.py        # 50 轮抗漂移（--provider openai_compatible 接真实 API）
python tools/param_sweep.py       # 参数扫描，写调参报告 + CSV
python tools/ablation_engine.py   # 机制消融：弛豫 vs 回拉（离线）
python tools/drift_real.py        # 真实模型两臂漂移对照（需 LLM_API_KEY）
python tools/judge_eval.py        # LLM-as-judge 三维评分
```

## 能力清单

| 能力 | 落点 |
|---|---|
| 时间感/自传体记忆 | 相关记忆带相对时间（今天/上周），初识锚点 `known_days`，回调带时间感 |
| 场景感知（打字勿扰） | `POST /presence {typing}` → 90s 内主动评分打扰成本 +0.25 |
| 反思调行为参数 | 反思可提议 `gate_shift`（L2 阈值 ±0.05/步、±0.15 封顶），随拒绝率自适应 |
| 多步手部任务 | 感知可提议动作序列（≤3，按序执行，结果合并审计） |
| 后台任务持久化 | 歌曲/画图先落 `tasks`；`task.status` / `task.cancel` / `task.retry` 可查、可取消、可重试；桥重启或心跳超时会通知用户 |
| 异步看视频 | `bilibili.watch` 先回“在看了”，看完把结果交回 Agent 做第二次人格渲染再发，不把系统摘要直接冒充她说话 |
| 媒体出站补发 | 文本/歌/图/文件发送失败落 `outbox`，按指数退避重试；健康页可看积压与失败 |
| 任务权限硬闸 | 群友可点歌/画图，但 `shell.run` 只允许创造者；任务记录发起人，不能互相取消/重试 |
| 入站触发审计 | `triggers` 记录 respond / observe / duplicate，1.5 秒内重复投递不再走 Agent |
| 轻量召回 | `memory.recall` 按关键词翻旧消息和任务，任务卡随提示词注入，不把全量历史塞窗口 |
| 成本护栏 | `cost.daily_token_budget`（默认 60 万 token/日），耗尽自动降级非核心路径，核心聊天不降 |
| VL 看画面 | `bilibili.watch` 附封面视觉描述；`AGENT_VLM_MODEL` 可指定视觉模型 |
| 语音 TTS / ASR | `POST /tts`（edge-tts）；ASR 走 OpenAI 兼容 `/audio/transcriptions` |
| 图片理解/生成 | 收图走视觉理解；`image.draw` 画图随回复发送 |
| 多角色 | `AGENT_CONFIG_DIR` + `AGENT_DB_PATH`，一进程一角色 |
