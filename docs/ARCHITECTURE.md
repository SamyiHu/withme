# WithMe 架构与代码地图

> 本文是工程结构说明；论文与投稿材料不随开源版发布，见 [OPEN_SOURCE.md](OPEN_SOURCE.md)。

## 1. 分层

```
浏览器 / 微信 / 第三方
        │ REST / wxhook / wcferry
        ▼
I/O 层：src/server.py（HTTP）、src/wechat_bridge.py + src/wxhook.py（微信）
        │
        ▼
编排层：src/main.py（Agent，单锁串行）
  ├─ 状态：state_engine.py / state_dims.py / chaos.py / persona_attractor.py
  ├─ 成长：growth.py（慢速外环，p 收敛到用户拟合目标 P*，见 docs/growth.md）
  ├─ 记忆：memory.py（SQLite）/ memory_mem0.py（可选）
  ├─ 人物：person.py（persons / person_cards / person_events / person_facts）
  ├─ 主动：motivation.py / gate.py / intents.py
  ├─ 渲染：prompt_builder.py / reviewer.py / style_ctrl.py / state_style.py
  └─ 手部：hands.py + plugins/
        │
        ▼
存储：db/schema.sql + SQLite（data/agent.db / data/wechat_users/<room>/agent.db）
```

## 2. 数据模型（schema v10）

| 表 | 作用 |
|---|---|
| `messages` | 对话事件；`sender=user/agent`，`speaker/speaker_role` 记录“谁说的”，`meta.addressee` 记录“对谁说” |
| `memories` | 事件/意图/反思/关系记忆；`person_id` + `scope` 做人物与上下文隔离 |
| `persons` | 身份：wxid、姓名、kind（creator/friend/group_member/agent/stranger） |
| `person_cards` | 关系卡：affinity/familiarity/trust_identity/trust_benevolence/trust_competence/trust_disclosure/caution/attachment/authority |
| `person_events` | 信任变化账本：事件、权重、来源 message_id、前后值 |
| `person_facts` | 事实/声明（预留；v1 只建表） |
| `states` | 角色当前状态向量、关系状态、冷却/预算等 |
| `active_logs` | 主动行为审计：意图、评分、闸门、来源 |
| `reflections` / `hands_log` | 反思与手部动作审计 |
| `growth_events` | 成长账本（append-only）：每次外环推进前后的 p / P* / 证据量；回滚也留痕 |
| `review_logs` | 审核 Agent 留痕：原稿、病灶、改写要求、终稿、是否重写、是否走 LLM、耗时 |
| `tasks` | 后台任务（唱歌/翻唱/画图等）：先落库再执行，状态 queued/running/ready/failed/cancelled + 心跳；桥重启时清理属于 bridge 的孤儿任务并补一句人话通知；心跳超时由看门狗标失败 |
| `outbox` | 出站消息重试队列：文本直发三次失败后落库；歌/图/文件发送失败同样入队，按指数退避补发；`sent/failed` 供健康页查看 |
| `triggers` | 入站触发账本：respond / observe / duplicate，记录是否点名、原因与 1.5 秒内重复投递；只审计，不参与权限 |
| `settings` | 运行时设置与 `config_overrides`（含 `growth_state`：p / 目标 / 证据量） |

## 3. 人物卡与记忆的关系

- **记忆是日记**：发生过的对话、事件、意图、反思；
- **人物卡是当前投影**：这个人在当前上下文里的身份、关系、信任；
- **person_events 是账本**：每次信任变化的原因；
- 卡片不存聊天流水，记忆不存信任分数；记忆按 `person_id + scope` 过滤。

当前 v1：每个 Agent/群各自一个 SQLite，所以**同一个人跨群是两张卡、两套记忆**。跨群全局身份需要被升级人分层同意，见 [OPEN_SOURCE.md](OPEN_SOURCE.md) 与设计说明。

## 4. 群聊行为

- 群消息带 `speaker/speaker_role`；
- Prompt 内显示 `[创造者 阿澄 → 你]` / `[群友 胖达 → 你]` / `[群友 → 群聊]`；
- 创造者是稳定的信任锚，不是主人/上级；信任不等于盲从；
- 群聊保持友善、友爱、帮忙、玩笑，但理智、不站队、不替任何一方攻击；
- 群聊滚动摘要禁用，避免把群友断言/自己的气话写成事实；
- 旁听消息也会建/更新人物卡。

## 5. 双 Agent（互动 Agent + 审核 Agent）

不是两层正则，是两次独立调用、两套独立提示词：

```
输入 → 感知/状态/记忆 → 互动 Agent 写出一句话
                              │
                              ▼
                     审核 Agent（前额叶，不说话）
                     规则快路径 + 独立 LLM 审稿员
                              │ 通过            │ 不通过（给一句改写要求）
                              ▼                 ▼
                           发送        互动 Agent 重写一次 → 复审 → 发送/丢弃
```

- 模式：`reviewer.mode = off | rule | llm`（默认 llm）；
- `reviewer.llm_purposes` 里的用途（默认 respond / proactive / chime）每条都请审稿员；
  其他用途（如到点提醒）只跑规则，规则命中且开了 `risk_triggers` 才终审；
- 病灶清单：能力否认、AI 自曝、客服腔、答非所问/认错人、群聊站队、书面语腔、
  主动消息空转；AI 腔（句号/起手词/两段式）走确定性机械修正，零额外成本；
- 主动/插嘴被审死就**不发**（宁可沉默，也不发一条翻车的话）；回复路径不能沉默，
  所以只重写、仍不合格则保留终稿并把 `cap_fail` 抬升（能力提醒常驻）；
- 审稿调用失败/超时**不阻断**发送；日调用上限 `reviewer.max_llm_per_day` 防预算烧穿；
- 每条出门的话都写 `review_logs`，控制台“审核 Agent”卡片可见模式/命中率/最近审稿。

## 6. 主动回复

`Agent._evaluate_proactive_locked` 的渲染会带：

- 最近对话（`recent_dialogue`）
- 更早对话摘要（`early_summary`）
- 来源记忆/意图
- 状态驱动语气约束
- `主动发起` 规则：具体、有由头、不空转、提问只问一句

**观察模式**：`wechat.observe_chats` 列出的会话，主动/插嘴只写 `active_logs`
（`chime_observe` 或 gate 里带 `observe`）并打日志，不发送、也不写 `messages`，
避免“本来会说的话”污染她自己的对话记忆。先观察再切 `auto`。

**任务 / 出站**：慢任务（唱歌、画图、看B站）先写 `tasks` 再执行；发送失败的消息/歌/图
落 `outbox`，桥每 5 秒按退避补发。`bilibili.watch` 完成后不直接发系统摘要，而是
回 `Agent.on_task_result` 做第二次人格渲染，再由桥发出。`wechat.task_stale_seconds`
（默认 1200）内没有心跳的 running 任务会被 `_task_watch_loop` 标失败并给用户一句
通知；桥重启时 `fail_orphans` 只清自己 owner 的任务，近期歌曲任务会补一句
“没唱完，要重来吗”。

**应用健康看门狗**：`tools/health_watch.py`（`deploy/health_watch.bat`）读
`GET /health/detail`，写 `data/health/watch.json` 与 `logs/health_watch.log`，
异常时退出码 2。它只暴露问题，不自动重启进程，避免根因未除时形成重启风暴。

## 7. 控制台参数

- 页面：`GET /console`
- 数据：`GET /console-data`（人物卡跨会话聚合：主库 + 每个群/私聊用户库的
  `persons` 表，同 `wxid` 合并成一张卡，`sessions` 标出出现在哪些会话）
- 写入：`POST /config`（立即生效 + `settings.config_overrides` 持久化）
- 人格可改：`POST /persona`（基线：身份摘要 / 表达风格 / 基线状态 / 吸引子；
  极限：核心不变量 / 边界 / 容差）→ 写 `config/persona.overrides.json`，
  与 `persona.yaml` 深合并，**不改原文**、可一键恢复出厂；`serve` 立即生效，
  其它进程（微信桥）重启后生效。
- 提示词：`GET /prompt[?session=<id>]`。基线（persona.yaml 原文 + 静态能力清单）、
  初始基线提示词（不接状态/记忆，`base.base_prompt`）、每个会话的叠加项
  （`persona_style` / `growth_rules` / 摘要 / 记忆量），以及**最近一次真正发出去的**
  提示词快照（`settings.last_prompt`，每次 `_render` 覆盖写入，含 sections + 全文）
- 轨迹：`GET /trajectory?limit=120` → `{sessions: [...]}`。**每个会话一根时间轴**：
  私聊一条，每个群 / 私聊用户 Agent（`data/wechat_users/<sid>/agent.db`）各一条；
  条目带绝对 `created_at`，前端按真实时间摆位（输入/模型/工具三条泳道），
  点色块弹出该轮详情抽屉。
  - 控制台只渲染**当前选中的一条会话**：`?meta=1` 拿会话清单（`total`/`last`），
    `?session=<id>` 只搬这一条的消息（避免一次 6 个库、几十万字符）；
  - 轮次切分：用户开口起一轮；非用户条目距上一条超过 15 分钟就另起一条"主动"，
    不把隔夜的主动消息并成跨小时的"模型回合"；
  - 应答消息的 `created_at` 记的是真实落库时刻（meta 里另有 `turn_ms`），
    不再等于用户说话那一刻，所以"模型"泳道能看出真实应答耗时；
  - 接口按 `limit` 截断时 `truncated=true`，页面标"最近 N / 共 M"。
- 可调：tick、chaos、convergence、gate thresholds、motivation weights、dialogue、cost、reflection、reviewer（模式/用途/日上限/超时）、persona 权限/冷却/预算
- 聊天页 `GET /` 的右侧状态栏 + 设置弹窗负责日常使用。

## 8. 测试

```bash
python tests/run_all.py
```

当前覆盖：状态/噪声/收敛、记忆与人物卡、主动闸门、Prompt、双 Agent 审核、多会话轨迹、微信桥、HTTP、Web 页面、配置写入等。
