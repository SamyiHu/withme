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
  ├─ 记忆：memory.py（SQLite）/ memory_mem0.py（可选）
  ├─ 人物：person.py（persons / person_cards / person_events / person_facts）
  ├─ 主动：motivation.py / gate.py / intents.py
  ├─ 渲染：prompt_builder.py / reviewer.py / style_ctrl.py / state_style.py
  └─ 手部：hands.py + plugins/
        │
        ▼
存储：db/schema.sql + SQLite（data/agent.db / data/wechat_users/<room>/agent.db）
```

## 2. 数据模型（schema v6）

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
| `review_logs` | 审核 Agent 留痕：原稿、病灶、改写要求、终稿、是否重写、是否走 LLM、耗时 |
| `settings` | 运行时设置与 `config_overrides` |

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

## 7. 控制台参数

- 页面：`GET /console`
- 数据：`GET /console-data`
- 写入：`POST /config`（立即生效 + `settings.config_overrides` 持久化）
- 轨迹：`GET /trajectory?limit=120` → `{sessions: [...]}`。**每个会话一根时间轴**：
  私聊一条，每个群 Agent（`data/wechat_users/<room>/agent.db`）各一条；
  条目带绝对 `created_at`，前端按真实时间摆位（输入/模型/工具三条泳道），
  点色块弹出该轮详情抽屉，点“收起”折叠某条会话轴；窗口可选近 1h/6h/24h/全部。
- 可调：tick、chaos、convergence、gate thresholds、motivation weights、dialogue、cost、reflection、reviewer（模式/用途/日上限/超时）、persona 权限/冷却/预算
- 聊天页 `GET /` 的右侧状态栏 + 设置弹窗负责日常使用。

## 8. 测试

```bash
python tests/run_all.py
```

当前覆盖：状态/噪声/收敛、记忆与人物卡、主动闸门、Prompt、双 Agent 审核、多会话轨迹、微信桥、HTTP、Web 页面、配置写入等。
