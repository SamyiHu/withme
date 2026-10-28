# WithMe — API-only 拟人化主动智能体

不训练、不微调任何模型。人格、记忆、状态、主动性、权限、收敛全部由工程层实现，
LLM API 只负责把**已选定的意图**渲染成自然语言——混沌负责"活"，收敛负责"是"。

- **混沌**：哈希种子 → 平滑有界噪声，只扰动 7 维内部状态，不决定说什么
- **收敛**：人格吸引子（角色卡定义中心/容差/权重）+ 每轮重建 Prompt，长期不漂移
- **主动**：候选意图（必须挂记忆/意图来源）→ 评分 → L0-L3 闸门 → 才渲染
- **成长**：反思层定期把经历沉淀为洞察，有界预算内微调人格，可回滚
- **审核**：双 Agent——互动 Agent 写，审核 Agent 独立过审，不过审打回重写

零第三方依赖（Python 3.9+ 标准库）；默认 mock 渲染，**无 API key 完整可跑**。

## 快速开始

```bash
python -m src.main serve     # HTTP API + 网页客户端 + 内置调度
# 浏览器打开 http://127.0.0.1:8787/ ——聊天、主动消息、右上角 ⚙ 设置面板

python -m src.main demo      # 端到端演示：应答 → 主动提醒 → 拒绝 → 冷却拦截
python -m src.main status    # 状态向量 / 距吸引子距离 / 关系 / 今日主动
python -m src.main chat      # 终端对话（不经 HTTP）
```

## 接真实模型

```bash
export AGENT_LLM_PROVIDER=openai_compatible
export AGENT_LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4   # 任意 OpenAI 兼容端点
export AGENT_LLM_MODEL=glm-4-flash
export LLM_API_KEY=sk-xxx
python -m src.main serve
```

也可改 `config/config.yaml`（首次运行自动从 `config.example.yaml` 生成），或用网页 ⚙ 面板热改。

## 部署与接入

| 方式 | 说明 |
|---|---|
| 本机 | `python -m src.main serve`，默认只绑 127.0.0.1 |
| 服务器 | `deploy/withme.service`（systemd 示例，含令牌与 HTTPS 要点） |
| HTTP API | 接入唯一边界：`/api/v1` 前缀、增量轮询、统一错误契约 → [docs/api.md](docs/api.md) |
| 个人微信 | 文字/图片/语音双向（附加案例）→ [docs/wechat_setup.md](docs/wechat_setup.md) |
| Windows 自启 | `python -m src.main install`（计划任务，无窗口） |

## 特色

- **双 Agent 审核**：出门前拦能力否认 / AI 自曝 / 客服腔 / 答非所问 / 群聊站队；主动被审死宁可沉默
- **手部（行动层）**：提醒、随手记、抓网页、看 B 站、桌面通知、画图、联网搜索——LLM 只提议，引擎校验执行并审计
- **主动分级闸门**：L0 沉默 → L1 微反应 → L2 轻主动（冷却/预算/免打扰）→ L3 强主动（默认关）；每个主动意图必须挂来源
- **记忆双后端**：SQLite 关键词 + 时间衰减（零依赖）↔ Mem0 语义记忆（"猫叫年糕"能被"我养的宠物"命中），可热切换
- **洞察层**：每条消息先做廉价结构化分析（情绪/意图/紧急度），喂外部指标、不夺决策权
- **状态驱动语言**：极端情绪推导可机检约束，违规打回重写 + 机械兜底，全程留痕
- **反思成长**：证据足够才触发；洞察 / 新习惯 / 人格微调全部有界、留痕、一键回滚
- **插件系统**：往 `plugins/` 丢一个 `.py` 即挂载，沙箱 ctx，范例 `weather.py` 免 key 开箱即用
- **前缀缓存优化**：Prompt 按变化频率排序 + 确定性序列化，DeepSeek 系稳定命中缓存价
- **多角色**：环境变量分库，一进程一角色

机制详解（闸门判定顺序 / 记忆去重与遗忘 / 反思边界 / 手部动作清单 / 插件开发 / 审计表）：[docs/features.md](docs/features.md)

## 数据支撑

| 指标 | 结果 |
|---|---|
| 50 轮抗漂移 | 风格命中率 **0.94**（验收线 0.8） |
| LLM-as-judge 三维评分 | 人味 **5.00** / 一致性 **4.80** / 分寸感 **4.60**（5 分制，验收线 ≥4.0） |
| 主动行为调参（两阶段 240 组扫描） | 推荐配置打扰率 **0.07**、约 1 次轻主动/天、全部收敛 |

复现：`python tools/drift_test.py` · `python tools/judge_eval.py` · `python tools/param_sweep.py`

## 文档

| 文档 | 内容 |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | 分层架构、数据模型、群聊、双 Agent |
| [docs/features.md](docs/features.md) | 功能与机制详解 |
| [docs/api.md](docs/api.md) | HTTP API 对接指南（端点 / 认证 / 轮询 / 已知坑） |
| [docs/wechat_setup.md](docs/wechat_setup.md) | 微信桥安装（wxhook / wcf 双通道） |
| [docs/OPEN_SOURCE.md](docs/OPEN_SOURCE.md) | 开源范围与发布流程 |
| [plugins/README.md](plugins/README.md) | 插件开发 |

## 已知边界

- 不实现真实意识/情感，角色自述为 AI（核心不变量之一）
- SQLite 记忆是关键词检索，转述改写可能检索不到（Mem0 后端解决）
- mock 渲染是确定性模板基线，语气生动度请接真实 API

## License

MIT
