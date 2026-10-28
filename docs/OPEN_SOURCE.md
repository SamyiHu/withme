# 开源范围与发布说明

WithMe 采用 MIT License（见根目录 `LICENSE`）。

本文件描述的导出流程运行在**上游开发仓库**（私有，含论文材料、真实配置、实验报告）；
开源版由 `tools/export_open_source.py` 按 `.open-source-exclude` 过滤导出快照，二者内容不同属正常。

## 1. 随开源版发布的内容

- `src/`：Agent 引擎、记忆、人物卡、主动闸门、微信桥、Web UI
- `config/config.example.yaml`、`config/persona.example.yaml`：脱敏配置模板（真实 `config.yaml` / `persona.yaml` 含个人 wxid 与创造者信息，只留在私有仓库）
- `.env.example`：环境变量模板（真实 `.env` 绝不入库）
- `db/`：SQLite schema
- `tests/`：回归测试
- `docs/`：架构、部署、开发者文档
- `tools/`：通用辅助脚本 + 导出脚本（论文评估脚本按第 2 节排除）
- `plugins/`：动作插件
- `deploy/`：部署模板（本机启动脚本不入库，见第 3 节）

## 2. 不随开源版发布（论文/投稿相关）

以下内容属于论文、实验或投稿材料，不进入公开仓库/发行包：

- `reports/`（整目录）：论文稿、实验报告、评审数据、图表
- `docs/handoff_prompt_*.md`
- `docs/human_eval_*.md`
- `docs/publication_action_plan.md`
- `tools/human_eval*.py`、`tools/make_paper_figs.py`、`tools/benchmarks/`

## 3. 绝不发布

- `.env`、API key、token
- `data/`、`logs/`、`*.db` / `*.db-wal` / `*.db-shm`
- `config/config.yaml`、`config/persona.yaml`（真实主人 wxid、创造者身份）
- `deploy/withme_env.bat`、`deploy/start_withme.bat`、`deploy/start_hidden.vbs`、
  `deploy/wechat_bot.bat`、`deploy/wechat_bot.vbs`（本机路径/本机 key）
- 任何包含个人微信 wxid、真实昵称、聊天原文、头像、语音的素材
  （源码注释与测试夹具一律用虚构占位名，禁词清单在本地 `.open-source-banned.txt`）

## 4. 发行打包（唯一流程）

```bash
python tools/export_open_source.py --out ../withme-open-source --zip
```

脚本做四件事：文件全集取 `git ls-files`（未跟踪产物天然不进包）→ 按
`.open-source-exclude` 排除 → 复制到目标目录 → 自动扫描（`.env` 真实 key 值、
`.open-source-banned.txt` 禁词、真实形态 wxid、sk- 密钥、本机路径、硬编码 key），
命中即非零退出。`--zip` 额外产出可直接上传的压缩包。

发布前人工检查（扫描之外的眼睛）：

1. 翻一遍导出包的 `docs/` 与 `README.md`，确认无指向私有材料的失效引用；
2. `python tests/run_all.py` 全绿（在导出包里跑一遍更稳）；
3. `GET /console` 和 `GET /` 可打开；
4. LICENSE 与 README 中的开源范围一致。

CI（`.github/workflows/ci.yml`）在每次 push/PR 上自动跑四道门：
语法与导入检查、测试矩阵（ubuntu/windows × Python 3.9–3.13）、敏感信息扫描
（同一导出脚本的扫描半程），打 `v*` tag 时自动构建发布包并上传 artifact——
发布即打 tag，不再手工打包。建议在 GitHub 仓库设置里把这些检查设为
required checks 后再开分支保护。

## 5. 尚未实现的跨群身份合并

当前默认仍是**按群/按 Agent 隔离**：同一个 wxid 在两个群是两张人物卡、两套记忆。  
如果未来做全局身份，必须满足：

- 默认不合并；
- 被升级人本人明确同意（分层：身份 / 关系 / 记忆）；
- 同意可撤回、可审计；
- 群主或其他群成员不能代同意；
- 没有同意前，群 A 不能看到群 B 的聊天/记忆。

设计草案见团队讨论记录；代码层尚未启用全局 `person_global`。
