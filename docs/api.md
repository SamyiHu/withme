# HTTP API 对接指南（v1）

WithMe 的 REST API 是唯一边界：网页客户端、终端客户端、微信桥、任何第三方
bot 框架都走同一套 HTTP 接口。零构建、零 SDK，`curl` 就能对接。

## 快速开始

```bash
python -m src.main serve                # 默认 127.0.0.1:8787，无令牌

curl http://127.0.0.1:8787/health
curl -X POST http://127.0.0.1:8787/api/v1/chat \
     -H "Content-Type: application/json" \
     -d '{"text": "在吗"}'
curl "http://127.0.0.1:8787/api/v1/pending?after_id=0"
```

## 版本前缀与兼容

- **新对接一律用 `/api/v1` 前缀**：`/api/v1/chat`、`/api/v1/pending`…
- 裸路径（`/chat`）与 `/api`、`/api/v1` 前缀**完全等价**，裸路径是历史形态、
  长期保留（自带网页客户端与微信桥都在用）；
- 握手探测：`GET /health`（免鉴权）响应带 `"api": "v1"`，可据此确认版本。

## 认证

| 部署方式 | 配置 | 结果 |
|---|---|---|
| 本机 | 不设 `AGENT_API_TOKEN` | 全部放行（默认） |
| 局域网/公网 | `export AGENT_API_TOKEN=xxx` | 除 `/health` 与两个网页外全部需要令牌 |

令牌二选一随请求携带：

```
Authorization: Bearer <token>
X-API-Token: <token>
```

**安全语义**：设置令牌后，500 错误只返回 `{"error": "internal error"}`
（完整堆栈进服务端日志，不泄露给公网）；未设令牌（本机调试）时 500 返回
`repr(错误)` 便于排障。

## 通用约定

- 所有响应 JSON、UTF-8；CORS 已开放（`Access-Control-Allow-Origin: *`）；
- **错误契约**：`{"error": "…"}`，400=参数问题（消息可读，中文）、401=未授权、
  404=路径不存在、500=内部错误、502/503=TTS 上游问题；
- **请求 body**：合法 JSON dict；空 body 等价 `{}`（`/tick`、`/reflect` 这类
  无参端点直接 POST 即可）；非法 JSON 或非 dict 一律回 400 `invalid json body`；
- 查询参数 `limit` / `after_id` 必须是整数，非法回 400；
- `PUT` 按 `POST` 处理（宽容处理，别依赖）；
- POST 无幂等保证：`/chat` 每次都是一轮真实对话。

## 端点参考

### 网页（免鉴权）

| 方法 路径 | 说明 |
|---|---|
| GET `/` | 聊天网页（零构建单文件） |
| GET `/console` | 后台控制台（人物卡/记忆/轨迹/设置） |

### 观测

| 方法 路径 | 说明 |
|---|---|
| GET `/health` | 存活 + 版本握手，`{"ok": true, "api": "v1", "role_id": …, "llm_provider": …}` |
| GET `/status` | 状态向量 / 关系 / 今日主动 / 开关 |
| GET `/console-data` | 控制台聚合快照 |
| GET `/trajectory?limit=80` | 状态轨迹 |
| GET `/messages?limit=50` | 最近消息（双向，含主动消息） |
| GET `/pending?after_id=N` | **增量拉取** id>N 的 agent 消息，返回 `{"items": […], "last_id": M}` |
| GET `/reminders` | 未完成提醒 |
| GET `/settings` | 当前可调设置视图 |

### 动作与设置

| 方法 路径 | 请求 → 响应 |
|---|---|
| POST `/chat` | `{"text": "…", "images": [{"b64": "…", "mime": "image/jpeg"}]}` → `{"reply", "addon", "id"}`；text 与 images 至少一项，最多 3 张图 |
| POST `/tick` | 无参 → 手动触发一次调度评估 |
| POST `/reflect` | 无参 → 立即执行一次反思 |
| POST `/tts` | `{"text": "…"}`（≤300 字）→ `audio/mpeg`（未装 edge-tts 回 503） |
| POST `/presence` | `{"typing": true}` → 输入心跳 |
| POST `/settings` | `{…}` → 应用设置；非法项 400 |
| POST `/config` | `{…}` → 运行时改配置；非法项 400 |
| POST `/proactive` | `{"enabled": bool}` → 主动消息总开关 |
| POST `/level` | `{"level": "L1|L2|L3", "enabled": bool}` → 分级开关 |

## 对接模式：增量轮询（拉主动消息）

Agent 的回复与主动消息都从 `/pending` 出（微信桥就是这么写的）：

```python
after_id = 0
while True:
    data = get(f"{base}/api/v1/pending?after_id={after_id}").json()
    for msg in data["items"]:
        deliver(msg)            # 按你的渠道发送
    after_id = data["last_id"]  # 水位线：无新消息时原地不动
    time.sleep(2)               # 建议 2s（桥的默认 poll_seconds）
```

要点：

- `last_id` 是**幂等水位线**——崩溃后拿它续传，不会重发也不会漏；
- 别用 0.5s 级高频轮询（无意义，引擎 tick 粒度是秒级）；
- 消息体里有 `kind`（text/image/voice…）与 `id`，`id` 单调递增。

## 上游 LLM / 网关的已知坑（对接必读）

以下全部来自实测，配置示例见 `config/config.example.yaml`：

1. **部分网关丢 `system` 角色**：设定整段注入 user 消息即可解决——
   `llm.prompt_in_user: true`（默认已开）。
2. **本地中转网关不转发音频/画图端点**（405）：`/audio/transcriptions`
   与 `/images/generations` 直连供应商 `base_url`（`asr.base_url`、
   `image_gen.base_url` 各自可配），不走聊天网关。
3. **两把 key 的分工**：`LLM_API_KEY` 给聊天网关；`BIGMODEL_API_KEY` 给
   智谱直连的 ASR / Web Search / CogView（`api_key_env` 可指向任意环境变量名）。
4. **`max_tokens` 要拉满**：带思考链的模型思考也计 token，网关实测 65536
   帽子不计费；给小了表现为回复被截断或空回复。
5. **超时给足 60s**（`llm.timeout_seconds`）：思考链模型长思考时慢。

## 微信桥对接的已知坑

完整安装步骤见 [wechat_setup.md](wechat_setup.md)，这里是对接排障速查：

| 坑 | 处置 |
|---|---|
| 高频查微信库会把库查坏（实测致损坏） | 收消息以 hook 通知推送为主，兜底轮询默认 60s（`fallback_poll_seconds`），别调小 |
| @ 我但没进 atuserlist（写库竞争） | 桥内置懒重查，无需处理；自建通道时注意消息落库有延迟 |
| 通知收不到 | 微信目录建空文件 `hook-push.on`；`CallBackURL` 端口必须等于 `wechat.wxhook.notify_port`（默认 18998） |
| 联系人查不到 | contact.db 懒加载：登录后**点一次微信左侧「通讯录」** |
| 微信自动升级毁掉 hook | 版本锁死：wxhook=4.1.10.27，wcf=3.9.12.51；关自动更新 |
| `version.dll` 来源 | 上游预编译包仅供学习研究，**不要再分发、不要提交进仓库** |
| serve 与微信桥同时用 mem0 | mem0 的本地 qdrant 有跨进程文件锁，两边抢会炸：`serve` 强制 sqlite 记忆（事实仍进同一个库），微信桥侧才启用 mem0 |
| 图片"没能打开" | 微信端单图上限约 4MB；wxhook 的 `/Decode_Pic` 失效时设 `wechat.wxhook.data_root` |

## 集成检查清单

- [ ] `GET /health` 通，`"api": "v1"`；
- [ ] 对接代码只打 `/api/v1` 前缀；
- [ ] 公网/局域网部署已设 `AGENT_API_TOKEN`，且客户端带令牌；
- [ ] `/pending` 用 `last_id` 做水位线，重启可续传；
- [ ] 处理了 400/401/404/500 四种错误形状（`{"error": …}`）；
- [ ] 上游网关按上面第 1–4 条核对过配置。
