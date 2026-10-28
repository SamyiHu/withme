# 微信桥部署指南（个人微信 · 双通道）

把小满接到**个人微信**：文字双向、图片双向（看图+画图）、语音双向。
桥是独立进程，不在 Agent 进程内——REST API 仍是边界（与网页/终端客户端同级）。

## 通道怎么选（关键差异：语音形态 + 微信版本）

| | **wxhook**（默认） | wcf |
|---|---|---|
| 微信 PC 版本 | **4.1.10.27**（锁死） | 3.9.12.51（锁死） |
| hook 来源 | [mengyunqwq/wechat-hook](https://github.com/mengyunqwq/wechat-hook) 系（**有预编译包**，见下） | [WeChatFerry](https://github.com/lich0821/WeChatFerry)（pip 包 `wcferry`，仓库已归档） |
| 接入方式 | version.dll 代理 + HTTP(:30001) + 通知回调 | pip 包，注入 RPC |
| **语音回复形态** | **真语音气泡**（type 34，SILK，长按可转文字） | MP3 **文件卡片**（点开即播；3.9 客户端无发送语音功能，接口层面补不了） |
| 语音收（听） | media_0.db VoiceInfo → SILK → WAV → GLM-ASR | get_audio_msg → mp3 → GLM-ASR |
| Python 依赖 | `pysilk-mod zstandard`（+已有 edge-tts、imageio-ffmpeg） | `wcferry` |

> 为什么默认 wxhook：语音气泡 + 微信版本更新 + hook 活跃维护。
> wcf 通道保留作备选（pip 一条命令、久经考验），切换只需 `wechat.channel` 一行。

```
你（手机微信） ── 微信服务器 ── PC 微信客户端(锁版本, 扫码常驻)
                                      │ version.dll hook（wxhook）或 wcferry
                                 src/wechat_bridge.py
                                      │ REST（127.0.0.1:8787）
                                 src.main serve（小满本体）
```

## 通用配置（两通道共用）

`.env`（仓库根，桥启动时自动读，setdefault 不覆盖系统变量）：

```ini
LLM_API_KEY=sk-xxx            # LLM + ASR + 画图共用
# AGENT_API_TOKEN=xxx         # 服务端开了令牌就填
```

`config/config.yaml` 关键项：

```yaml
wechat:
  channel: wxhook             # 默认；切 wcf 即为备用通道
  owner_wxid: ""              # 空 = 第一个私聊的人自动绑定为唯一主人
  voice_reply: mirror         # mirror=你发语音她回语音 | always | never
  voice_max_chars: 220        # 超过该长度只发文字（微信语音上限 60s）
asr:
  model: glm-asr-2512         # base_url 空 = 复用 llm 网关
image_gen:
  enabled: true               # CogView-3-Flash 免费档（"给我画一只猫"）
llm:
  vision_model: ""            # 看图用；空=回落对话模型
```

启动（两通道一致）：

```bash
python -m src.main serve            # 1) 小满服务端
python -m src.main wechat --probe   # 2) 按通道自检
python -m src.main wechat           # 3) 起桥
```

第一个私聊机器人账号的人自动绑定为唯一主人（`data/wechat_bridge_state.json` 持久化）；
回复与主动消息统一走 `/pending` 轮询投递，微信端体感约 3~10 秒。

---

# 通道 wxhook 部署（默认 · 微信 4.1.10.27 · 语音气泡）

## 1) Python 依赖（全部有预编译 wheel，无需编译器）

```bash
pip install pysilk-mod zstandard edge-tts imageio-ffmpeg
```

> SILK 编解码自动择一：`pysilk-mod`（推荐，cp312 有 wheel）→ `pilk`（≤cp311）
> → Node + `silk-wasm`（机器上有 Node 即可，首次自动 npm install）。

## 2) version.dll：用预编译成品包（无需 VS2022）

上游在 [mengyunqwq/wechat-bot](https://github.com/mengyunqwq/wechat-bot) 的
`release/` 目录发布预编译 `version.dll`（下载后核对 SHA256 再使用）。

自己取的话：

```bash
gh api -H "Accept: application/vnd.github.raw" \
  repos/mengyunqwq/wechat-bot/contents/release/version.dll > version.dll
gh api -H "Accept: application/vnd.github.raw" \
  repos/mengyunqwq/wechat-bot/contents/release/SHA256.txt   # 校验 sha256sum version.dll
```

> 该项目声明「仅供学习研究」且衍生自无 LICENSE 的上游——自己用没问题，
> **不要把 dll 再分发或提交进本仓库**。想自己改 hook 才需要 VS2022 编译
>（`scripts/build.ps1`）。

## 3) 安装微信 4.1.10.27 并注入

1. 装 PC 微信 **4.1.10.27**（自动更新锁死；版本不匹配 hook 失效），例如放 `D:\wxhook\wx4110\`。
2. 完全退出微信（任务管理器里 `Weixin.exe` 清零），把 `version.dll` 拷进微信目录。
3. 带参数启动（**CallBackURL 端口必须等于 `wechat.wxhook.notify_port`，默认 18998**）：

```powershell
Start-Process 'D:\wxhook\wx4110\Weixin.exe' `
  -ArgumentList 'StartPort=30001','RecvType=1','CallBackURL=http://127.0.0.1:18998/recv' `
  -WorkingDirectory 'D:\wxhook\wx4110'
```

4. 扫码登录后验证：`curl http://127.0.0.1:30001/QueryDB/status` 应返回 `"IsLogin":1`；
   **点一次微信左侧「通讯录」**（contact.db 懒加载，不点影响联系人查询）。

## 4) 打开收消息推送（语音发送开关桥会自动开）

在微信目录创建空文件 `hook-push.on`（收消息通知，默认关）。

## 5) 起桥

```bash
python -m src.main wechat --probe   # 应全部 [ok]
python -m src.main wechat
```

语音回复链路（自动降级，任一环失败回落文字）：

```
回复文本 → edge-tts(mp3) → ffmpeg 24kHz/mono/PCM16 → SILK（pysilk/pilk/silk-wasm）
        → POST /SendVoiceMsg{str0Mode:"bytes"} → 微信语音工厂 → 真语音气泡
```

## wxhook 注意事项

- 图片接收依赖 hook 的 `/Decode_Pic`。个别构建上解密可能失效，桥会回复
  "没能打开图片"并记日志——可设 `wechat.wxhook.data_root` 固定数据目录再试。
- 收消息是「通知推送 + 兜底轮询」双保险：通知丢了你也能收到（轮询线程每 2s 扫）。
- 微信升级会破坏 hook（偏移绑定 4.1.10.27）；上游有新版本适配文档可跟。

---

# 通道 wcf 部署（备选 · 微信 3.9.12.51 · 语音=MP3 文件卡片）

1. `pip install --upgrade wcferry`
2. 装 PC 微信 **3.9.12.51**（[WeChatFerry releases](https://github.com/lich0821/WeChatFerry/releases) 页找安装包），**关闭自动更新**。
3. `wechat.channel: wcf` → `python -m src.main wechat`（扫码登录）。

已知限制：语音回复 = TTS MP3 **文件卡片**（3.9 客户端本身没有发送语音的代码路径，
协议层面无解）。要语音气泡请用 wxhook 通道。

---

## 常见问题（通用）

| 现象 | 处置 |
|---|---|
| 收不到消息 | PC 微信在前台登录着；确认你是绑定主人；wxhook 确认 `hook-push.on` 存在 + CallBackURL 端口一致 |
| 语音回复变文字 | edge-tts 未装或回复超长；wxhook 还可能无 SILK 编码器（probe 会查出）或语音开关没开——看桥日志 |
| 图片"没能打开" | wcf：重发一次；wxhook：/Decode_Pic 失败（见上）；超大图（>4MB）会被拒 |
| 想换主人 | 删 `data/wechat_bridge_state.json` 里的 `owner_wxid` 后重启桥 |
| 换微信账号 | 删整个 `data/wechat_bridge_state.json` |

## 安全与风险须知

- 两个通道都通过 DLL 注入 hook PC 微信，属非官方方案：**理论上有封号风险**，
  建议小号 + 单主人 + 低频（默认闸门与人格冷却已足够克制）。
- 桥只处理绑定主人的私聊消息；所有收发在 `logs/` 与 `data/wechat_media/` 留痕。
- 微信版本升级会破坏 hook——升级前先确认 hook 项目适配。
