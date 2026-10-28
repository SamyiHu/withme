# 动作插件（社区扩展）

给她加"手"不需要改核心代码：往本目录丢一个 `.py` 文件，重启服务即生效。
感知层会自动看到新动作并学会在合适时机提议；执行、审计（`hands_log`）、
结果回填回复，全部走现成管线。

## 最小模板

```python
# plugins/my_tool.py
DESC = "一句话描述（展示用）"

def do_something(params, ctx):
    # params: 感知层从用户消息里提取的参数（你注册时声明的那些）
    # ctx:    受限上下文（见下）
    return {"ok": True, "summary": "给回复模型看的结果文本"}

ACTIONS = [{
    "name": "my_tool.do",            # 必须 前缀.动作（小写字母数字下划线）
    "desc": "什么情况下用这个动作（感知层按此判断）",
    "params": {"location": "参数说明，写清楚格式示例"},
    "handler": do_something,
}]
```

## ctx 能用什么（PluginContext）

| 方法 | 说明 |
|---|---|
| `ctx.fetch_text(url)` / `fetch_json(url)` / `fetch(url)` | 带正常 UA 的网络请求 |
| `ctx.notify(title, text)` | 弹桌面通知 |
| `ctx.kv_set(key, value)` / `ctx.kv_get(key)` | 按插件隔离的持久 KV（存 settings 表） |
| `ctx.log(...)` | 打日志（前缀 plugin:名字） |

**拿不到**：数据库句柄、Agent 内部对象、用户消息历史——插件只能看到
本轮提议的参数。返回值必须是 `{"ok": bool, "summary": str}`（summary 会
原样进入回复 Prompt，别塞敏感信息）。

## 规则与安全（诚实声明）

- 插件与主程序**同进程**运行，拥有完整 Python 能力——安装第三方插件
  意味着你信任它的作者。请只安装来源可信的插件。
- 加载失败/单个动作崩溃只影响该插件，主程序照常；可在 settings 表的
  `plugins_disabled`（列表）里禁用某个插件（重启生效）。
- 动作名全局唯一，与内置冲突的注册会被整体拒绝。
- 内置动作清单见 `src/hands.py`；范例插件：`weather.py`（Open-Meteo，免 key）。

## 通道类扩展

想让她出现在别的平台（Telegram/微信/QQ bot、桌面小组件）**不需要插件**：
REST API（`POST /chat`、`GET /pending` 增量轮询）就是边界，任何语言直接调，
见根 README 的 API 一览。
