"""可替换的 LLM 客户端。LLM 只负责把已选定的意图渲染成自然语言，
不决定是否主动、不决定主动什么（原则 D）。

- MockLLMClient：无 API key、离线可跑，确定性输出（seed 控制表达变体）。
- OpenAICompatibleClient：任何 OpenAI /chat/completions 兼容端点（含 GLM）。
"""
import json
import os
import time
import urllib.error
import urllib.request

from .config import LLMConfig


class LLMClientError(Exception):
    pass


class BaseLLMClient:
    name = "base"

    def render(self, prompt: dict, seed: int) -> str:
        raise NotImplementedError

    def render_raw(self, text: str, seed: int, *, temperature: float = 0.3,
                   max_tokens: int = 250, timeout: float = None) -> str:
        """轻量推理调用：不做角色渲染，直接发一段指令文本。mock 不支持。"""
        return ""


def render_prompt_text(prompt: dict) -> str:
    """把结构化 prompt 渲染成 system 消息文本。"""
    sections = []
    for key, value in prompt.items():
        if isinstance(value, (dict, list)):
            body = json.dumps(value, ensure_ascii=False, indent=None)
        else:
            body = str(value)
        sections.append(f"【{key}】{body}")
    return "\n".join(sections)


# ---------------- Mock ----------------

_TEMPLATES = {
    "respond": [
        "嗯哼，你说的「{snippet}」我记下了。{catch}",
        "诶，「{snippet}」听起来有点意思，后来呢？{catch}",
        "我在听呢。「{snippet}」……嗯，先别急，慢慢说。{catch}",
    ],
    "micro_reaction": ["（{catch}）", "嗯…我在。", "……{catch}"],
    "ask_light_question": [
        "话说回来，{topic}那边怎么样了？{catch}",
        "对了，有点好奇：{topic}后来有动静吗？{catch}",
    ],
    "share_related_memory": [
        "突然想起来你之前说的「{mem}」[M{id}]，嗯……就突然想到。{catch}",
        "话说回来，记得你提过「{mem}」[M{id}]嘛。{catch}",
    ],
    "remind_open_intent": [
        "诶，之前你说的{intent}，现在怎么样啦？{catch}",
        "突然想起来，{intent}——有进展吗？{catch}",
    ],
    "express_care": [
        "有点担心你，别太累了。{catch}",
        "{catch}记得好好休息，我一直都在。",
    ],
    "shift_topic": [
        "换个话题！{topic}，你觉得怎么样？{catch}",
    ],
    "propose_small_activity": [
        "要不要一起{activity}？就一小会儿。{catch}",
    ],
    "reminder_due": [
        "⏰ 到点啦：{snippet}。{catch}",
        "诶，你之前让我提醒你——{snippet}。{catch}",
    ],
    "silence": [""],
}


class MockLLMClient(BaseLLMClient):
    name = "mock"

    def __init__(self, persona_style: dict = None):
        self.persona_style = persona_style or {}
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "estimated": True}

    def render(self, prompt: dict, seed: int) -> str:
        task = prompt.get("本次任务", {})
        intent = task.get("意图", "respond")
        if intent in ("silence",) or prompt.get("主动权限") == "L0-仅沉默":
            return ""
        templates = _TEMPLATES.get(intent) or _TEMPLATES["respond"]
        template = templates[seed % len(templates)]
        catch = self._catchphrase(seed)
        memories = prompt.get("相关记忆") or []
        mem = memories[0] if memories else {}
        mem_content = mem.get("内容") if isinstance(mem, dict) else None
        mem_content = mem_content or (mem.get("content") if isinstance(mem, dict) else "")
        intents = prompt.get("未完成意图") or []
        open_intent = intents[0].get("内容") or intents[0].get("content") if intents else "那件事"
        snippet = self._snippet(prompt.get("最近对方消息", ""))
        topic = mem_content or "最近的事"
        if len(topic) > 18:
            topic = topic[:18] + "…"
        text = template.format(
            snippet=snippet, catch=catch,
            mem=self._short(mem_content or "那件事"),
            id=mem.get("id", 0),
            intent=self._short(open_intent),
            topic=topic,
            activity=self._short(open_intent),
        )
        return self._with_tone(text, prompt)

    def _catchphrase(self, seed: int) -> str:
        phrases = self.persona_style.get("catchphrases") or ["嗯哼"]
        return phrases[seed % len(phrases)]

    @staticmethod
    def _snippet(text: str, limit: int = 20) -> str:
        text = (text or "").strip()
        if len(text) > limit:
            text = text[:limit] + "…"
        return text or "刚才的事"

    @staticmethod
    def _short(text: str, limit: int = 16) -> str:
        text = (text or "").strip()
        return text[:limit] + "…" if len(text) > limit else text

    def _with_tone(self, text: str, prompt: dict) -> str:
        values = (prompt.get("当前状态") or {}).get("数值") or {}
        energy = float(values.get("energy", 0.6))
        if energy < 0.35 and not text.startswith("（"):
            return "（有点犯困…）" + text
        return text


# ---------------- OpenAI 兼容 ----------------

def parse_usage(usage: dict) -> dict:
    """提取 token 用量，兼容 DeepSeek 风格的缓存命中字段（桥接会透传）。"""
    if not isinstance(usage, dict):
        return {"prompt_tokens": 0, "completion_tokens": 0, "cache_hit_tokens": None,
                "cache_miss_tokens": None, "estimated": False}
    hit = usage.get("prompt_cache_hit_tokens")
    if hit is None:
        hit = usage.get("cache_read_input_tokens")
    miss = usage.get("prompt_cache_miss_tokens")
    return {"prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "cache_hit_tokens": hit,
            "cache_miss_tokens": miss,
            "estimated": False}


class OpenAICompatibleClient(BaseLLMClient):
    name = "openai_compatible"

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "estimated": False}

    def build_messages(self, prompt: dict) -> list:
        text = render_prompt_text(prompt)
        if self.cfg.prompt_in_user:
            # 适配丢弃 system 角色的网关：设定整段并入 user 消息
            return [{"role": "user",
                     "content": text + "\n\n请只依据以上设定与本轮任务，直接输出回复内容。"}]
        return [
            {"role": "system", "content": text},
            {"role": "user", "content": "请按以上设定与本轮任务输出回复。"},
        ]

    def render(self, prompt: dict, seed: int) -> str:
        body = {
            "model": self.cfg.model,
            "messages": self.build_messages(prompt),
            "temperature": self.cfg.temperature_surface,
            "seed": seed % (2 ** 31),
            "max_tokens": self.cfg.max_tokens,
        }
        return self._post(body)

    def render_raw(self, text: str, seed: int, *, temperature: float = 0.3,
                   max_tokens: int = 250, timeout: float = None) -> str:
        """不做角色渲染的短调用（审核 Agent / 插嘴决策等）。
        timeout 可单独收紧：审核超时宁可不审，也不能卡住回复。"""
        if self.cfg.prompt_in_user:
            messages = [{"role": "user", "content": text + "\n只按要求输出结果。"}]
        else:
            messages = [
                {"role": "system", "content": "你是消息分析器，严格按要求输出。"},
                {"role": "user", "content": text},
            ]
        return self._post({
            "model": self.cfg.model,
            "messages": messages,
            "temperature": temperature,
            "seed": seed % (2 ** 31),
            "max_tokens": max_tokens,
        }, timeout=timeout)

    def render_image(self, text: str, image_bytes: bytes, mime: str = "image/jpeg",
                     seed: int = 0, max_tokens: int = 500) -> str:
        """视觉理解：图 + 指令 -> 文字描述（需 llm.vision_model）。"""
        # 视觉模型未单独配置时回落到对话模型（deepseek-v4.1-flash 等原生支持看图）
        vision = self.cfg.vision_model or self.cfg.model
        import base64
        b64 = base64.b64encode(image_bytes).decode()
        body = {
            "model": vision,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": text},
                {"type": "image_url",
                 "image_url": {"url": "data:%s;base64,%s" % (mime, b64)}},
            ]}],
            "temperature": 0.2, "max_tokens": max_tokens,
        }
        return self._post(body)

    def _post(self, body: dict, timeout: float = None) -> str:
        debug_path = os.environ.get("AGENT_DEBUG_PROMPTS")
        if debug_path:
            import hashlib
            text0 = body["messages"][0]["content"]
            with open(debug_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "ts": time.time(), "model": body.get("model"),
                    "head": text0[:600],
                    "md5": hashlib.md5(text0.encode("utf-8")).hexdigest(),
                    "len": len(text0)}, ensure_ascii=False) + "\n")
        api_key = os.environ.get(self.cfg.api_key_env, "")
        if not api_key:
            raise LLMClientError(
                f"环境变量 {self.cfg.api_key_env} 未设置；离线请用 llm.provider=mock")
        url = self.cfg.base_url.rstrip("/") + "/chat/completions"
        last_err = None
        for attempt in range(2):  # 5xx/超时重试一次
            req = urllib.request.Request(
                url, data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {api_key}"},
                method="POST")
            try:
                with urllib.request.urlopen(
                        req, timeout=timeout or self.cfg.timeout_seconds) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                self.last_usage = parse_usage(data.get("usage") or {})
                content = data["choices"][0]["message"]["content"]
                return content.strip()
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode("utf-8", "replace")[:300]
                except Exception:
                    pass
                if e.code >= 500 and attempt == 0:
                    last_err = LLMClientError(f"HTTP {e.code}: {detail}")
                    time.sleep(1.0)
                    continue
                raise LLMClientError(f"HTTP {e.code}: {detail}")
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
                if attempt == 0:
                    last_err = LLMClientError(f"请求失败：{e!r}")
                    time.sleep(1.0)
                    continue
                raise LLMClientError(f"请求失败：{e!r}")
        raise last_err or LLMClientError("未知错误")


def create_client(cfg: LLMConfig, persona_style: dict = None) -> BaseLLMClient:
    if cfg.provider == "mock":
        return MockLLMClient(persona_style)
    if cfg.provider == "openai_compatible":
        return OpenAICompatibleClient(cfg)
    raise LLMClientError(f"未知 llm.provider: {cfg.provider}")
