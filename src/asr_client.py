"""语音识别客户端：任何 OpenAI 兼容 /audio/transcriptions 端点。

默认对接智谱 GLM-ASR（base_url 复用 llm 段的网关），用标准库手工拼
multipart/form-data——核心零依赖的约定不变。桥接层（微信语音听写）专用。
"""
import json
import os
import time
import urllib.error
import urllib.request

from .config import ASRConfig, LLMConfig


class ASRError(Exception):
    pass


def multipart_body(fields: list, file_field: str, filename: str,
                   mime: str, data: bytes) -> tuple:
    """拼 multipart/form-data。fields 为 (name, value) 列表，返回 (boundary, body)。"""
    boundary = "----WithMeFormBoundary%s" % (int(time.time() * 1000) % 10**9)
    lines = []
    for name, value in fields:
        lines.append(f"--{boundary}\r\n"
                     f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                     f"{value}\r\n")
    lines.append(f"--{boundary}\r\n"
                 f'Content-Disposition: form-data; name="{file_field}"; '
                 f'filename="{filename}"\r\n'
                 f"Content-Type: {mime}\r\n\r\n")
    body = "".join(lines).encode("utf-8") + data + f"\r\n--{boundary}--\r\n".encode()
    return boundary, body


class ASRClient:
    """OpenAI 兼容 /audio/transcriptions。base_url 为空时复用 llm 段的网关。"""

    def __init__(self, cfg: ASRConfig, llm_cfg: LLMConfig):
        self.base_url = (cfg.base_url or llm_cfg.base_url).rstrip("/")
        self.cfg = cfg

    def transcribe(self, audio: bytes, filename: str = "audio.mp3",
                   mime: str = "audio/mpeg") -> str:
        api_key = os.environ.get(self.cfg.api_key_env, "")
        if not api_key:
            raise ASRError(f"环境变量 {self.cfg.api_key_env} 未设置（语音识别需要）")
        boundary, body = multipart_body(
            [("model", self.cfg.model)], "file", filename, mime, audio)
        req = urllib.request.Request(
            self.base_url + "/audio/transcriptions", data=body, method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                     "Authorization": f"Bearer {api_key}"})
        last_err = None
        for attempt in range(2):
            try:
                with urllib.request.urlopen(req, timeout=self.cfg.timeout_seconds) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                text = (data.get("text") or "").strip()
                if not text:
                    raise ASRError("识别结果为空")
                return text
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    detail = e.read().decode("utf-8", "replace")[:300]
                except Exception:  # noqa: BLE001
                    pass
                if e.code >= 500 and attempt == 0:
                    last_err = ASRError(f"HTTP {e.code}: {detail}")
                    time.sleep(1.0)
                    continue
                raise ASRError(f"HTTP {e.code}: {detail}")
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
                if attempt == 0:
                    last_err = ASRError(f"请求失败：{e!r}")
                    time.sleep(1.0)
                    continue
                raise ASRError(f"请求失败：{e!r}")
        raise last_err or ASRError("未知错误")
