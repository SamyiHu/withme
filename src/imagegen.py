"""画图客户端：CogView 或任何 OpenAI 兼容 /images/generations 端点。

手部动作 image.draw 的后端（微信桥把生成结果 send_image 发给用户）。
base_url 为空时复用 llm 段的网关，key 复用——零额外配置。
"""
import json
import os
import time
import urllib.error
import urllib.request

from .config import ImageGenConfig, LLMConfig


class ImageGenError(Exception):
    pass


def generate_image(prompt: str, cfg: ImageGenConfig, llm_cfg: LLMConfig,
                   out_dir: str) -> dict:
    """画一张图，下载到 out_dir，返回 {"url": …, "path": 本地路径}。"""
    base_url = (cfg.base_url or llm_cfg.base_url).rstrip("/")
    api_key = os.environ.get(cfg.api_key_env, "")
    if not api_key:
        raise ImageGenError(f"环境变量 {cfg.api_key_env} 未设置（画图需要）")
    if not cfg.enabled:
        raise ImageGenError("image_gen.enabled 为 false")
    body = json.dumps({"model": cfg.model, "prompt": prompt, "size": cfg.size}).encode("utf-8")
    req = urllib.request.Request(
        base_url + "/images/generations", data=body, method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"})
    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout_seconds) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        url = (data.get("data") or [{}])[0].get("url", "")
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", "replace")[:300]
        except Exception:  # noqa: BLE001
            pass
        raise ImageGenError(f"HTTP {e.code}: {detail}")
    except (urllib.error.URLError, TimeoutError, KeyError, IndexError,
            json.JSONDecodeError) as e:
        raise ImageGenError(f"请求失败：{e!r}")
    if not url:
        raise ImageGenError("端点未返回图片 URL")
    # 下载到本地（微信发送更稳，文件也留档）
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "drawn_%s.jpg" % time.strftime("%Y%m%d_%H%M%S"))
    try:
        with urllib.request.urlopen(
                urllib.request.Request(url, headers={"User-Agent": "WithMeAgent"}),
                timeout=cfg.timeout_seconds) as resp:
            raw = resp.read(20_000_000)
        with open(path, "wb") as f:
            f.write(raw)
    except Exception as e:  # noqa: BLE001  下载失败仍返回 URL，交给 send_image 的 URL 模式
        return {"url": url, "path": None, "error": repr(e)}
    return {"url": url, "path": path}
