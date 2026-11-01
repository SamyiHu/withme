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


# 版权角色名 -> 长相描述。上游（CogView 等）碰到知名 IP 不报错、也不拦：
# 它静默换一张毫无关系的图回来（2026-10-30 实测：提示词「奶龙…」回来一张
# 餐馆照片，接口 200、URL 正常、AI生成水印齐全），于是「画好了」其实是
# 错图。发请求前把名字换成长相，画出来像就够用。真人/品牌不建表——
# 由出图后的视觉自检兜底。
_IP_REWRITES = (
    ("奶龙", "圆滚滚的黄色卡通小胖龙，大脑袋短手脚，憨憨的表情"),
    ("哆啦A梦", "蓝白配色的圆脸机器猫，红项圈挂金色铃铛，肚子上有白色口袋"),
    ("叮当猫", "蓝白配色的圆脸机器猫，红项圈挂金色铃铛，肚子上有白色口袋"),
    ("机器猫", "蓝白配色的圆脸机器猫，红项圈挂金色铃铛，肚子上有白色口袋"),
    ("哆啦美", "黄色圆脸机器猫，头顶红色蝴蝶结"),
    ("皮卡丘", "黄色圆脸电气小老鼠，红脸颊、尖耳朵、闪电形尾巴"),
    ("米老鼠", "穿红短裤黄鞋子的黑色大耳朵卡通老鼠"),
    ("海绵宝宝", "黄色方形海绵小人，白衬衫红领带棕色短裤"),
    ("派大星", "粉红色五角星卡通海星，穿绿短裤"),
    ("小猪佩奇", "粉红色小猪，穿红裙子，鼻子细长"),
    ("蜡笔小新", "五岁圆脸小男孩，粗黑眉毛，红色上衣黄色短裤"),
    ("喜羊羊", "白色卡通小羊，蓝色项圈带小铃铛"),
    ("灰太狼", "灰色卡通狼，戴带补丁的橙色帽子"),
    ("熊大", "棕色大块头卡通熊"),
    ("熊二", "浅棕色胖乎乎的卡通熊"),
    ("光头强", "戴黄色安全帽、留小胡子的卡通伐木工"),
    ("柯南", "戴圆框眼镜的小学生侦探，蓝西装红领结"),
    ("奥特曼", "银红配色紧身衣的蒙面宇宙英雄，胸口有彩色计时器"),
    ("初音未来", "蓝绿色超长双马尾的卡通虚拟歌姬"),
    ("樱桃小丸子", "剪齐刘海的小女孩，白衬衫红裙子"),
    ("龙猫", "灰色胖乎乎的毛绒大猫，肚子有月牙纹"),
    ("Hello Kitty", "白色圆脸小猫，左耳戴红色蝴蝶结，没有嘴"),
    ("凯蒂猫", "白色圆脸小猫，左耳戴红色蝴蝶结，没有嘴"),
    ("库洛米", "白色小兔子，戴黑色兜帽，兜帽上有粉色骷髅头"),
    ("玲娜贝儿", "粉色狐狸，大耳朵蓝色眼睛"),
    ("星黛露", "紫色长耳兔子，蓝色眼睛"),
    ("可达鸭", "黄色胖鸭子，抱着脑袋一脸茫然"),
    ("杰尼龟", "蓝色小乌龟，圆脑袋"),
    ("小火龙", "橙色小恐龙，尾巴尖上有火焰"),
    ("妙蛙种子", "蓝绿色四足小恐龙，背上有一个绿色花苞"),
    ("路飞", "戴草帽、穿红背心的少年海盗"),
    ("鸣人", "金色刺猬头、脸颊有胡须纹的忍者少年，橙色衣服"),
    ("变形金刚", "能变形的巨型机器人"),
)
# 长名先换，避免「小猪佩奇」里的「佩奇」先被别的词吃掉
_IP_REWRITES = tuple(sorted(_IP_REWRITES, key=lambda kv: -len(kv[0])))


def sanitize_prompt(prompt: str):
    """把提示词里的版权角色名换成长相描述。

    返回 (新提示词, 被换掉的词)。非 IP 提示词原样返回。
    """
    text = str(prompt or "")
    swapped = []
    for name, desc in _IP_REWRITES:
        if name in text:
            text = text.replace(name, desc)
            swapped.append(name)
    return text, swapped


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
