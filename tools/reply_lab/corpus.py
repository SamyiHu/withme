"""把公开聊天语料整理成"上下文 -> 人类回复"的用例。

支持两种来源：
  - LCCC（CDial-GPT 的 toy_data.json / 正式数据同构）：{"train": [[t1, t2, ...], ...]}
  - 通用 JSONL：每行 {"context": ["..."], "reply": "..."} 或 {"turns": [...]}

只做离线语料，不碰任何真实会话；机器人不会对真人发消息。
"""
import json
import re

_URL = re.compile(r"https?://|www\.")
_AT = re.compile(r"@[\w\-\u4e00-\u9fff]{1,20}")
_SP = re.compile(r"(?<=[\u4e00-\u9fff\uff00-\uffef])\s+(?=[\u4e00-\u9fff\uff00-\uffef])")


def despace(s: str) -> str:
    """LCCC 的 token 之间带空格（"遭 淋 安 逸 了 ？"），中文之间要去掉。"""
    s = str(s or "").strip()
    for _ in range(2):
        s = _SP.sub("", s)
    return re.sub(r"\s{2,}", " ", s).strip()


def _ok(text: str, lo: int = 2, hi: int = 40) -> bool:
    t = (text or "").strip()
    if not (lo <= len(t) <= hi):
        return False
    if _URL.search(t) or _AT.search(t):
        return False
    if t.startswith(("[图片", "[表情", "[语音")):
        return False
    return True


def load_lccc(path: str, limit: int = None) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cases = []
    for split in ("train", "valid", "test"):
        for turns in data.get(split) or []:
            turns = [despace(t) for t in turns if despace(t)]
            if len(turns) < 2:
                continue
            ctx, reply = turns[:-1], turns[-1]
            if not _ok(ctx[-1]) or not _ok(reply):
                continue
            cases.append({"context": ctx[-4:], "reply": reply})
            if limit and len(cases) >= limit:
                return cases
    return cases


def load_jsonl(path: str, limit: int = None) -> list:
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            turns = [despace(t) for t in (row.get("context") or row.get("turns") or [])]
            reply = despace(row.get("reply") or (turns[-1] if turns else ""))
            ctx = turns[:-1] if row.get("reply") is None else turns
            if not ctx or not _ok(ctx[-1]) or not _ok(reply):
                continue
            cases.append({"context": ctx[-4:], "reply": reply})
            if limit and len(cases) >= limit:
                return cases
    return cases


def load(path: str, limit: int = None) -> list:
    if path.endswith(".jsonl"):
        return load_jsonl(path, limit)
    return load_lccc(path, limit)
