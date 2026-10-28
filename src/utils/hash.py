"""稳定哈希：同输入必同输出，用作混沌层的受控熵源。"""
import hashlib


def stable_hash(*parts) -> int:
    """把任意多个部分拼成一个 64 位稳定整数（跨进程/跨平台可复现）。"""
    h = hashlib.sha256()
    for p in parts:
        s = p if isinstance(p, str) else repr(p)
        h.update(s.encode("utf-8"))
        h.update(b"\x1f")  # 分隔符，避免相邻字段拼接歧义
    return int.from_bytes(h.digest()[:8], "big")


def hash_to_unit(value: int, index: int = 0) -> float:
    """从哈希整数派生第 index 个 [-1, 1] 均匀分布的确定性伪随机数。"""
    h = hashlib.sha256(f"{value}:{index}".encode("utf-8")).digest()
    return (int.from_bytes(h[:4], "big") % 20001 - 10000) / 10000.0
