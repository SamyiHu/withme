"""yamlite: 本仓库专用的 YAML 子集解析器（纯标准库，避免 PyYAML 依赖）。

支持的语法（config/ 下两个文件即全部用法）：
- 嵌套映射，固定 2 空格缩进，禁用 Tab
- 块列表 `- 项`，以及 `- key: value` 起头的字典列表项
- 行内列表 ["a", "b"]、行内映射 {k: v, k2: v2}
- 单/双引号字符串、数字、布尔、null
- # 注释（引号外才生效）

不支持锚点、多文档、块标量等。遇到不支持的语法会给出带行号的错误。
"""


class YAMLError(Exception):
    pass


def loads(text: str):
    lines = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = _strip_comment(raw).rstrip()
        if not line.strip():
            continue
        if "\t" in line:
            raise YAMLError(f"line {lineno}: 禁用 Tab 缩进，请用空格")
        stripped = line.strip()
        if stripped in ("---", "..."):
            continue
        indent = len(line) - len(line.lstrip(" "))
        lines.append((lineno, indent, stripped))
    if not lines:
        return {}
    value, idx = _parse_block(lines, 0, lines[0][1])
    if idx != len(lines):
        raise YAMLError(f"line {lines[idx][0]}: 缩进与上文不一致")
    return value


def _strip_comment(raw: str) -> str:
    quote = None
    for i, ch in enumerate(raw):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "#" and (i == 0 or raw[i - 1] in " \t"):
            return raw[:i]
    return raw


def _parse_block(lines, i, indent):
    """解析从 lines[i] 开始、缩进为 indent 的块，返回 (value, next_index)。"""
    _, _, first = lines[i]
    if first == "-" or first.startswith("- "):
        return _parse_list(lines, i, indent)
    return _parse_map(lines, i, indent)


def _parse_list(lines, i, indent):
    items = []
    while i < len(lines):
        lineno, ind, content = lines[i]
        if ind != indent or not (content == "-" or content.startswith("- ")):
            if ind > indent:
                raise YAMLError(f"line {lineno}: 列表项内容缩进异常")
            break
        rest = content[2:].strip() if content.startswith("- ") else ""
        if not rest:
            # `- ` 后为空：取后续更深缩进的块
            if i + 1 < len(lines) and lines[i + 1][1] > indent:
                value, i = _parse_block(lines, i + 1, lines[i + 1][1])
            else:
                value = None
                i += 1
            items.append(value)
            continue
        if _looks_like_kv(rest):
            # `- key: value` 字典项：把本行改写为更深缩进的映射首行，连同后续同缩进行一起解析
            lines[i] = (lineno, indent + 2, rest)
            value, i = _parse_map(lines, i, indent + 2)
            items.append(value)
        else:
            items.append(_scalar(rest, lineno))
            i += 1
    return items, i


def _parse_map(lines, i, indent):
    result = {}
    while i < len(lines):
        lineno, ind, content = lines[i]
        if ind != indent:
            if ind > indent:
                raise YAMLError(f"line {lineno}: 缩进异常（期望 {indent} 空格）")
            break
        key, sep, rest = _split_key(content, lineno)
        if not sep:
            raise YAMLError(f"line {lineno}: 期望 `key: value` 形式，得到 `{content}`")
        rest = rest.strip()
        if rest:
            result[key] = _scalar(rest, lineno)
            i += 1
        else:
            if i + 1 < len(lines) and lines[i + 1][1] > indent:
                value, i = _parse_block(lines, i + 1, lines[i + 1][1])
            else:
                value = {}
                i += 1
            result[key] = value
    return result, i


def _looks_like_kv(text: str) -> bool:
    if text[0] in "[{":
        return False  # 行内列表/映射整体作为一个标量项
    try:
        _split_key(text, 0)
        return True
    except YAMLError:
        return False


def _split_key(content: str, lineno: int):
    quote = content[0] if content and content[0] in "\"'" else None
    if quote:
        end = content.find(quote, 1)
        if end == -1:
            raise YAMLError(f"line {lineno}: 引号未闭合")
        key = content[1:end]
        rest = content[end + 1:]
        if not rest.startswith(":"):
            raise YAMLError(f"line {lineno}: 引号 key 后必须有冒号")
        return key, ":", rest[1:]
    idx = content.find(":")
    while idx != -1:
        after = content[idx + 1: idx + 2]
        if after in ("", " "):
            return content[:idx].strip(), ":", content[idx + 1:]
        idx = content.find(":", idx + 1)
    raise YAMLError(f"line {lineno}: 无法解析 `{content}`")


def _scalar(text: str, lineno: int):
    text = text.strip()
    if not text:
        return None
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_scalar(p, lineno) for p in _split_inline(inner)]
    if text.startswith("{") and text.endswith("}"):
        inner = text[1:-1].strip()
        result = {}
        if not inner:
            return result
        for part in _split_inline(inner):
            key, sep, rest = _split_key(part.strip(), lineno)
            if not sep:
                raise YAMLError(f"line {lineno}: 行内映射项 `{part}` 缺少冒号")
            result[key.strip()] = _scalar(rest, lineno)
        return result
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "~"):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _split_inline(text: str):
    parts, buf, quote, depth = [], [], None, 0
    for ch in text:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
            buf.append(ch)
        elif ch in "[{":
            depth += 1
            buf.append(ch)
        elif ch in "]}":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf))
    return parts
