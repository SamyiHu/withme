"""中文自然时间解析（手部 reminder.set 用）。

支持：
- 相对：30分钟后 / 1.5小时后 / 2天后 / 十分钟后
- 时段：明天早上9点 / 后天下午3点半 / 今天晚上8:30 / 早上七点
- 星期：周一早上8点（下一个未来周一）
- 绝对：2026-10-23 09:00 / 10-23 09:00
规则：只有时刻没日期视为今天，已过则顺延到明天；解析不了返回 None。
"""
import re
import time

_CN_NUM = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
           "六": 6, "七": 7, "八": 8, "九": 9, "十": 10, "半": 30}

_PERIODS = [("早上", 0), ("清晨", 0), ("上午", 0), ("中午", 12), ("午后", 13),
            ("下午", 12), ("傍晚", 18), ("晚上", 12), ("夜里", 12), ("凌晨", 0)]


def _mk(lt_tuple):
    return time.mktime(lt_tuple + (0, 0, 0, -1))


def _tomorrow_base(now):
    lt = time.localtime(now)
    return lt[0], lt[1], lt[2] + 1


def _num(s):
    s = s.strip()
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s)
    if s in _CN_NUM:
        return float(_CN_NUM[s])
    m = re.fullmatch(r"十([零一二两三四五六七八九]?)", s)
    if m:
        return 10.0 + (_CN_NUM.get(m.group(1), 0) if m.group(1) else 0)
    m = re.fullmatch(r"([一二两三四五六七八九])十([零一二两三四五六七八九]?)", s)
    if m:
        return _CN_NUM[m.group(1)] * 10 + (_CN_NUM.get(m.group(2), 0) if m.group(2) else 0)
    return None


def _hm(time_part, period_offset):
    """'9点30' / '9:30' / '9点半' / '9' -> (hour, minute)（已加时段偏移）。"""
    m = re.match(r"(?P<h>\d{1,2}|[零一二两三四五六七八九十]+)\s*[点时:：]\s*(?P<m>\d{1,2}|半)?", time_part)
    hour = None
    if m:
        hour = _num(m.group("h"))
        minute = _num(m.group("m")) if m.group("m") else 0
        if m.group("m") == "半":
            minute = 30
    else:
        m2 = re.fullmatch(r"(?P<h>\d{1,2}|[零一二两三四五六七八九十]+)(?P<m>\d{2})", time_part)
        if m2:
            hour, minute = _num(m2.group("h")), _num(m2.group("m"))
        elif "半" == time_part[-1:]:
            m3 = re.fullmatch(r"(?P<h>\d{1,2}|[零一二两三四五六七八九十]+)半", time_part)
            if m3:
                hour, minute = _num(m3.group("h")), 30
        else:
            hour, minute = _num(time_part), 0
    if hour is None:
        return None
    hour += period_offset
    if hour >= 24 or minute >= 60:
        return None
    return int(hour), int(minute)


def parse_when(text: str, now: float = None):
    """把自然语言时间转成 epoch；解析不了返回 None。"""
    now = time.time() if now is None else now
    lt = time.localtime(now)
    text = text.strip().replace(" ", "")
    if not text:
        return None

    # 相对时间
    m = re.search(r"(?P<n>\d+(?:\.\d+)?|[一二两三四五六七八九十]+)\s*"
                  r"(?P<u>秒钟?|分钟|小时|个?小时|天)", text)
    if m:
        n = _num(m.group("n"))
        u = m.group("u")
        if n is None:
            return None
        if u.startswith("秒"):
            return now + n
        if u == "分钟" or u == "分":
            return now + n * 60
        if "小时" in u:
            return now + n * 3600
        return now + n * 86400

    # 绝对日期时间
    m = re.search(r"(?P<d>\d{4}-\d{1,2}-\d{1,2}|\d{1,2}-\d{1,2})[ T]?"
                  r"(?P<t>\d{1,2}[:：]\d{1,2}|\d{1,2}点\d{1,2}|凌晨|早上|上午|下午|晚上)?"
                  r"(?P<t2>\d{1,2}[:：]\d{1,2}|\d{1,2}点半?|\d{1,2}点)?", text)
    if m and m.group("d"):
        d = m.group("d")
        parts = [int(x) for x in d.split("-")]
        if len(parts) == 2:
            y, mo, day = lt[0], parts[0], parts[1]
        else:
            y, mo, day = parts
        period = 0
        if m.group("t") in ("下午", "晚上"):
            period = 12
        tp = m.group("t2") or ""
        hm = _hm(re.sub(r"^\d+[:：]\d+", "", tp) or "9", period) if tp else (9, 0)
        if hm is None:
            hm = (9, 0)
        try:
            return _mk((y, mo, day, hm[0], hm[1]))
        except (OverflowError, ValueError):
            return None

    # 星期
    m = re.search(r"周(?P<w>[一二三四五六日天])|星期(?P<w2>[一二三四五六日天])", text)
    weekday = None
    if m:
        wd = m.group("w") or m.group("w2")
        weekday = 6 if wd in ("日", "天") else "一二三四五六".index(wd)
        rest = re.sub(r"周[一二三四五六日天]|星期[一二三四五六日天]", "", text)
        days_ahead = (weekday - lt.tm_wday + 7) % 7 or 7
        base = (lt[0], lt[1], lt[2] + days_ahead)
        return _with_time(rest, base, now)

    # 今天/明天/后天 + 时刻
    day_offset = None
    for word, off in (("大后天", 3), ("后天", 2), ("明天", 1), ("今天", 0), ("今晚", 1)):
        if word in text:
            day_offset, base = off, (lt[0], lt[1], lt[2] + off)
            rest = text.replace(word, "", 1)
            return _with_time(rest, base, now)

    # 只有时刻（含时段词）
    return _with_time(text, (lt[0], lt[1], lt[2]), now, allow_next_day=True)


def _with_time(rest, base, now, allow_next_day=False):
    period_offset = 0
    for word, off in _PERIODS:
        if word in rest:
            period_offset = off
            rest = rest.replace(word, "", 1)
            break
    m = re.search(r"(?P<t>\d{1,2}[:：]\d{1,2}|\d{1,2}点半?|\d{1,2}点(?P<mm>\d{1,2})?|"
                  r"[零一二两三四五六七八九十]+点[零一二两三四五六七八九十半]?|"
                  r"[零一二两三四五六七八九十]+点半?)", rest)
    if not m:
        return None
    hm = _hm(m.group("t"), period_offset)
    if hm is None:
        return None
    try:
        ts = _mk((base[0], base[1], base[2], hm[0], hm[1]))
    except (OverflowError, ValueError):
        return None
    if allow_next_day and ts <= now:
        ts += 86400
    return ts
