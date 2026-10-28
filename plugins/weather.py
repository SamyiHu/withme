"""天气插件（范例）：Open-Meteo，免 API key，开箱即用。

展示社区插件的最小形态——只要 DESC + ACTIONS 两个模块级变量。
"""
import json
from urllib.parse import quote

DESC = "天气查询（Open-Meteo，免 key）"

WEATHER_CN = {0: "晴", 1: "基本晴", 2: "局部多云", 3: "阴", 45: "雾", 48: "雾凇",
              51: "毛毛雨", 53: "毛毛雨", 55: "毛毛雨", 61: "小雨", 63: "中雨",
              65: "大雨", 71: "小雪", 73: "中雪", 75: "大雪", 80: "阵雨",
              81: "阵雨", 82: "强阵雨", 95: "雷阵雨", 96: "雷阵雨伴冰雹",
              99: "雷阵雨伴冰雹"}


def get_weather(params, ctx):
    loc = str(params.get("location", "")).strip()
    if not loc:
        return {"ok": False, "summary": "缺少 location"}
    geo = ctx.fetch_json(
        f"https://geocoding-api.open-meteo.com/v1/search?name={quote(loc)}"
        f"&count=1&language=zh&format=json")
    results = geo.get("results") or []
    if not results:
        return {"ok": False, "summary": f"找不到地名：{loc}"}
    g = results[0]
    w = ctx.fetch_json(
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={g['latitude']}&longitude={g['longitude']}"
        f"&current=temperature_2m,apparent_temperature,weather_code"
        f"&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max"
        f"&timezone=auto&forecast_days=2")
    cur = w.get("current", {})
    day = w.get("daily", {})
    code = cur.get("weather_code", 0)
    desc = WEATHER_CN.get(code, f"天气代码{code}")
    rain = (day.get("precipitation_probability_max") or [0])[0]
    tomorrow = ""
    if len(day.get("temperature_2m_max", [])) > 1:
        tomorrow = (f"；明天{(day['temperature_2m_min'][1]):.0f}~"
                    f"{(day['temperature_2m_max'][1]):.0f}°C，"
                    f"降水概率{(day.get('precipitation_probability_max') or [0, 0])[1]}%")
    return {"ok": True, "summary": (
        f"{g.get('name', loc)}当前{cur.get('temperature_2m')}°C"
        f"（体感{cur.get('apparent_temperature')}°C），{desc}；"
        f"今天{(day.get('temperature_2m_min') or [0])[0]:.0f}~"
        f"{(day.get('temperature_2m_max') or [0])[0]:.0f}°C，降水概率{rain}%"
        f"{tomorrow}")}


ACTIONS = [{
    "name": "weather.get",
    "desc": "用户问某地天气/要不要带伞/穿什么时使用",
    "params": {"location": "城市名，如：广州"},
    "handler": get_weather,
}]
