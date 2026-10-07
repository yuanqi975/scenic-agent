"""Current weather from Amap's server-side weather API."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
from fastapi import APIRouter

from ..core.cache import cache_get, cache_set
from ..core.config import AMAP_API_BASE_URL, AMAP_WEATHER_CITY, AMAP_WEATHER_KEY

router = APIRouter(tags=["weather"])


def _unavailable(reason: str, message: str) -> dict[str, object]:
    return {
        "available": False,
        "source": "amap",
        "reason": reason,
        "message": message,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/weather")
def weather() -> dict[str, object]:
    """Return normalized current weather for Jiuzhaigou County."""
    if not AMAP_WEATHER_KEY:
        return _unavailable("not_configured", "请在后端 .env 配置 AMAP_WEATHER_KEY")

    cache_key = f"weather:amap:{AMAP_WEATHER_CITY}"
    cached = cache_get(cache_key)
    if cached:
        return cached

    try:
        response = httpx.get(
            f"{AMAP_API_BASE_URL}/v3/weather/weatherInfo",
            params={
                "key": AMAP_WEATHER_KEY,
                "city": AMAP_WEATHER_CITY,
                "extensions": "base",
                "output": "JSON",
            },
            timeout=5.0,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") != "1" or not payload.get("lives"):
            return _unavailable("provider_error", str(payload.get("info", "高德天气接口暂不可用")))
        live = payload["lives"][0]
        result = {
            "available": True,
            "source": "amap",
            "city": live.get("city", "九寨沟县"),
            "province": live.get("province", "四川省"),
            "weather": live.get("weather", "未知"),
            "temperature": live.get("temperature", "--"),
            "wind_direction": live.get("winddirection", "--"),
            "wind_power": live.get("windpower", "--"),
            "humidity": live.get("humidity", "--"),
            "report_time": live.get("reporttime", ""),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        cache_set(cache_key, result, ttl=600)
        return result
    except (httpx.HTTPError, ValueError, TypeError, KeyError) as exc:
        return _unavailable("request_failed", f"天气服务暂不可用（{type(exc).__name__}）")
