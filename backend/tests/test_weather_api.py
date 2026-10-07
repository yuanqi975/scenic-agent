from __future__ import annotations

import sys

sys.path.insert(0, "backend")

from fastapi.testclient import TestClient

from app.main import app
from app.api import weather as weather_api


def test_weather_endpoint_degrades_without_amap_key(monkeypatch):
    monkeypatch.setattr(weather_api, "AMAP_WEATHER_KEY", "")
    response = TestClient(app).get("/api/v1/weather")
    assert response.status_code == 200
    assert response.json()["available"] is False
    assert response.json()["reason"] == "not_configured"


def test_weather_endpoint_normalizes_amap_live_response(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "status": "1",
                "lives": [
                    {
                        "province": "四川省",
                        "city": "九寨沟县",
                        "weather": "多云",
                        "temperature": "18",
                        "winddirection": "西北",
                        "windpower": "3",
                        "humidity": "62",
                        "reporttime": "2026-10-04 15:00:00",
                    }
                ],
            }

    monkeypatch.setattr(weather_api, "AMAP_WEATHER_KEY", "test-key")
    monkeypatch.setattr(weather_api.httpx, "get", lambda *args, **kwargs: FakeResponse())
    monkeypatch.setattr(weather_api, "cache_get", lambda key: None)
    monkeypatch.setattr(weather_api, "cache_set", lambda *args, **kwargs: None)
    response = TestClient(app).get("/api/v1/weather")
    assert response.status_code == 200
    assert response.json()["available"] is True
    assert response.json()["temperature"] == "18"
    assert response.json()["wind_power"] == "3"
