"""Cache and rate limiting with deliberate Redis -> process-memory degradation.

Production uses Redis so that caches and counters are shared across workers; local
development without Redis must still behave correctly, never silently break.
"""

from __future__ import annotations

import json
import hashlib
import threading
import time
from collections import OrderedDict
from typing import Any

from fastapi import HTTPException, Request

from .config import CACHE_TTL_SECONDS, RATE_LIMIT_PER_MINUTE, REDIS_URL
from . import config

_redis_client: Any | None = None
_redis_disabled = False
_memory_cache: dict[str, tuple[float, str]] = {}
_memory_rate_limits: OrderedDict[str, tuple[float, int]] = OrderedDict()
_rate_lock = threading.Lock()
_RATE_SCRIPT = """
local count = redis.call('INCR', KEYS[1])
if count == 1 then redis.call('EXPIRE', KEYS[1], ARGV[1]) end
return count
"""


def redis_client():
    """Return a live Redis client, or ``None`` when Redis is unavailable."""
    global _redis_client, _redis_disabled
    if _redis_disabled:
        return None
    if _redis_client is not None:
        return _redis_client
    try:
        from redis import Redis

        client = Redis.from_url(
            REDIS_URL, decode_responses=True, socket_connect_timeout=0.15, socket_timeout=0.15
        )
        client.ping()
        _redis_client = client
        return client
    except Exception:
        _redis_disabled = True
        return None


def reset_redis_state() -> None:
    """Test helper: forget a previously detected Redis outage."""
    global _redis_client, _redis_disabled
    _redis_client = None
    _redis_disabled = False


def cache_get(key: str) -> dict[str, Any] | None:
    client = redis_client()
    if client:
        try:
            raw = client.get(key)
            return json.loads(raw) if raw else None
        except Exception:
            pass
    item = _memory_cache.get(key)
    if not item or item[0] <= time.monotonic():
        _memory_cache.pop(key, None)
        return None
    return json.loads(item[1])


def cache_set(key: str, value: dict[str, Any], ttl: int | None = None) -> None:
    serialized = json.dumps(value, ensure_ascii=False, default=str)
    client = redis_client()
    if client:
        try:
            client.setex(key, ttl or CACHE_TTL_SECONDS, serialized)
            return
        except Exception:
            pass
    _memory_cache[key] = (time.monotonic() + (ttl or CACHE_TTL_SECONDS), serialized)


def cache_clear(prefix: str | None = None) -> None:
    """Drop cached values, optionally only keys with a given prefix."""
    if prefix is None:
        _memory_cache.clear()
    else:
        for key in list(_memory_cache):
            if key.startswith(prefix):
                _memory_cache.pop(key, None)
    client = redis_client()
    if client:
        try:
            pattern = f"{prefix}*" if prefix else "*"
            keys = list(client.scan_iter(match=pattern, count=200))
            if keys:
                client.delete(*keys)
        except Exception:
            pass


def _rate_count(key: str) -> int:
    """Atomically increment AND expire the Redis counter; bound memory fallback."""
    client = redis_client()
    if client:
        try:
            return int(client.eval(_RATE_SCRIPT, 1, key, 70))
        except Exception:
            pass
    now = time.monotonic()
    with _rate_lock:
        while _memory_rate_limits:
            oldest, (expires, _) = next(iter(_memory_rate_limits.items()))
            if expires > now and len(_memory_rate_limits) < 10000:
                break
            _memory_rate_limits.pop(oldest)
        expires, count = _memory_rate_limits.get(key, (now + 70, 0))
        count = count + 1 if expires > now else 1
        _memory_rate_limits[key] = (expires if expires > now else now + 70, count)
        return count


def enforce_public_rate_limit(request: Request) -> None:
    """Per-endpoint quota plus shared IP protection, with verified JWT identity."""
    ip = request.client.host if request.client else "anonymous"
    identity = f"ip:{ip}"
    # Never trust arbitrary X-User-ID / forwarded-IP headers as a quota identity.
    if request.headers.get("authorization") or request.cookies.get("admin_session"):
        from .security import admin_from_token

        payload = admin_from_token(request, request.headers.get("authorization"), request.cookies.get("admin_session"))
        identity = "user:" + hashlib.sha256(str(payload["sub"]).encode()).hexdigest()
    bucket = int(time.time() // 60)
    path = request.url.path
    scope = "stream" if path.endswith("/chat/stream") else "recommend" if path.endswith("/recommendations") else "chat" if path.endswith("/chat/messages") else "feedback"
    limit = {"stream": config.RATE_LIMIT_STREAM_PER_MINUTE, "recommend": config.RATE_LIMIT_RECOMMEND_PER_MINUTE}.get(scope, RATE_LIMIT_PER_MINUTE)
    for key, quota in ((f"rate:v2:ip:{ip}:{bucket}", config.RATE_LIMIT_IP_PER_MINUTE), (f"rate:v2:{scope}:{identity}:{bucket}", limit)):
        if _rate_count(key) > max(1, quota):
            raise HTTPException(429, "请求过于频繁，请稍后再试", headers={"Retry-After": str(max(1, 60 - int(time.time() % 60)))})
