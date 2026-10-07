"""Cache and rate limiting with deliberate Redis -> process-memory degradation.

Production uses Redis so that caches and counters are shared across workers; local
development without Redis must still behave correctly, never silently break.
"""

from __future__ import annotations

import json
import time
from typing import Any

from fastapi import HTTPException, Request

from .config import CACHE_TTL_SECONDS, RATE_LIMIT_PER_MINUTE, REDIS_URL

_redis_client: Any | None = None
_redis_disabled = False
_memory_cache: dict[str, tuple[float, str]] = {}
_memory_rate_limits: dict[str, list[float]] = {}


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


def enforce_public_rate_limit(request: Request) -> None:
    """Redis atomics when available, bounded process-local fallback otherwise."""
    identity = request.client.host if request.client else "anonymous"
    bucket = int(time.time() // 60)
    key = f"rate:{identity}:{bucket}"
    client = redis_client()
    if client:
        try:
            count = client.incr(key)
            if count == 1:
                client.expire(key, 70)
            if count > RATE_LIMIT_PER_MINUTE:
                raise HTTPException(429, "请求过于频繁，请稍后再试")
            return
        except HTTPException:
            raise
        except Exception:
            pass
    now = time.monotonic()
    visits = [timestamp for timestamp in _memory_rate_limits.get(identity, []) if now - timestamp < 60]
    if len(visits) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(429, "请求过于频繁，请稍后再试")
    visits.append(now)
    _memory_rate_limits[identity] = visits
