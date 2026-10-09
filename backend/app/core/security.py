"""Admin authentication: bcrypt password checks and HS256 bearer tokens."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import time
from collections import defaultdict
from typing import Any
from uuid import uuid4

import bcrypt
import jwt
from fastapi import Cookie, Header, HTTPException, Request
from sqlalchemy import text

from .config import ADMIN_LOGIN_RATE_LIMIT, ADMIN_LOGIN_WINDOW_SECONDS, ENVIRONMENT, JWT_SECRET
from .cache import redis_client
from .db import engine


_revoked_tokens: dict[str, float] = {}


def _revocation_key(token: str) -> str:
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return f"admin:revoked:{digest}"


def revoke_token(token: str) -> None:
    """Revoke a JWT until expiry, shared through Redis when available."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"], options={"verify_exp": False})
        expires_at = float(payload.get("exp") or 0)
    except jwt.PyJWTError:
        expires_at = 0
    ttl = max(1, int(expires_at - datetime.now(timezone.utc).timestamp())) if expires_at else 8 * 60 * 60
    key = _revocation_key(token)
    client = redis_client()
    if client:
        try:
            client.setex(key, ttl, "1")
            return
        except Exception:
            pass
    _revoked_tokens[key] = time.monotonic() + ttl


def is_token_revoked(token: str) -> bool:
    key = _revocation_key(token)
    client = redis_client()
    if client:
        try:
            return bool(client.get(key))
        except Exception:
            pass
    now = time.monotonic()
    expires_at = _revoked_tokens.get(key)
    if expires_at is None:
        return False
    if expires_at <= now:
        _revoked_tokens.pop(key, None)
        return False
    return True


def admin_from_token(
    request: Request,
    authorization: str | None = Header(default=None),
    admin_session: str | None = Cookie(default=None),
) -> dict[str, Any]:
    token = admin_session
    if not token and authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1]
    if not token:
        raise HTTPException(401, "需要管理员登录")
    if is_token_revoked(token):
        raise HTTPException(401, "admin session expired")
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        if not payload.get("sub"):
            raise HTTPException(401, "无效令牌")
        return payload
    except jwt.PyJWTError as exc:
        raise HTTPException(401, "无效令牌") from exc


_failed_logins: dict[str, list[float]] = defaultdict(list)


def check_login_rate_limit(identity: str) -> None:
    now = time.monotonic()
    values = [timestamp for timestamp in _failed_logins.get(identity, []) if now - timestamp < ADMIN_LOGIN_WINDOW_SECONDS]
    _failed_logins[identity] = values
    if len(values) >= ADMIN_LOGIN_RATE_LIMIT:
        raise HTTPException(429, "登录失败次数过多，请稍后再试")


def record_login_failure(identity: str) -> None:
    _failed_logins[identity].append(time.monotonic())


def reset_login_failures(identity: str) -> None:
    _failed_logins.pop(identity, None)


def authenticate(email: str, password: str) -> dict[str, str] | None:
    """Return a token payload for valid active credentials, else ``None``."""
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT email,password_hash FROM admin_users WHERE email=:email AND is_active=true"),
            {"email": email},
        ).first()
    if not row or not bcrypt.checkpw(password.encode(), row[1].encode()):
        return None
    token = jwt.encode(
        {
            "sub": row[0],
            "jti": str(uuid4()),
            "exp": datetime.now(timezone.utc) + timedelta(hours=8),
        },
        JWT_SECRET,
        algorithm="HS256",
    )
    return {"access_token": token, "token_type": "bearer"}
