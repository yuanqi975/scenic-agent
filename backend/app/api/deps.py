"""Shared FastAPI dependencies."""

from __future__ import annotations

from fastapi import Header, HTTPException

from ..core.security import admin_from_token  # noqa: F401  (re-exported for routes)

__all__ = ["admin_from_token"]
