"""Attraction catalog tools (read-only)."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field

from ..core.db import payload_rows
from .registry import tool

KNOWLEDGE_AGENTS = ("knowledge_agent", "route_agent", "realtime_agent", "feedback_agent")


class SearchAttractionsArgs(BaseModel):
    keyword: str | None = Field(default=None, max_length=60, description="景点名称或关键词，例如「五花海」")
    valley: str | None = Field(default=None, max_length=20, description="沟名，例如 日则沟 / 树正沟 / 则查洼沟 / 扎如沟")
    category: str | None = Field(default=None, max_length=30, description="景点类型，例如 湖泊/海子、瀑布、藏寨人文")
    groups: list[str] | None = Field(default=None, description="同行人群，例如 老年游客、亲子家庭")
    max_duration_minutes: int | None = Field(default=None, ge=1, le=1440, description="单个景点最长可接受游览时长")
    open_only: bool = Field(default=True, description="是否只返回当前开放、且在本季常规线路内的景点")
    limit: int = Field(default=20, ge=1, le=100)


class AttractionDetailArgs(BaseModel):
    attraction_id: str = Field(min_length=1, max_length=64, description="景点 ID，例如 attr_021")


def load_attractions() -> list[dict[str, Any]]:
    return payload_rows("attractions", 100)


def _matches(item: dict[str, Any], query: str) -> bool:
    haystack = json.dumps(item, ensure_ascii=False).lower()
    return query.lower() in haystack


@tool(
    name="search_attractions",
    description="按关键词、沟名、类型、人群与时长筛选九寨沟景点。返回结构化的景点列表（含开放时间、时长、票价说明）。",
    args_schema=SearchAttractionsArgs,
    allowed=KNOWLEDGE_AGENTS,
    tags=("read", "catalog"),
)
def search_attractions(args: SearchAttractionsArgs) -> dict[str, Any]:
    items = load_attractions()
    if args.keyword:
        items = [item for item in items if _matches(item, args.keyword)]
    if args.valley:
        items = [item for item in items if args.valley in str(item.get("valley") or "")]
    if args.category:
        items = [item for item in items if args.category in str(item.get("category") or "")]
    if args.groups:
        items = [
            item
            for item in items
            if any(group in (item.get("suitable_for") or []) for group in args.groups or [])
        ]
    if args.max_duration_minutes:
        items = [
            item
            for item in items
            if int(item.get("visit_duration_minutes") or 0) <= args.max_duration_minutes
        ]
    if args.open_only:
        items = [
            item
            for item in items
            if item.get("status") == "open" and item.get("in_standard_tour", True)
        ]
    return {
        "total": len(items),
        "items": [
            {
                "attraction_id": item.get("attraction_id"),
                "name": item.get("name"),
                "valley": item.get("valley"),
                "category": item.get("category"),
                "visit_duration_minutes": item.get("visit_duration_minutes"),
                "difficulty": item.get("difficulty"),
                "suitable_for": item.get("suitable_for"),
                "opening_hours": item.get("opening_hours"),
                "ticket_note": item.get("ticket_note"),
                "elevation_meters": item.get("elevation_meters"),
                "status": item.get("status"),
                "seasonal_closure": item.get("seasonal_closure"),
            }
            for item in items[: args.limit]
        ],
    }


@tool(
    name="get_attraction_detail",
    description="读取单个九寨沟景点的完整公开字段（含注意事项、海拔、来源标记）。",
    args_schema=AttractionDetailArgs,
    allowed=("knowledge_agent", "route_agent", "realtime_agent"),
    tags=("read", "catalog"),
)
def get_attraction_detail(args: AttractionDetailArgs) -> dict[str, Any]:
    for item in load_attractions():
        if item.get("attraction_id") == args.attraction_id:
            return {"found": True, "attraction": item}
    return {"found": False, "attraction_id": args.attraction_id, "message": "未找到该景点"}
