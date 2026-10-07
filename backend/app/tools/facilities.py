"""Facility tools (read-only).

The published Jiuzhaigou dataset links facilities to attractions through
``nearby_attraction_id`` and never publishes a toilet point list, so "toilets along
the way" can only ever be answered as a *nearest available* statement, never as a
promise. That limitation is surfaced in the tool output instead of being smoothed over.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..core.db import payload_rows
from .registry import tool

FACILITY_AGENTS = ("knowledge_agent", "route_agent", "realtime_agent", "feedback_agent")

#: Facility types that the dataset actually publishes.
PUBLISHED_TYPES = (
    "观光车站",
    "停车场",
    "购物点",
    "医院",
    "观景台",
    "游客服务中心",
    "检票口",
    "门禁",
    "医疗点",
    "警务点",
    "餐饮点",
    "吸烟区",
)


class GetFacilitiesArgs(BaseModel):
    facility_type: str | None = Field(default=None, max_length=30, description="设施类型，例如 观光车站、餐饮点、停车场、医疗点")
    keyword: str | None = Field(default=None, max_length=60, description="设施名称关键词")
    inside_park_only: bool = Field(default=True, description="是否只返回景区内部设施")
    limit: int = Field(default=20, ge=1, le=100)


class FindFacilitiesNearArgs(BaseModel):
    attraction_id: str = Field(min_length=1, max_length=64, description="景点 ID，例如 attr_021")
    facility_types: list[str] | None = Field(default=None, description="需要的设施类型；留空表示全部类型")
    limit: int = Field(default=10, ge=1, le=50)


def load_facilities() -> list[dict[str, Any]]:
    return payload_rows("facilities", 100)


def _slim(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "facility_id": item.get("facility_id"),
        "name": item.get("name"),
        "facility_type": item.get("facility_type"),
        "status": item.get("status"),
        "inside_park": item.get("inside_park"),
        "nearby_attraction_id": item.get("nearby_attraction_id"),
        "phone": item.get("phone"),
        "accessibility": item.get("accessibility"),
        "distance_to_gate_meters": item.get("distance_to_gate_meters"),
    }


@tool(
    name="get_facilities",
    description="按类型或关键词查询九寨沟景区设施（观光车站、停车场、餐饮点、医疗点等）。",
    args_schema=GetFacilitiesArgs,
    allowed=FACILITY_AGENTS,
    tags=("read", "catalog"),
)
def get_facilities(args: GetFacilitiesArgs) -> dict[str, Any]:
    items = load_facilities()
    if args.facility_type:
        items = [item for item in items if args.facility_type in str(item.get("facility_type") or "")]
    if args.keyword:
        items = [item for item in items if args.keyword in str(item.get("name") or "")]
    if args.inside_park_only:
        items = [item for item in items if item.get("inside_park", True)]
    return {
        "total": len(items),
        "published_types": list(PUBLISHED_TYPES),
        "items": [_slim(item) for item in items[: args.limit]],
    }


@tool(
    name="find_facilities_near",
    description="查询某个景点附近已公开的服务设施（依据数据集的 nearby_attraction_id 关联）。",
    args_schema=FindFacilitiesNearArgs,
    allowed=("route_agent", "feedback_agent", "knowledge_agent"),
    tags=("read", "catalog"),
)
def find_facilities_near(args: FindFacilitiesNearArgs) -> dict[str, Any]:
    wanted = set(args.facility_types or [])
    linked = [
        item
        for item in load_facilities()
        if (item.get("nearby_attraction_id") or item.get("related_attraction_id")) == args.attraction_id
        and (not wanted or item.get("facility_type") in wanted)
    ]
    result: dict[str, Any] = {
        "attraction_id": args.attraction_id,
        "total": len(linked),
        "items": [_slim(item) for item in linked[: args.limit]],
    }
    if not linked:
        # Be explicit rather than implying full coverage of the park.
        result["note"] = (
            "该景点附近设施未在公开资料中逐一公布；景区设施与景点的关联信息并不完整，"
            "不能据此断定沿线没有所需设施。"
            if not wanted
            else f"公开资料中未收录该景点附近的 {'、'.join(sorted(wanted))} 关联信息。"
        )
    return result
