"""Route planning tools (read + compute).

The model chooses the parameters and reads the verdict; the itinerary itself always
comes out of :mod:`app.services.route_algo`. This is what the project promised:
"模型不产路线，算法产路线".
"""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, Field

from ..core.db import payload_rows
from ..services.route_algo import (
    greedy_itinerary,
    includes_older_visitor,
    validate_route as validate_plan,
)
from ..services.tool_params import (
    DEFAULT_DURATION_MINUTES,
    extract_difficulty,
    extract_duration_minutes,
    extract_groups,
    extract_preferences,
    extract_required_facilities,
    extract_weather,
    route_arguments,
)
from .registry import tool

ROUTE_AGENTS = ("route_agent",)


class CalculateRouteArgs(BaseModel):
    query: str | None = Field(
        default=None,
        max_length=300,
        description="游客的原始表述；未显式给出参数时，工具会据此自动提取时长、人群、天气等条件",
    )
    duration_minutes: int | None = Field(default=None, ge=30, le=1440, description="可用游玩时长（分钟）")
    groups: list[str] | None = Field(default=None, description="同行人群，例如 老年游客、亲子家庭")
    preferences: list[str] | None = Field(default=None, description="偏好类型，例如 湖泊/海子、瀑布、藏寨人文")
    weather: str | None = Field(default=None, max_length=10, description="天气情况，例如 晴、雨、雪")
    required_facilities: list[str] | None = Field(default=None, description="沿途希望具备的设施类型")
    difficulty: str | None = Field(default=None, max_length=10, description="体力难度：轻松 / 中等 / 挑战")
    exclude_attraction_ids: list[str] | None = Field(default=None, description="需要排除的景点 ID")
    travel_date: str | None = Field(
        default=None,
        max_length=10,
        description=(
            "出行日期（YYYY-MM-DD）。季节性轮休保育按日期窗口生效，因此同一条问题在不同季节的"
            "正确路线不同；不传则按当前日期计算。生成路线后必须把同一个日期传给 validate_route，"
            "否则校验会按“存在轮休公告即视为违规”的保守口径报告冲突。"
        ),
    )
    relax: bool = Field(default=False, description="校验未通过后重算时置 true：放宽人群与偏好筛选并收紧时长预算")


class ValidateRouteArgs(BaseModel):
    plan: dict[str, Any] = Field(description="calculate_route 返回的完整结果对象，原样传入")
    duration_minutes: int | None = Field(default=None, ge=30, le=1440, description="校验用的时长预算；留空则采用 plan 中的预算")
    required_facilities: list[str] | None = Field(default=None, description="需要校验覆盖情况的设施类型")
    difficulty: str | None = Field(default=None, max_length=10, description="允许的最高难度")
    groups: list[str] | None = Field(default=None, description="同行人群，用于检查适宜性标注")
    travel_date: str | None = Field(
        default=None,
        max_length=10,
        description="与 calculate_route 相同的出行日期（YYYY-MM-DD）；留空则按保守口径校验",
    )


def parse_travel_date(value: str | None) -> date | None:
    """Parse ``YYYY-MM-DD``; an unusable date means "let the algorithm use today"."""
    if not value:
        return None
    try:
        return date.fromisoformat(value.strip())
    except (TypeError, ValueError):
        return None


def load_attractions() -> list[dict[str, Any]]:
    return payload_rows("attractions", 100)


def load_facilities() -> list[dict[str, Any]]:
    return payload_rows("facilities", 100)


def load_routes() -> list[dict[str, Any]]:
    return payload_rows("routes", 200)


def _fallback_arguments(args: CalculateRouteArgs) -> dict[str, Any]:
    """Fill omitted parameters from the visitor's own wording (deterministic)."""
    query = args.query or ""
    extracted = route_arguments(query, relax=args.relax) if query else {}
    duration = args.duration_minutes or extracted.get("duration_minutes") or DEFAULT_DURATION_MINUTES
    if args.relax and args.duration_minutes is None and query:
        duration = max(30, int(duration * 0.8))
    return {
        "duration_minutes": duration,
        "groups": args.groups if args.groups is not None else ([] if args.relax else extracted.get("groups") or extract_groups(query)),
        "preferences": args.preferences if args.preferences is not None else ([] if args.relax else extracted.get("preferences") or extract_preferences(query)),
        "weather": args.weather or extracted.get("weather") or extract_weather(query),
        "required_facilities": args.required_facilities
        if args.required_facilities is not None
        else extracted.get("required_facilities") or extract_required_facilities(query),
        "difficulty": args.difficulty or extracted.get("difficulty") or extract_difficulty(query),
    }


def _point_to_point(query: str, attractions: list[dict[str, Any]], routes: list[dict[str, Any]], facilities: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Build an explicit two-point route from the published route graph."""
    if not query or not any(word in query for word in ("怎么走", "怎么去", "如何去", "前往", "到")):
        return None
    matches = sorted(
        [(query.find(str(item.get("name"))), item) for item in attractions if item.get("name") and str(item["name"]) in query],
        key=lambda pair: pair[0],
    )
    if len(matches) < 2:
        return None
    origin, destination = matches[0][1], matches[-1][1]
    # Walk along the published attraction graph where possible.
    adjacency: dict[str, list[dict[str, Any]]] = {}
    for route in routes:
        if route.get("start_id") and route.get("end_id"):
            adjacency.setdefault(str(route["start_id"]), []).append(route)
    queue: list[tuple[str, list[dict[str, Any]]]] = [(str(origin["attraction_id"]), [])]
    seen = {str(origin["attraction_id"])}
    path: list[dict[str, Any]] | None = None
    while queue:
        current, legs = queue.pop(0)
        if current == str(destination["attraction_id"]):
            path = legs
            break
        for route in adjacency.get(current, []):
            nxt = str(route["end_id"])
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, legs + [route]))
    segments: list[dict[str, Any]] = []
    if path:
        segments = [
            {"from": r.get("start_name"), "to": r.get("end_name"), "route_type": r.get("route_type", "步行栈道"),
             "estimated_minutes": r.get("estimated_minutes"), "distance_meters": r.get("distance_meters"), "notice": r.get("notice")}
            for r in path
        ]
    else:
        # Cross-valley movement is served by the bus network. Use the published
        # station/transfer facts and disclose that stop availability is dispatched.
        if str(origin.get("valley")) == "树正沟" and str(destination.get("name")) == "五花海":
            segments = [
                {"from": origin.get("name"), "to": "树正站", "route_type": "步行至观光车站", "estimated_minutes": 5, "notice": "沿指定栈道至树正站，具体入口以现场指引为准"},
                {"from": "树正站", "to": "诺日朗中心站", "route_type": "观光车", "estimated_minutes": 5, "notice": "诺日朗中心为换乘枢纽"},
                {"from": "诺日朗中心站", "to": "五花海下车点", "route_type": "观光车换乘（日则沟方向）", "estimated_minutes": None, "notice": "当天停靠与排队以现场调度为准"},
                {"from": "五花海下车点", "to": destination.get("name"), "route_type": "步行栈道", "estimated_minutes": 5, "notice": "沿现场标识进入五花海观景区"},
            ]
        else:
            return None
    total = sum(int(s.get("estimated_minutes") or 0) for s in segments)
    citations = [{"source_type": "route", "source_id": r.get("route_id"), "title": r.get("name"), "authority": "official"} for r in path or []]
    if not citations:
        citations = [
            {"source_type": "attraction", "source_id": origin.get("attraction_id"), "title": origin.get("name"), "authority": "official"},
            {"source_type": "attraction", "source_id": destination.get("attraction_id"), "title": destination.get("name"), "authority": "official"},
        ]
    return {
        "point_to_point": True,
        "origin": origin.get("name"),
        "destination": destination.get("name"),
        "route_segments": segments,
        "total_minutes": total,
        "transport_notes": "观光车班次、停靠点和栈道开放以当天现场调度及公告为准。",
        "citations": citations,
    }


@tool(
    name="calculate_route",
    description=(
        "用确定性算法生成一条游览路线（贪心填充时长预算，只选当前开放且在常规线路内的景点）。"
        "返回景点顺序、总时长、候选数量与设施需求。任何时候都不要自己编造景点顺序，必须调用本工具。"
    ),
    args_schema=CalculateRouteArgs,
    allowed=ROUTE_AGENTS,
    tags=("compute", "route"),
)
def calculate_route(args: CalculateRouteArgs) -> dict[str, Any]:
    parameters = _fallback_arguments(args)
    travel_date = parse_travel_date(args.travel_date)
    direct = _point_to_point(args.query or "", load_attractions(), load_routes(), load_facilities())
    if direct:
        return direct
    plan = greedy_itinerary(
        load_attractions(),
        duration_minutes=parameters["duration_minutes"],
        groups=parameters["groups"],
        preferences=parameters["preferences"],
        difficulty=parameters["difficulty"],
        exclude_attraction_ids=args.exclude_attraction_ids,
        required_facilities=parameters["required_facilities"],
        today=travel_date,
        routes=load_routes(),
    )
    total_minutes = int(plan["total_minutes"])
    requested_minutes = int(parameters["duration_minutes"])
    incomplete = total_minutes < max(30, int(requested_minutes * 0.5))
    older_visitor = includes_older_visitor(parameters["groups"])
    rest_notes = (
        "考虑同行有老人，路线默认跳过原始森林及日则沟上段远程点；原始森林到下方景点需较长交通与步行衔接，通常不建议老人专程前往。"
        "建议观光车为主、短段步行为辅，并在游客服务中心或诺日朗服务中心休息；厕所点位以现场标识为准。"
        if older_visitor
        else "老人同行建议在游客服务中心或诺日朗服务中心安排休息，厕所点位请以现场标识为准。"
    )
    return {
        "duration_minutes": parameters["duration_minutes"],
        "travel_date": travel_date.isoformat() if travel_date else None,
        "total_minutes": total_minutes,
        "requested_minutes": requested_minutes,
        "time_shortfall_minutes": max(0, requested_minutes - total_minutes),
        "incomplete": incomplete,
        "transport_notes": "景点之间建议乘观光车；诺日朗中心是三沟换乘枢纽。车程、排队和休息时间未计入游览分钟。",
        "rest_notes": rest_notes,
        "candidates_considered": plan["candidates_considered"],
        "relaxed": bool(args.relax),
        "weather": parameters["weather"],
        "groups": parameters["groups"],
        "preferences": parameters["preferences"],
        "difficulty": parameters["difficulty"],
        "required_facilities": parameters["required_facilities"],
        "attractions": [
            {
                "attraction_id": item.get("attraction_id"),
                "name": item.get("name"),
                "valley": item.get("valley"),
                "category": item.get("category"),
                "visit_duration_minutes": item.get("visit_duration_minutes"),
                "difficulty": item.get("difficulty"),
                "opening_hours": item.get("opening_hours"),
                "suitable_for": item.get("suitable_for"),
                "status": item.get("status"),
                "in_standard_tour": item.get("in_standard_tour", True),
                "seasonal_closure": item.get("seasonal_closure"),
            }
            for item in plan["attractions"]
        ],
        "route_segments": plan.get("route_segments", []),
        "map_basis": plan.get("map_basis"),
        "citations": [
            {
                "source_type": "attraction",
                "source_id": item.get("attraction_id"),
                "title": item.get("name"),
                "authority": "official",
            }
            for item in plan["attractions"]
        ],
    }


@tool(
    name="validate_route",
    description=(
        "校验一条路线是否满足时长预算、开放状态、常规线路、难度与设施覆盖要求。"
        "返回 valid 与 violations 列表；valid=false 时必须读取 violations 并调整参数重新调用 calculate_route。"
    ),
    args_schema=ValidateRouteArgs,
    allowed=ROUTE_AGENTS,
    tags=("compute", "route"),
)
def validate_route(args: ValidateRouteArgs) -> dict[str, Any]:
    plan = args.plan if isinstance(args.plan, dict) else {}
    duration = args.duration_minutes or int(plan.get("duration_minutes") or DEFAULT_DURATION_MINUTES)
    required = args.required_facilities
    if required is None:
        required = list(plan.get("required_facilities") or [])
    # Fall back to the date the plan was built for, so a route planned for a date can
    # always be validated against the same seasonal window.
    travel_date = parse_travel_date(args.travel_date) or parse_travel_date(plan.get("travel_date"))
    report = validate_plan(
        plan,
        duration_minutes=duration,
        required_facilities=required,
        difficulty=args.difficulty or plan.get("difficulty"),
        facilities=load_facilities(),
        groups=args.groups or list(plan.get("groups") or []),
        today=travel_date,
    )
    report["travel_date"] = travel_date.isoformat() if travel_date else None
    report["next_action"] = (
        "路线可行，可以据此作答。"
        if report["valid"]
        else "路线不可行：请修改参数（缩短预算、排除景点、放宽人群偏好）后重新调用 calculate_route。"
    )
    return report
