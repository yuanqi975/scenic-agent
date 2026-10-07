"""Deterministic route algorithms.

The language model is allowed to *choose the parameters* and to *read the
validation report*, but it never invents an itinerary: every itinerary returned
here comes from this module. That split is what keeps planned routes truthful.

One Jiuzhaigou-specific subtlety drives the seasonal logic below: every attraction
record carries ``status == "open"`` all year round, and seasonal rotation closure is
expressed **only** through ``seasonal_closure``. Filtering on ``status`` alone would
happily route a visitor to a closed valley.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

#: Keep the route readable while allowing a real half-day route to cross several
#: map nodes. The previous ceiling of five stops made a 240-minute request look
#: incomplete even when the map contained enough open attractions.
MAX_ITINERARY_STOPS = 24

# The high Rize Valley tail above Panda Lake is remote from the lower viewpoints
# on the supplied park map. For a half-day budget there is not enough slack for the
# long return/transfer sequence; for older visitors it adds avoidable elevation and
# distance. Keep these attractions available to full-day general routes, but do not
# make them default stops for either case.
REMOTE_UPPER_RIZE_IDS = frozenset({
    "attr_028",  # Primitive Forest
    "attr_027",  # Fragrant Grass Lake
    "attr_026",  # Swan Lake
    "attr_024",  # Arrow Bamboo Lake
    "attr_025",  # Arrow Bamboo Lake Waterfall
    "attr_022",  # Panda Lake
    "attr_023",  # Panda Lake Waterfall
})
_OLDER_VISITOR_MARKERS = tuple(
    "\u8001\u4eba \u8001\u5e74 \u957f\u8f88 \u7236\u6bcd \u7238\u5988 \u7237\u7237 \u5976\u5976".split()
)


def includes_older_visitor(groups: list[str] | None) -> bool:
    """Whether the declared travel group includes an older visitor."""
    group_text = "".join(str(group) for group in (groups or []))
    return any(marker in group_text for marker in _OLDER_VISITOR_MARKERS)

# The map's standard sightseeing-bus spine, from the high Rize Valley down through
# Nuorilang and Shuzheng Valley. These ids are also present in routes.jsonl, but the
# explicit order keeps planning deterministic when a database has no route rows.
MAP_SPINES: tuple[tuple[str, ...], ...] = (
    (
        "attr_028", "attr_027", "attr_026", "attr_024", "attr_025", "attr_022",
        "attr_023", "attr_021", "attr_019", "attr_020", "attr_017", "attr_018",
        "attr_016", "attr_015", "attr_014", "attr_001", "attr_002", "attr_003",
        "attr_004", "attr_005", "attr_006", "attr_007", "attr_008", "attr_009",
        "attr_010", "attr_011", "attr_012", "attr_013",
    ),
    ("attr_031", "attr_032", "attr_035", "attr_034", "attr_033"),
)

DIFFICULTY_ORDER = {"轻松": 0, "中等": 1, "挑战": 2}

_MONTH_DAY = re.compile(r"(\d{1,2})\s*[-月]\s*(\d{1,2})")


def _duration(item: dict[str, Any]) -> int:
    """Planned visit length in minutes; unknown values are treated as zero."""
    try:
        return int(item.get("visit_duration_minutes") or 0)
    except (TypeError, ValueError):
        return 0


def _month_day(value: str) -> tuple[int, int] | None:
    match = _MONTH_DAY.search(value or "")
    if not match:
        return None
    month, day = int(match.group(1)), int(match.group(2))
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return None
    return month, day


def in_seasonal_closure(item: dict[str, Any], *, today: date | None = None) -> bool:
    """Is this attraction inside its published seasonal rotation closure window?

    ``seasonal_closure.range`` is written as ``"11-16 至次年 03-31"``, i.e. it wraps
    the new year, so the comparison has to wrap too. When the range cannot be parsed
    we assume the closure applies: refusing to promise a closed valley is the safer
    failure.
    """
    closure = item.get("seasonal_closure")
    if not closure:
        return False
    if isinstance(closure, str):
        raw = closure
    elif isinstance(closure, dict):
        raw = str(closure.get("range") or closure.get("reason") or "")
    else:
        return True
    if not raw:
        return True

    bounds = _MONTH_DAY.findall(raw)
    if len(bounds) < 2:
        return True
    try:
        start = (int(bounds[0][0]), int(bounds[0][1]))
        end = (int(bounds[1][0]), int(bounds[1][1]))
    except (TypeError, ValueError, IndexError):
        return True

    now = today or datetime.now().date()
    current = (now.month, now.day)
    if start <= end:
        return start <= current <= end
    return current >= start or current <= end


def eligible_attractions(
    attractions: list[dict[str, Any]],
    *,
    duration_minutes: int,
    groups: list[str] | None = None,
    preferences: list[str] | None = None,
    exclude_attraction_ids: list[str] | None = None,
    ignore_group_filter: bool = False,
    today: date | None = None,
    respect_seasonal_closure: bool = True,
    difficulty: str | None = None,
) -> list[dict[str, Any]]:
    """Apply the open / closure / itinerary / duration / audience / category filters.

    The order matters and is part of the contract: audience and preference filters
    fall back to the wider set rather than emptying the result, mirroring the
    original behaviour so that a narrow request never returns nothing.

    ``respect_seasonal_closure`` defaults to true because ``status`` stays ``"open"``
    year-round in this dataset; without the extra check a winter request would be
    routed into a rotation-closed valley.
    """
    excluded = set(exclude_attraction_ids or [])
    selected = [
        item
        for item in attractions
        if item.get("status") == "open"
        and item.get("in_standard_tour", True)
        and _duration(item) <= duration_minutes
        and (item.get("attraction_id") not in excluded)
        and not (respect_seasonal_closure and in_seasonal_closure(item, today=today))
        and (
            difficulty is None
            or DIFFICULTY_ORDER.get(item.get("difficulty", "轻松"), 0)
            <= DIFFICULTY_ORDER.get(difficulty, 99)
        )
    ]
    if includes_older_visitor(groups) or duration_minutes <= 240:
        selected = [
            item for item in selected
            if str(item.get("attraction_id") or "") not in REMOTE_UPPER_RIZE_IDS
        ]
    if groups and not ignore_group_filter:
        group_selected = [
            item
            for item in selected
            if any(group in (item.get("suitable_for") or []) for group in groups)
        ]
        # Audience labels are advisory. A narrow label such as ``老年游客`` only
        # appears on a few catalog records; treating it as a hard filter made a
        # half-day request collapse to one lake. Keep the full open map spine when
        # the audience-matched records cannot cover at least half of the budget.
        group_minutes = sum(_duration(item) for item in group_selected)
        selected = group_selected if group_minutes >= max(30, int(duration_minutes * 0.5)) else selected
    if preferences:
        preferred = [item for item in selected if item.get("category") in preferences]
        if preferred:
            # Preferences guide the first part of the route. They are not a hard
            # filter for a full park itinerary: the map needs connected waterfalls,
            # forests and transfer nodes to fill a half-day without teleporting
            # between isolated lakes.
            remainder = [item for item in selected if item not in preferred]
            # A preference ranks the first stops; it never removes connected
            # waterfalls, forests, villages and transfer nodes from a full route.
            selected = preferred + remainder
    return selected


def greedy_itinerary(
    attractions: list[dict[str, Any]],
    *,
    duration_minutes: int,
    required_facilities: list[str] | None = None,
    today: date | None = None,
    routes: list[dict[str, Any]] | None = None,
    **filters: Any,
) -> dict[str, Any]:
    """Fill the time budget with eligible attractions, nearest-first within budget.

    ``today`` is passed down to :func:`eligible_attractions` so the seasonal filter is
    reproducible. Left unset it means "the real current date", which is fine for a
    visitor-facing plan but makes the *output* depend on when it is called - so any
    caller that also validates the plan must pass the same date to both.
    """
    selected = eligible_attractions(
        attractions, duration_minutes=duration_minutes, today=today, **filters
    )

    # ``difficulty=轻松`` is a useful first pass, but it should not resurrect the
    # old "40 minutes for a 4-hour request" failure. For the full park catalog,
    # broaden the audience/difficulty ranking only when the first pass cannot cover
    # half the requested time. The returned stops still carry their real difficulty
    # so the UI can disclose the trade-off.
    first_pass_minutes = sum(_duration(item) for item in selected)
    if len(attractions) >= 8 and first_pass_minutes < max(30, int(duration_minutes * 0.5)):
        broadened = dict(filters)
        broadened["difficulty"] = None
        broadened["ignore_group_filter"] = True
        wider = eligible_attractions(
            attractions, duration_minutes=duration_minutes, today=today, **broadened
        )
        if sum(_duration(item) for item in wider) > first_pass_minutes:
            selected = wider
    by_id = {str(item.get("attraction_id")): item for item in selected}

    # Use the map spine whenever the catalog is the full Jiuzhaigou dataset. A
    # preference is a ranking hint, not a hard valley filter: visitors asking for
    # lakes should still receive the connected waterfall and forest nodes needed for
    # a truthful route. Small injected test catalogs retain the old stable ordering.
    map_order: list[str] = []
    if len(by_id) >= 8:
        preference_set = set(filters.get("preferences") or [])
        spines = list(MAP_SPINES)
        if preference_set:
            def spine_score(spine: tuple[str, ...]) -> int:
                return sum(1 for attraction_id in spine if by_id.get(attraction_id, {}).get("category") in preference_set)
            spines.sort(key=spine_score, reverse=True)
        for spine in spines:
            map_order.extend(attraction_id for attraction_id in spine if attraction_id in by_id)
        map_order.extend(attraction_id for attraction_id in by_id if attraction_id not in map_order)
        selected = [by_id[attraction_id] for attraction_id in map_order]

    total = 0
    itinerary: list[dict[str, Any]] = []
    for item in selected:
        duration = _duration(item) or 60
        if total + duration > duration_minutes:
            continue
        itinerary.append(item)
        total += duration
        if len(itinerary) >= MAX_ITINERARY_STOPS:
            break
    route_lookup = {
        (str(route.get("start_id")), str(route.get("end_id"))): route
        for route in (routes or [])
        if route.get("start_id") and route.get("end_id")
    }
    route_segments: list[dict[str, Any]] = []
    for previous, current in zip(itinerary, itinerary[1:]):
        route = route_lookup.get((str(previous.get("attraction_id")), str(current.get("attraction_id"))))
        if not route:
            continue
        route_segments.append(
            {
                "route_id": route.get("route_id"),
                "from": previous.get("name"),
                "to": current.get("name"),
                "route_type": route.get("route_type", "步行栈道"),
                "estimated_minutes": route.get("estimated_minutes"),
                "distance_meters": route.get("distance_meters"),
                "difficulty": route.get("difficulty"),
                "accessible": route.get("accessible"),
            }
        )
    return {
        "attractions": itinerary,
        "total_minutes": total,
        "candidates_considered": len(selected),
        "required_facilities": list(required_facilities or []),
        "route_segments": route_segments,
        "map_basis": "九寨沟三沟地图顺序：日则沟高处，经诺日朗中心换乘，再到树正沟；则查洼沟作为长海—五彩池支线。",
    }


def _mentioned_names(message: str, items: list[dict[str, Any]], key: str) -> list[str]:
    found: list[str] = []
    for item in items:
        name = item.get("name") or ""
        if name and name in message:
            identifier = item.get(key)
            if identifier:
                found.append(identifier)
    return list(dict.fromkeys(found))


def mentioned_entity_ids(
    message: str,
    attractions: list[dict[str, Any]],
    facilities: list[dict[str, Any]],
) -> list[str]:
    """Return attraction/facility ids whose name literally appears in the question.

    Chinese has no word delimiters, so PostgreSQL ``simple`` full-text search cannot
    match questions like「五花海怎么去？」. Matching known POI names against the
    question first keeps retrieval accurate without a tokenizer.
    """
    return _mentioned_names(message, attractions, "attraction_id") + _mentioned_names(
        message, facilities, "facility_id"
    )


def facilities_for_attraction(
    facilities: list[dict[str, Any]],
    attraction_id: str,
    facility_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Facilities linked to one attraction, optionally filtered by type.

    The Jiuzhaigou dataset links facilities with ``nearby_attraction_id``; the older
    ``related_attraction_id`` key is still honoured so runtime-created records work.
    """
    wanted = set(facility_types or [])
    result = []
    for facility in facilities:
        linked = facility.get("nearby_attraction_id") or facility.get("related_attraction_id")
        if linked != attraction_id:
            continue
        if wanted and facility.get("facility_type") not in wanted:
            continue
        result.append(facility)
    return result


def validate_route(
    plan: dict[str, Any],
    *,
    duration_minutes: int,
    required_facilities: list[str] | None = None,
    difficulty: str | None = None,
    facilities: list[dict[str, Any]] | None = None,
    groups: list[str] | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    """Check an itinerary against every declared constraint.

    Returns a machine-readable report so the route agent can *see* what is wrong and
    plan again, instead of hallucinating a fix.
    """
    attract = plan.get("attractions") or []
    total = int(plan.get("total_minutes") or 0)
    violations: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    if not attract:
        violations.append({"rule": "empty_itinerary", "detail": "没有可用的开放景点"})
    if total > duration_minutes:
        violations.append(
            {
                "rule": "duration_exceeded",
                "detail": f"总时长 {total} 分钟超过预算 {duration_minutes} 分钟",
                "over_minutes": total - duration_minutes,
            }
        )
    for item in attract:
        if item.get("status") != "open":
            violations.append(
                {"rule": "closed_attraction", "detail": f"{item.get('name')} 当前非开放状态"}
            )
        if item.get("in_standard_tour") is False:
            violations.append(
                {"rule": "off_itinerary", "detail": f"{item.get('name')} 不在常规观光车线路内"}
            )
        # Two distinct modes, and the difference matters:
        #
        # * with an explicit ``today`` the check is exact - it asks whether *that date*
        #   falls inside the published window, so a route planned for a summer visit is
        #   not reported as invalid merely because the attraction closes in winter;
        # * without a date it is a conservative safety check that surfaces any published
        #   seasonal restriction at all.
        #
        # Running the conservative branch against a date-aware plan is what produced a
        # route that excluded a seasonally closed attraction while its own validation
        # still claimed a violation.
        if in_seasonal_closure(item, today=today) or (today is None and item.get("seasonal_closure")):
            reason = item.get("seasonal_closure")
            if isinstance(reason, dict):
                reason = reason.get("reason") or reason.get("range")
            violations.append(
                {
                    "rule": "seasonal_closure",
                    "detail": f"{item.get('name')} 当前处于季节性轮休保育期，不对外开放（{reason or '以景区公告为准'}）",
                }
            )
        if difficulty and DIFFICULTY_ORDER.get(item.get("difficulty", "轻松"), 0) > DIFFICULTY_ORDER.get(difficulty, 99):
            violations.append(
                {
                    "rule": "difficulty_exceeded",
                    "detail": f"{item.get('name')} 难度 {item.get('difficulty')} 高于要求的 {difficulty}",
                }
            )
        if groups and not any(group in (item.get("suitable_for") or []) for group in groups):
            warnings.append(
                {"rule": "audience_mismatch", "detail": f"{item.get('name')} 未标注适合 {'、'.join(groups)}"}
            )

    coverage: dict[str, list[str]] = {}
    for facility_type in required_facilities or []:
        covered: list[str] = []
        for item in attract:
            for facility in facilities_for_attraction(facilities or [], item.get("attraction_id"), [facility_type]):
                covered.append(facility.get("facility_id") or facility.get("name"))
        coverage[facility_type] = covered
        if not covered:
            # Facility-to-attraction links are only published for part of the park,
            # so this is a soft constraint the agent must disclose, not a hard failure.
            warnings.append(
                {
                    "rule": "facility_coverage_unverified",
                    "detail": f"未能确认路线沿线有{ facility_type }（该点位未公开关联信息）",
                    "facility_type": facility_type,
                }
            )

    return {
        "valid": not violations,
        "violations": violations,
        "warnings": warnings,
        "facility_coverage": coverage,
        "total_minutes": total,
        "duration_minutes": duration_minutes,
        "stops": len(attract),
    }


def render_itinerary(plan: dict[str, Any], message: str) -> str:
    """Human-readable itinerary used by the deterministic (fallback) answer path."""
    names = "、".join(item.get("name", "") for item in plan.get("attractions") or [])
    return f"{message}\n\n推荐：{names}" if names else message
