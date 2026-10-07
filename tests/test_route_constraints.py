"""Route constraint regression baseline.

These tests exist because the previous failure mode was invisible: an itinerary that
still contained 「原始森林」 during its seasonal rotation closure. Two properties have to
hold, and only the second one is interesting:

1. a route planned for a date inside the closure window must exclude the attraction;
2. a route planned for a date **outside** the window may include it, and its own
   validation must agree.

Property 2 is what was broken. Planning is date-aware while a validation call without a
date is deliberately conservative ("any published closure is a violation"), so the same
plan could be reported as both valid and invalid depending on nothing but timing. A
test that only asserts property 1 passes for free half the year and pins nothing, which
is exactly why every date here is explicit.
"""

from datetime import date

from tests.conftest import async_test

#: Inside 「11-16 至次年 03-31」 for 原始森林.
WINTER = date(2026, 1, 15)
#: Outside that window.
SUMMER = date(2026, 7, 15)

CLOSED_IN_WINTER = "原始森林"
OFF_STANDARD_ROUTE = "扎依扎嘎神山"


async def _plan(arguments: dict) -> dict:
    from app.tools.registry import TOOL_REGISTRY

    outcome = await TOOL_REGISTRY.execute(
        agent="route_agent", tool_name="calculate_route", arguments=arguments
    )
    assert outcome.ok, outcome.error
    return outcome.result


async def _validate(arguments: dict) -> dict:
    from app.tools.registry import TOOL_REGISTRY

    outcome = await TOOL_REGISTRY.execute(
        agent="route_agent", tool_name="validate_route", arguments=arguments
    )
    assert outcome.ok, outcome.error
    return outcome.result


@async_test
async def test_route_filters_seasonal_closure_and_off_standard_route(fake_catalog):
    """A winter plan never promises a rotation-closed valley."""
    plan = await _plan({"query": "带老人玩半天", "travel_date": WINTER.isoformat()})
    names = {item["name"] for item in plan["attractions"]}
    assert CLOSED_IN_WINTER not in names
    assert OFF_STANDARD_ROUTE not in names
    assert plan["attractions"], "冬天也应至少给出一条可用路线"


@async_test
async def test_planning_and_validation_agree_on_the_same_date(fake_catalog):
    """The plan's own validation must not contradict the plan.

    Both directions are asserted, because a validator that always says "valid" would
    pass a one-sided test.
    """
    plan = await _plan({"query": "带老人玩半天", "travel_date": SUMMER.isoformat()})
    report = await _validate({"plan": plan, "duration_minutes": 240})

    assert report["travel_date"] == SUMMER.isoformat()
    assert report["valid"] is True, report["violations"]
    assert not [item for item in report["violations"] if item["rule"] == "seasonal_closure"]
    assert report["total_minutes"] <= report["duration_minutes"]


@async_test
async def test_validation_without_a_date_stays_conservative(fake_catalog):
    """The conservative mode is a safety net, so it must still fire.

    A plan that *does* contain a seasonally restricted attraction has to be reported
    when no date is supplied - that is the behaviour that protects a caller which
    forgot to pass one.
    """
    from app.services.route_algo import validate_route

    plan = {
        "duration_minutes": 240,
        "total_minutes": 40,
        "attractions": [
            {
                "attraction_id": "attr_028",
                "name": CLOSED_IN_WINTER,
                "status": "open",
                "in_standard_tour": True,
                "seasonal_closure": {"range": "11-16 至次年 03-31", "reason": "季节性轮休保育"},
            }
        ],
    }
    report = validate_route(plan, duration_minutes=240)
    assert report["valid"] is False
    assert {item["rule"] for item in report["violations"]} == {"seasonal_closure"}

    # ...but the same plan validated for a summer date is fine.
    summer = validate_route(plan, duration_minutes=240, today=SUMMER)
    assert summer["valid"] is True, summer["violations"]
