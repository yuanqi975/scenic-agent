"""Deterministic parameter extraction from visitor wording.

Two callers need exactly the same mapping, so it lives in one place:

* the **rule brain** (``LLM_MODE=fallback``) uses it to build tool arguments;
* the **route tool** uses it to fill defaults the model omitted.

Keeping it here - instead of inside a tool - is what lets the routing tools stay
independent of the agent runtime.
"""

from __future__ import annotations

import re
from typing import Any

#: Visitor wording -> the ``suitable_for`` values actually published in the dataset.
GROUP_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"老人|老年|长辈|父母|爸妈|爷爷|奶奶", "老年游客"),
    (r"孩子|儿童|小朋友|亲子|带娃|宝宝|婴儿", "亲子家庭"),
    (r"摄影|拍照|出片|机位", "摄影爱好者"),
    (r"徒步|户外|登山|露营", "户外爱好者"),
)

#: Visitor wording -> ``category`` values in the dataset.
PREFERENCE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"湖泊|海子|水景|彩林|倒影", "湖泊/海子"),
    (r"瀑布|诺日朗", "瀑布"),
    (r"藏寨|人文|村寨|寺庙|文化", "藏寨人文"),
    (r"森林|树正", "森林"),
    (r"地质|岩体|钙华|地貌", "岩体地质"),
    (r"山|峰|雪山", "山峰/神山"),
)

#: Visitor wording -> the ``facility_type`` values in the dataset.
FACILITY_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"卫生间|厕所|洗手间|如厕|方便一下", "观光车站"),  # no published toilet list; nearest proxy
    (r"吃饭|餐厅|餐饮|午饭|吃饭的地方", "餐饮点"),
    (r"停车|自驾|车位", "停车场"),
    (r"观光车|换乘|坐车|车站", "观光车站"),
    (r"医院|医务|医疗|高反|看病", "医疗点"),
    (r"警察|警务|求助", "警务点"),
    (r"服务中心|咨询|寄存", "游客服务中心"),
    (r"购物|纪念品|特产", "购物点"),
    (r"吸烟", "吸烟区"),
    (r"观景台", "观景台"),
)

WEATHER_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"雨|下雨|降雨|阴雨", "雨"),
    (r"雪|下雪|结冰", "雪"),
    (r"晴|晴天|太阳", "晴"),
    (r"阴|多云", "阴"),
)

DIFFICULTY_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"轻松|不累|省力|体力不好|腿脚不便|老人|长辈", "轻松"),
    (r"挑战|高强度|徒步穿越", "挑战"),
    (r"中等|有点累", "中等"),
)

DURATION_PATTERNS: tuple[tuple[str, int], ...] = (
    (r"(\d+)\s*个?\s*小时", 60),
    (r"(\d+)\s*分钟", 1),
)

DEFAULT_DURATION_MINUTES = 240
MAX_DURATION_MINUTES = 1440
MIN_DURATION_MINUTES = 30


def _collect(message: str, patterns: tuple[tuple[str, str], ...]) -> list[str]:
    found: list[str] = []
    for pattern, value in patterns:
        if re.search(pattern, message) and value not in found:
            found.append(value)
    return found


def extract_groups(message: str) -> list[str]:
    return _collect(message, GROUP_PATTERNS)


def extract_preferences(message: str) -> list[str]:
    return _collect(message, PREFERENCE_PATTERNS)


def extract_required_facilities(message: str) -> list[str]:
    return _collect(message, FACILITY_PATTERNS)


def extract_weather(message: str) -> str:
    found = _collect(message, WEATHER_PATTERNS)
    return found[0] if found else "晴"


def extract_difficulty(message: str) -> str:
    found = _collect(message, DIFFICULTY_PATTERNS)
    return found[0] if found else "轻松"


def extract_duration_minutes(message: str) -> int:
    """Half day -> 240 minutes, full day -> 480, explicit numbers win."""
    if re.search(r"半天|半日", message):
        return 240
    if re.search(r"两日|2\s*天|两天", message):
        return 960
    if re.search(r"一日|一天|全天|整天|1\s*天", message):
        return 480
    for pattern, multiplier in DURATION_PATTERNS:
        match = re.search(pattern, message)
        if match:
            try:
                value = int(match.group(1)) * multiplier
            except (TypeError, ValueError):
                continue
            return max(MIN_DURATION_MINUTES, min(MAX_DURATION_MINUTES, value))
    return DEFAULT_DURATION_MINUTES


def route_arguments(message: str, *, relax: bool = False) -> dict[str, Any]:
    """Build a full ``calculate_route`` argument set from natural language.

    ``relax=True`` is the deterministic re-plan used when ``validate_route`` reports a
    violation: the audience/preference filters are dropped and a tighter budget is
    requested so the next itinerary fits.
    """
    duration = extract_duration_minutes(message)
    arguments: dict[str, Any] = {
        "duration_minutes": duration,
        "groups": [] if relax else extract_groups(message),
        "preferences": [] if relax else extract_preferences(message),
        "weather": extract_weather(message),
        "required_facilities": extract_required_facilities(message),
        "difficulty": extract_difficulty(message),
    }
    if relax:
        # Ask for a slightly smaller budget so a re-plan actually sheds a stop.
        arguments["duration_minutes"] = max(MIN_DURATION_MINUTES, int(duration * 0.8))
        arguments["relaxed"] = True
    return arguments
