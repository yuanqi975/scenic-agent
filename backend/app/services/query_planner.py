"""Deterministic multi-query planning used before hybrid retrieval.

The planner is intentionally conservative: the visitor's original wording is always
kept as Q0 and rule expansions only remove connective noise.  It never invents dates,
prices, or locations that were not present in the request.
"""

from __future__ import annotations

import re


_TOPIC_EXPANSIONS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("开放", "关闭", "开放时间", "入园"), "开放时间 开放范围 当前公告"),
    (("门票", "票价", "购票", "预约", "售罄", "限流", "承载"), "门票 预约 承载量 官方公告"),
    (("路线", "行程", "怎么走", "怎么去", "换乘", "一日游"), "游览路线 观光车 换乘 景点"),
    (("老人", "儿童", "孩子", "少走路", "无障碍"), "老人 儿童 少步行 观光车 服务设施"),
    (("厕所", "卫生间", "洗手间", "服务设施"), "游客服务中心 服务设施 卫生间"),
)


def _normalise(text: str) -> str:
    return " ".join(text.split()).strip(" ，,。；;？?！!")


def plan_queries(query: str, *, max_queries: int = 4) -> list[dict[str, object]]:
    """Return conservative recall variants without adding unsupported facts.

    Every expansion is a retrieval vocabulary bridge (for example ``少走路`` to
    ``观光车``), never a fabricated date, price, attraction or operating status.
    """
    original = _normalise(str(query or ""))
    if not original:
        return []
    plans: list[dict[str, object]] = [{"text": original, "type": "original", "weight": 1.0}]
    # Keep expansion for compound questions only; simple questions should have one
    # retrieval round and predictable latency.
    parts = [part.strip(" ，,。；;？?！!") for part in re.split(r"(?:和|以及|并且|同时|，|,|；|;)", original)]
    parts = [part for part in parts if len(part) >= 2 and part != original]
    if len(parts) >= 2:
        for part in parts[: max_queries - 1]:
            plans.append({"text": part, "type": "attribute", "weight": 0.8})
    for triggers, vocabulary in _TOPIC_EXPANSIONS:
        if len(plans) >= max_queries or not any(token in original for token in triggers):
            continue
        # Keep the visitor wording in the variant so a generic topic term does not
        # outrank the question's named attraction or business constraint.
        plans.append({"text": f"{original} {vocabulary}", "type": "rule", "weight": 0.7})
    deduplicated: list[dict[str, object]] = []
    seen: set[str] = set()
    for plan in plans:
        text = str(plan["text"])
        if text not in seen:
            seen.add(text)
            deduplicated.append(plan)
    return deduplicated[:max_queries]


__all__ = ["plan_queries"]
