"""Deterministic multi-query planning used before hybrid retrieval.

The planner is intentionally conservative: the visitor's original wording is always
kept as Q0 and rule expansions only remove connective noise.  It never invents dates,
prices, or locations that were not present in the request.
"""

from __future__ import annotations

import re


def plan_queries(query: str, *, max_queries: int = 4) -> list[dict[str, object]]:
    original = " ".join(str(query or "").split()).strip()
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
    if any(token in original for token in ("开放", "预约", "门票", "票价", "承载", "限流", "公告", "现在", "今天")) and len(plans) < max_queries:
        plans.append({"text": f"{original} 官方公告 当前有效时间", "type": "rule", "weight": 0.7})
    return plans[:max_queries]


__all__ = ["plan_queries"]
