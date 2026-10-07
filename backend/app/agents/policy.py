"""Deterministic guardrails around model tool decisions."""

from __future__ import annotations

import re


def required_tools(agent: str, instruction: str) -> list[str]:
    text = instruction or ""
    if agent == "route_agent":
        return ["search_attractions", "calculate_route"]
    if agent == "realtime_agent":
        return ["rag_search"] if re.search(r"门票|开放|限流|轮休|公告|今天|当前|天气", text) else ["get_realtime_status"]
    if agent == "knowledge_agent":
        return ["rag_search"] if re.search(r"门票|票价|开放|规则|规定|海拔|景点|介绍|怎么去|厕所|设施|多少钱", text) else []
    return []


def required_arguments(tool: str, instruction: str) -> dict[str, object]:
    if tool in {"rag_search", "calculate_route"}:
        return {"query": instruction}
    if tool == "search_attractions":
        return {"keyword": None}
    if tool == "get_realtime_status":
        return {"subject": instruction}
    return {}


__all__ = ["required_arguments", "required_tools"]
