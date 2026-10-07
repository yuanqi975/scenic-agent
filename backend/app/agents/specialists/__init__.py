"""Specialist agents.

Each one is a thin wrapper: the *behaviour* lives in its system prompt and its tool
whitelist, and the ReAct loop lives in :mod:`app.agents.runtime`. Wrappers still exist
because the project's documentation exposes them as first-class entry points and because
they are the natural place to record agent-specific artefacts (a validated itinerary, a
delegated brief, a permission decision).
"""

from __future__ import annotations

from typing import Any

from ...core.errors import AgentError
from ...services import audit
from ...services.brains import Brain
from ...tools.registry import TOOL_REGISTRY
from ..registry import get_card
from ..runtime import RunContext, run_agent
from ..schemas import AgentResult, ROLE_WORKER


async def _run_specialist(name: str, task: str, context: RunContext, brain: Brain) -> AgentResult:
    card = get_card(name)
    return await run_agent(card, instruction=task, context=context, brain=brain)


def _artifact(result: AgentResult, key: str) -> Any:
    for observation in result.evidence:
        if observation.get("tool") == key and isinstance(observation.get("result"), dict):
            return observation["result"]
    return None


async def knowledge_agent(task: str, context: RunContext, brain: Brain) -> AgentResult:
    """景区知识问答：景点、票价、开放时间、政策与规定。"""
    result = await _run_specialist("knowledge_agent", task, context, brain)
    result.artifacts.setdefault("evidence_documents", len(result.citations))
    if not result.citations and result.status == "success":
        result.artifacts["no_citation"] = True
    return result


async def route_agent(task: str, context: RunContext, brain: Brain) -> AgentResult:
    """游览路线规划：参数由模型决定，路线由确定性算法产出并校验。"""
    result = await _run_specialist("route_agent", task, context, brain)
    plan = _artifact(result, "calculate_route")
    validation = _artifact(result, "validate_route")
    if isinstance(plan, dict):
        result.artifacts["itinerary"] = plan
    if isinstance(validation, dict):
        result.artifacts["validation"] = validation
        if validation.get("valid") is False:
            result.artifacts["unresolved_violations"] = validation.get("violations") or []
    return result


async def realtime_agent(task: str, context: RunContext, brain: Brain) -> AgentResult:
    """实时信息：没有接入的数据源必须如实声明，不得推测。"""
    result = await _run_specialist("realtime_agent", task, context, brain)
    status = _artifact(result, "get_realtime_status")
    if isinstance(status, dict):
        result.artifacts["realtime_available"] = bool(status.get("available"))
        result.artifacts["realtime_summary"] = status.get("summary")
    return result


async def feedback_agent(task: str, context: RunContext, brain: Brain) -> AgentResult:
    """反馈受理：写操作需要游客明确确认，且只能创建待审核候选。"""
    card = get_card("feedback_agent")
    result = await _run_specialist("feedback_agent", task, context, brain)
    created = _artifact(result, "create_feedback_candidate")
    if isinstance(created, dict):
        result.artifacts["feedback_status"] = created.get("status")
        result.artifacts["written"] = bool(created.get("written"))
        if created.get("status") == "awaiting_confirmation":
            result.artifacts["awaiting_confirmation"] = True
    elif card.requires_confirmation:
        result.artifacts.setdefault("consultation_only", True)
    return result


async def facility_agent(task: str, context: RunContext, brain: Brain) -> AgentResult:
    """设施向导。

    Kept as an explicit entry point for the documented ``facility_expert`` role. It is
    deliberately served by ``knowledge_agent``: facility lookups are knowledge work and
    a second agent with the same tools and prompt would be a rename, not an agent.
    """
    result = await knowledge_agent(task, context, brain)
    facilities = _artifact(result, "get_facilities")
    if isinstance(facilities, dict):
        result.artifacts["facilities"] = facilities
    return result


#: Supervisor roster -> callable. Keys are the names handed to the model.
SPECIALISTS = {
    "knowledge_agent": knowledge_agent,
    "route_agent": route_agent,
    "realtime_agent": realtime_agent,
    "feedback_agent": feedback_agent,
    "facility_agent": facility_agent,
}


async def run_specialist(name: str, task: str, context: RunContext, brain: Brain) -> AgentResult:
    """Dispatch one specialist by name, degrading instead of raising on unknown names."""
    handler = SPECIALISTS.get(name)
    if handler is None:
        result = AgentResult(
            agent=name,
            run_id="",
            instruction=task,
            status="unknown_agent",
            error=f"未注册的专员：{name}",
            source=context.mode,
        )
        audit.save_step(
            {
                "run_id": context.trace_id or name,
                "agent_name": "supervisor",
                "step_index": 0,
                "event_type": "error",
                "text_summary": result.error,
                "status": "error",
            }
        )
        return result
    try:
        return await handler(task, context, brain)
    except AgentError as exc:
        return AgentResult(
            agent=name,
            run_id="",
            instruction=task,
            status="rejected",
            error=exc.message,
            source=context.mode,
        )


def specialist_tool_names(name: str) -> list[str]:
    """Tool whitelist for one specialist, resolved through the registry."""
    card = get_card(name)
    return [spec.name for spec in TOOL_REGISTRY.for_agent(card.name)] if card.tools else []


__all__ = [
    "SPECIALISTS",
    "ROLE_WORKER",
    "facility_agent",
    "feedback_agent",
    "knowledge_agent",
    "realtime_agent",
    "route_agent",
    "run_specialist",
    "specialist_tool_names",
]
