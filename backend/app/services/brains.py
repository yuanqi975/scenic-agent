"""The decision layer: a model-backed brain and a deterministic rule-backed brain.

Both implement the same protocol so the *entire* multi-agent machinery - tool loop,
supervisor delegation, audit trail, SSE events - is identical whether or not a model
is configured. Without a model the system records ``mode="fallback"`` and a
``rule_brain`` guard step, so the UI never claims an LLM reasoned when it did not.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..core.config import AGENT_MAX_STEPS, llm_configured
from .llm import LLMClient, LLMUnavailable, ToolCall
from .prompts import SUPERVISOR_SYSTEM

FINISH = "FINISH"

#: Deterministic wording for the "should I file this?" turn. It deliberately does not
#: claim the feedback was submitted: the only write tool requires an explicit
#: confirmation, so the fallback path must ask for one instead of inventing a ticket.
FEEDBACK_CONFIRMATION_ASK = (
    "你反馈的问题我已记录，稍后可由景区工作人员核实处理。\n"
    "需要我帮你把这条反馈提交给景区吗？回复「提交」我就登记为待审核反馈，"
    "管理员审核通过后才会进入知识库。"
)

#: Cheap, bounded mapping from a supervisor task instruction to the concrete tool
#: calls the deterministic brain should issue. Ordered most-specific first.
TASK_TOOL_PLANS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("实时", "天气", "客流", "开放状态", "临时", "维护", "限流", "轮休"), ("get_realtime_status",)),
    (("路线", "行程", "安排", "游览顺序", "规划"), ("search_attractions", "calculate_route", "validate_route")),
    (("设施", "卫生间", "厕所", "餐饮", "停车", "轮椅", "换乘"), ("get_facilities",)),
    (("票", "门票", "优惠", "预约"), ("search_knowledge",)),
    (("反馈", "投诉", "建议", "报修"), ()),
    (("知识", "问答", "介绍", "开放时间", "规定"), ("search_knowledge",)),
)


@dataclass(slots=True)
class Decision:
    """What the brain decided for the next step of one agent."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict[str, Any] | None = None
    source: str = "model"

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


@dataclass(slots=True)
class Plan:
    """Supervisor plan: which specialists to run, with what instruction."""

    goal: str = ""
    tasks: list[dict[str, str]] = field(default_factory=list)
    finish: bool = False
    reason: str = ""
    source: str = "model"


class Brain(Protocol):
    """Decision interface implemented by :class:`LLMBrain` and :class:`RuleBrain`."""

    mode: str

    async def plan(self, message: str, history: list[dict[str, str]], agents: list[str]) -> Plan: ...

    async def review(self, message: str, summary: str, agents: list[str]) -> Plan:
        """Decide whether more delegation is needed after watching the results."""
        ...

    async def decide(
        self,
        *,
        agent: str,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        observations: list[dict[str, Any]],
        task: str,
        max_steps: int = AGENT_MAX_STEPS,
        user_message: str = "",
    ) -> Decision: ...


# --------------------------------------------------------------------------- model brain
class LLMBrain:
    """Delegates every decision to the configured OpenAI-compatible model."""

    mode = "agent"
    enforce_policy = True

    def __init__(self, client: LLMClient | None = None) -> None:
        self._client = client

    @property
    def client(self) -> LLMClient:
        if self._client is None:
            from .llm import get_client

            self._client = get_client()
        return self._client

    async def plan(self, message: str, history: list[dict[str, str]], agents: list[str]) -> Plan:
        roster = "、".join(agents)
        transcript = "\n".join(f"{item['role']}: {item['content'][:200]}" for item in history[-10:])
        user_content = (
            f"可派发的专员：{roster}\n"
            f"最近的对话：\n{transcript or '（无）'}\n"
            f"游客本次问题：{message}"
        )
        payload = await self.client.complete_json(
            [
                {"role": "system", "content": SUPERVISOR_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
        )
        return plan_from_payload(payload)

    async def review(self, message: str, summary: str, agents: list[str]) -> Plan:
        """Second supervision round: continue only if the model says so."""
        roster = "、".join(agents)
        user_content = (
            f"可派发的专员：{roster}\n"
            f"游客问题：{message}\n"
            f"已完成的专员工作：\n{summary or '（无）'}\n"
            "信息是否已经足够回答游客？若不足，指出还需要派发哪些专员；若已足够，finish=true 且 tasks 为空。"
        )
        try:
            payload = await self.client.complete_json(
                [
                    {"role": "system", "content": SUPERVISOR_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                temperature=0,
            )
        except LLMUnavailable:
            return Plan(finish=True, reason="supervisor-unavailable", source="model")
        return plan_from_payload(payload)

    async def decide(
        self,
        *,
        agent: str,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        observations: list[dict[str, Any]],
        task: str,
        max_steps: int = AGENT_MAX_STEPS,
        user_message: str = "",
    ) -> Decision:
        reply = await self.client.complete(messages, tools=tools or None)
        return Decision(
            content=reply.content,
            tool_calls=reply.tool_calls,
            raw={"usage": reply.usage, "finish_reason": reply.finish_reason, "reasoning_content": reply.reasoning_content},
            source="model",
        )


def plan_from_payload(payload: dict[str, Any]) -> Plan:
    """Validate a supervisor plan; never trust the model's shape."""
    raw_tasks = payload.get("tasks")
    tasks: list[dict[str, str]] = []
    if isinstance(raw_tasks, list):
        for item in raw_tasks:
            if not isinstance(item, dict):
                continue
            agent = str(item.get("agent") or "").strip()
            if not agent:
                continue
            tasks.append({"agent": agent, "instruction": str(item.get("instruction") or "").strip()})
    finish = bool(payload.get("finish")) or not tasks
    return Plan(
        goal=str(payload.get("goal") or "").strip(),
        tasks=tasks,
        finish=finish,
        reason=str(payload.get("reason") or "").strip()[:200],
        source="model",
    )


# --------------------------------------------------------------------------- rule brain
class RuleBrain:
    """Deterministic brain: the documented degraded path.

    It reuses the frozen keyword router for planning and a small task-instruction
    matcher for tool selection, then answers from the observations it collected.
    """

    mode = "fallback"

    def __init__(self, router: Any | None = None) -> None:
        self._router = router

    def _route(self, message: str) -> dict[str, Any]:
        if self._router is not None:
            return self._router(message)
        from .intent import keyword_route

        return keyword_route(message)

    async def plan(self, message: str, history: list[dict[str, str]], agents: list[str]) -> Plan:
        route = self._route(message)
        intent = route.get("intent", "qa")
        allowed = set(agents)
        tasks: list[dict[str, str]] = []
        if intent == "recommendation" and "route_agent" in allowed:
            tasks.append({"agent": "route_agent", "instruction": "按游客的时长、人群、天气与设施要求规划并校验路线"})
            if "realtime_agent" in allowed and re.search(r"天气|下雨|雨|雪|气温", message):
                tasks.append({"agent": "realtime_agent", "instruction": "查询与本次出行相关的实时状态"})
        elif intent == "realtime" and "realtime_agent" in allowed:
            tasks.append({"agent": "realtime_agent", "instruction": "查询游客询问的实时状态"})
        elif intent == "feedback" and "feedback_agent" in allowed:
            tasks.append({"agent": "feedback_agent", "instruction": "判断是咨询还是反馈，必要时创建待审核知识候选"})
        elif "knowledge_agent" in allowed:
            tasks.append({"agent": "knowledge_agent", "instruction": "依据景区知识库回答问题并给出引用"})
        return Plan(
            goal=f"按关键词路由处理 {intent} 类问题",
            tasks=tasks,
            finish=not tasks,
            reason="keyword-fallback",
            source="keyword",
        )

    async def review(self, message: str, summary: str, agents: list[str]) -> Plan:
        """The rule brain always converges: one deterministic round is the whole plan."""
        return Plan(finish=True, reason="rule-brain-single-round", source="keyword")

    async def decide(
        self,
        *,
        agent: str,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        observations: list[dict[str, Any]],
        task: str,
        max_steps: int = AGENT_MAX_STEPS,
        user_message: str = "",
    ) -> Decision:
        plan = self.tool_plan(agent, task, observations)
        if plan:
            # Search tools must be given the visitor's own wording, never the supervisor's
            # instruction. ``task`` is internal text such as "依据景区知识库回答问题并给出引用",
            # and retrieving with it returns nothing on every request - which is exactly
            # how the deterministic fallback ended up answering "暂未查询到足够信息" for
            # questions the knowledge base could answer.
            query = _search_query(task, user_message)
            return Decision(
                content=f"[规则决策] 调用 {'、'.join(plan)}",
                tool_calls=[
                    ToolCall(
                        id=f"rule_{agent}_{index}",
                        name=name,
                        arguments=self.tool_arguments(agent, name, task, observations, query=query),
                    )
                    for index, name in enumerate(plan)
                ],
                source="rule",
            )
        return Decision(content=render_rule_answer(agent, observations, user_message=user_message), source="rule")

    # ---------------------------------------------------------------- deterministic plans
    def tool_plan(self, agent: str, task: str, observations: list[dict[str, Any]]) -> list[str]:
        """Which tools to call next, given what has already been observed."""
        called = {str(item.get("tool")) for item in observations}
        available = self._available(agent)

        if agent == "knowledge_agent":
            sequence = ["search_knowledge"]
        elif agent == "route_agent":
            sequence = ["search_attractions", "calculate_route", "validate_route"]
            if re.search(r"卫生间|厕所|洗手间|餐饮|停车|轮椅|换乘|车站", task) or "设施" in task:
                sequence.insert(1, "get_facilities")
            if re.search(r"天气|下雨|雨|雪|气温|客流|开放", task):
                sequence.insert(1, "get_realtime_status")
        elif agent == "realtime_agent":
            sequence = ["get_realtime_status"]
        elif agent == "feedback_agent":
            # A write tool is only ever planned behind an explicit confirmation.
            sequence = [] if not _explicit_submission(task) else ["create_feedback_candidate"]
        else:
            sequence = []

        sequence = [name for name in sequence if name in available]
        if not called:
            return sequence[:1]
        # Feed the validator's verdict back into one deterministic re-plan.
        if agent == "route_agent" and "validate_route" in called:
            invalid = _invalid_route(observations)
            if invalid and "recalculated" not in called:
                return ["calculate_route"] if "calculate_route" in available else []
        pending = [name for name in sequence if name not in called]
        return pending[:1]

    @staticmethod
    def _available(agent: str) -> set[str]:
        from ..tools.registry import TOOL_REGISTRY, tools_for_agent

        return {spec.name for spec in tools_for_agent(agent)} or set(TOOL_REGISTRY)

    def tool_arguments(
        self,
        agent: str,
        tool: str,
        task: str,
        observations: list[dict[str, Any]],
        *,
        query: str | None = None,
    ) -> dict[str, Any]:
        """Deterministic arguments for the fallback path.

        ``query`` is the visitor-facing text resolved by :func:`_search_query`. It is the
        parameter extraction source for *every* visitor-facing tool, not just the search
        tools; it falls back to the instruction only so direct callers keep working.
        """
        from .tool_params import route_arguments

        text = query or _query_text(task)
        if tool in {"search_knowledge", "rag_search"}:
            return {"query": text}
        if tool in {"search_attractions", "get_facilities"}:
            return {"keyword": None} if tool == "search_attractions" else {"facility_type": None}
        if tool == "calculate_route":
            # ``text`` is the *visitor's* wording. Routing the instruction instead cost
            # the visitor their stated conditions: the regex extractor found no duration,
            # audience or preference in "按游客的时长、人群、天气与设施要求规划并校验路线", so
            # every fallback-mode request silently planned on the 240-minute default with
            # no audience and no preferences - and then validated that plan against
            # itself, so nothing ever surfaced the loss.
            if _invalid_route(observations):
                return route_arguments(text, relax=True)
            return route_arguments(text)
        if tool == "validate_route":
            plan = _last_route(observations)
            arguments: dict[str, Any] = {
                "plan": plan,
                "duration_minutes": plan.get("duration_minutes", 240),
            }
            # Validate against the exact date the plan was built for. Omitting it makes
            # the validator fall back to its conservative "any published closure is a
            # violation" mode and report a conflict with a plan that is actually valid.
            if plan.get("travel_date"):
                arguments["travel_date"] = plan["travel_date"]
            return arguments
        if tool == "get_realtime_status":
            return {"subject": text}
        if tool == "create_feedback_candidate":
            return {"content": task, "confirmed": True}
        return {}


def _explicit_submission(task: str) -> bool:
    """A write may only happen when the visitor explicitly asked to submit."""
    return bool(re.search(r"提交|帮我反馈|我要投诉|请反馈|上报", task)) and not bool(
        re.search(r"需要我帮你(把这条)?反馈|是否提交|要不要提交", task)
    )


def _query_text(task: str) -> str:
    cleaned = re.sub(r"^\[规则决策\][^\n]*\n?", "", task or "").strip()
    return cleaned or task or "九寨沟"


def _search_query(task: str, user_message: str) -> str:
    """Search text for a retrieval tool: the visitor's question wins over the brief.

    Falling back to the instruction rather than to a hardcoded string keeps a caller
    that only has the instruction working, but the instruction is internal text that no
    visitor ever typed, so searching with it reliably returns nothing.
    """
    question = (user_message or "").strip()
    return question or _query_text(task)


def _route_observation(observations: list[dict[str, Any]], tool: str) -> dict[str, Any] | None:
    for item in reversed(observations):
        if item.get("tool") == tool and isinstance(item.get("result"), dict):
            return item["result"]
    return None


def _last_route(observations: list[dict[str, Any]]) -> dict[str, Any]:
    result = _route_observation(observations, "calculate_route") or {}
    return result if isinstance(result, dict) else {}


def _invalid_route(observations: list[dict[str, Any]]) -> bool:
    validation = _route_observation(observations, "validate_route")
    if isinstance(validation, dict) and validation.get("valid") is False:
        return True
    plan = _last_route(observations)
    return bool(plan and plan.get("total_minutes", 0) > plan.get("duration_minutes", 10**9))


def render_rule_answer(
    agent: str,
    observations: list[dict[str, Any]],
    *,
    user_message: str = "",
) -> str:
    """Deterministic final answer, built only from tool output."""
    if not observations:
        # A reported problem is a *conversation*, not a lookup: the correct deterministic
        # behaviour is to describe what will happen and ask for explicit confirmation,
        # because the write tool refuses to write without it.
        if agent == "feedback_agent":
            return FEEDBACK_CONFIRMATION_ASK
        return (
            "暂未在九寨沟景区官方知识库中查询到足够信息，建议咨询游客服务中心（0837-7739753）。"
        )

    lines: list[str] = []
    for item in observations:
        if item.get("status") != "ok":
            continue
        result = item.get("result")
        tool = item.get("tool")
        if tool in {"search_knowledge", "rag_search"} and isinstance(result, dict):
            from .answer_focus import focused_evidence_answer

            evidence = [hit for hit in (result.get("items") or []) if isinstance(hit, dict)]
            focused = focused_evidence_answer(user_message, evidence)
            if focused:
                lines.append(focused)
        elif tool in {"search_attractions", "get_facilities"} and isinstance(result, dict):
            names = [str(entry.get("name")) for entry in (result.get("items") or [])[:5]]
            if names:
                lines.append(f"已确认开放的相关点位：{'、'.join(names)}。")
        elif tool == "calculate_route" and isinstance(result, dict):
            if result.get("point_to_point"):
                lines.append(
                    f"从{result.get('origin')}到{result.get('destination')}建议按以下顺序："
                    + "；".join(
                        f"{seg.get('from')}→{seg.get('to')}（{seg.get('route_type')}"
                        + (f"，约{seg.get('estimated_minutes')}分钟" if seg.get('estimated_minutes') else "")
                        + ")" for seg in (result.get("route_segments") or [])
                    )
                    + "。"
                )
                lines.append(str(result.get("transport_notes") or ""))
                continue
            names = [str(entry.get("name")) for entry in (result.get("attractions") or [])]
            if names:
                lines.append(
                    f"根据景区资料，按 {result.get('duration_minutes', 0)} 分钟游玩时长可安排："
                    f"{'、'.join(names)}，合计约 {result.get('total_minutes', 0)} 分钟。"
                )
        elif tool == "validate_route" and isinstance(result, dict):
            if result.get("valid"):
                lines.append("该行程已通过时长与开放状态校验。")
            for warning in (result.get("warnings") or [])[:2]:
                lines.append(f"提示：{warning.get('detail')}")
        elif tool == "get_realtime_status" and isinstance(result, dict):
            if result.get("available"):
                lines.append(f"实时数据：{result.get('summary')}")
            else:
                lines.append(
                    "九寨沟景区暂无接入的实时数据源，无法确认当前天气与客流，"
                    "请以景区现场公告为准（咨询电话 0837-7739753）。"
                )
        elif tool == "create_feedback_candidate" and isinstance(result, dict):
            lines.append("你的反馈已提交，正在等待景区人工审核，感谢你的建议。")

    body = "\n".join(line for line in lines if line.strip())
    if not body:
        return "暂未在九寨沟景区官方知识库中查询到足够信息，建议咨询游客服务中心（0837-7739753）。"
    return body + "\n以上信息来自当前景区知识库，请以现场公告为准。"


# --------------------------------------------------------------------------- factory
_default_brain: Brain | None = None


def build_brain(mode: str | None = None, *, client: LLMClient | None = None) -> Brain:
    """Return the brain for this request: model when usable, rules otherwise."""
    from ..core.config import resolved_agent_mode

    effective = mode or resolved_agent_mode()
    if effective == "agent" and llm_configured():
        return LLMBrain(client)
    return RuleBrain()


def set_default_brain(brain: Brain | None) -> None:
    """Test/deployment helper to pin one brain implementation."""
    global _default_brain
    _default_brain = brain


def get_default_brain() -> Brain:
    global _default_brain
    if _default_brain is None:
        _default_brain = build_brain()
    return _default_brain


__all__ = [
    "Brain",
    "Decision",
    "FINISH",
    "LLMBrain",
    "LLMUnavailable",
    "Plan",
    "RuleBrain",
    "build_brain",
    "get_default_brain",
    "plan_from_payload",
    "render_rule_answer",
    "set_default_brain",
]
