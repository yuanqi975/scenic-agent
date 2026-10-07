"""The agent runtime: one ReAct loop, used by every specialist.

```
decide -> (tool calls -> execute -> observe)* -> final answer
```

Everything the runtime needs comes in through :class:`RunContext`, so a specialist is
just a prompt plus a tool whitelist. The loop enforces the safety rails - step budget,
tool-call budget, wall clock, repeated-call detection - before any model is trusted.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.config import (
    AGENT_MAX_STEPS,
    AGENT_TOOL_OUTPUT_MAX_CHARS,
    AGENT_TOOL_TIMEOUT_SECONDS,
    LLM_MODEL,
    PARK_ID,
)
from ..core.errors import AgentError, BudgetExceeded, RepeatedToolCall
from ..services import audit
from ..services.brains import Brain, LLMBrain, render_rule_answer
from ..services.llm import LLMUnavailable
from ..tools.registry import TOOL_REGISTRY, ToolOutcome, tool_schemas, tools_for_agent
from .policy import required_arguments, required_tools
from .loop_guard import Budget, LoopGuard
from .schemas import ROLE_WORKER, AgentCard, AgentResult, StepRecord

Emitter = Callable[[str, dict[str, Any]], None]

MAX_AGENT_OUTPUT_CHARS = 800


def _noop_emitter(event: str, data: dict[str, Any]) -> None:  # pragma: no cover - default
    return None


def _clip(text: str, limit: int = MAX_AGENT_OUTPUT_CHARS) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + "…"


def _can_finish_point_to_point_route(card: AgentCard, context: "RunContext", observations: list[dict[str, Any]]) -> bool:
    """Stop after validated route evidence for a narrow A-to-B navigation request."""
    message = str(context.user_message or "")
    if card.name != "route_agent" or context.intent != "recommendation":
        return False
    if not (any(marker in message for marker in ("从", "我在", "人在")) and "到" in message and any(marker in message for marker in ("怎么走", "怎么去", "如何去"))):
        return False
    if any(marker in message for marker in ("今天", "当前", "明天", "余票", "限流", "暴雨", "泥石流")):
        return False
    completed = {str(item.get("tool")) for item in observations if item.get("status") == "ok"}
    return {"search_attractions", "calculate_route"}.issubset(completed)


def schedule_awaitable(awaitable: Any) -> None:
    """Run an emitter coroutine without forcing every call site to be async.

    Inside a running loop it becomes a task (the SSE generator is async, so this is
    the normal path). Outside one there is nowhere to await it, so it is closed
    explicitly to avoid a "coroutine was never awaited" warning.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()
        return
    loop.create_task(awaitable)


def _compact_observations(observations: list[dict[str, Any]], limit: int = 6) -> str:
    lines: list[str] = []
    for item in observations[-limit:]:
        body = json.dumps(item.get("result") or {}, ensure_ascii=False, default=str)
        lines.append(f"- {item.get('tool')}({item.get('status')}): {_clip(body, 700)}")
    return "\n".join(lines) or "（尚未调用任何工具）"


def handoff_packet(
    *,
    instruction: str,
    user_message: str,
    summary: str = "",
    observations: list[dict[str, Any]] | None = None,
    limit: int = 1500,
) -> str:
    """Compact brief handed to a specialist.

    Passing a *packet* rather than the whole transcript is what keeps context from
    exploding as agents multiply.
    """
    observed = _compact_observations(observations or [])
    packet = (
        f"【游客问题】{user_message}\n"
        f"【当前进度】{summary or '（无）'}\n"
        f"【已有观察】\n{observed}\n"
        f"【你的任务】{instruction}"
    )
    return packet if len(packet) <= limit else packet[:limit] + "…"


@dataclass
class RunContext:
    """Everything one request shares across agents."""

    conversation_id: str
    user_message: str
    history: list[dict[str, str]] = field(default_factory=list)
    intent: str = "unknown"
    mode: str = "fallback"
    budget: Budget = field(default_factory=Budget)
    trace_id: str = ""
    emitter: Emitter = _noop_emitter
    observations: list[dict[str, Any]] = field(default_factory=list)
    agent_chain: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    #: Raw tool observations, flattened out of every agent that ran. ``observations``
    #: holds one summary entry per *agent*; this holds one entry per *tool call*.
    tool_evidence: list[dict[str, Any]] = field(default_factory=list)
    park_id: str = PARK_ID
    model: str | None = None

    def __post_init__(self) -> None:
        if self.model is None:
            self.model = LLM_MODEL if self.mode == "agent" else None

    def emit(self, event: str, **data: Any) -> None:
        """Emit one event to the attached consumer.

        Consumers may be sync (tests, background jobs) or async (the SSE generator),
        so both shapes are supported without making every call site await.
        """
        try:
            outcome = self.emitter(event, data)
        except Exception:  # a broken consumer must not break the run
            return
        if inspect.isawaitable(outcome):
            schedule_awaitable(outcome)

    def absorb(self, result: AgentResult) -> None:
        self.observations.append(
            {
                "tool": f"agent:{result.agent}",
                "status": result.status,
                "result": {
                    "answer": _clip(result.answer, 1200),
                    "citations": result.citations,
                    "artifacts": result.artifacts,
                },
                "latency_ms": result.latency_ms,
            }
        )
        self.agent_chain.append(result.as_dict())
        if result.artifacts:
            self.artifacts.setdefault(result.agent, result.artifacts)
        # Keep the raw tool observations reachable from the run.
        #
        # Tool output lives in the agent's *local* observation list, so it never reached
        # ``context.observations`` (which only holds agent summaries). Anything that
        # wanted the retrieved documents - the legacy deterministic composer above all -
        # therefore always saw an empty evidence set and fell through to a generic
        # answer. Recording the evidence here is what makes the documented
        # "根据景区资料：…" wording able to use what was actually retrieved.
        for observation in result.evidence or []:
            if isinstance(observation, dict):
                self.tool_evidence.append(dict(observation))

    def citations(self) -> list[dict[str, Any]]:
        seen: set[str] = set()
        merged: list[dict[str, Any]] = []
        for item in self.observations:
            result = item.get("result") or {}
            for citation in result.get("citations") or []:
                key = str(citation.get("document_id") or citation.get("source_id"))
                if key in seen:
                    continue
                seen.add(key)
                merged.append(citation)
        return merged

    def summary(self) -> str:
        if not self.agent_chain:
            return "（还没有专员完成工作）"
        return "\n".join(
            f"- {entry['agent']}: {_clip(str(entry.get('answer') or ''), 220)}"
            for entry in self.agent_chain
        )


def build_messages(card: AgentCard, task_instruction: str, context: RunContext) -> list[dict[str, Any]]:
    """System prompt + memory + the delegated brief, in OpenAI message shape."""
    messages: list[dict[str, Any]] = [{"role": "system", "content": card.system_prompt}]
    for item in context.history[-10:]:
        role = "assistant" if item.get("role") == "assistant" else "user"
        messages.append({"role": role, "content": str(item.get("content") or "")[:600]})
    messages.append(
        {
            "role": "user",
            "content": handoff_packet(
                instruction=task_instruction,
                user_message=context.user_message,
                summary=context.summary(),
            ),
        }
    )
    return messages


async def run_agent(
    card: AgentCard,
    *,
    instruction: str,
    context: RunContext,
    brain: Brain,
    run_id: str | None = None,
    parent_run_id: str | None = None,
) -> AgentResult:
    """Execute one agent to completion (or to its budget)."""
    started = time.perf_counter()
    run_id = run_id or audit.start_run(
        agent_name=card.name,
        role=card.role,
        conversation_id=context.conversation_id,
        instruction=instruction,
        goal=context.artifacts.get("goal", ""),
        parent_run_id=parent_run_id,
        trace_id=context.trace_id or None,
        intent=context.intent,
        mode=context.mode,
        model=context.model,
    )
    context.emit(
        "agent_started",
        run_id=run_id,
        agent=card.name,
        role=card.role,
        parent_run_id=parent_run_id,
        instruction=_clip(instruction, 300),
    )

    messages = build_messages(card, instruction, context)
    schemas = tool_schemas(card.name) if card.tools else []
    guard = LoopGuard()
    local_observations: list[dict[str, Any]] = []
    step_index = 0
    iterations = 0
    tool_call_count = 0
    status = "success"
    error: str | None = None
    answer = ""

    record = lambda **kwargs: audit.save_step(  # noqa: E731 - short local alias
        StepRecord(
            run_id=run_id,
            agent_name=card.name,
            parent_run_id=parent_run_id,
            step_index=kwargs.pop("step_index", step_index),
            **kwargs,
        )
    )

    try:
        for iteration in range(min(card.max_steps, AGENT_MAX_STEPS)):
            iterations = iteration + 1
            try:
                context.budget.charge_step()
            except BudgetExceeded as exc:
                status, error = "budget_exceeded", exc.message
                record(step_index=step_index, event_type="guard", text_summary=exc.message, status="rejected")
                break

            step_index += 1

            # Model mode still has deterministic retrieval policy.  The model may
            # choose additional tools, but it cannot answer a high-risk question
            # without the evidence channel mandated by policy.
            if getattr(brain, "enforce_policy", False):
                observed_tools = {str(item.get("tool")) for item in local_observations}
                # The policy classifies the risk of the *visitor's* question. Passing the
                # instruction here keyed the guarantee on internal text: an instruction
                # without the risk vocabulary ("回答游客的问题") skipped the mandated
                # evidence channel entirely, so a ticket-price question could be answered
                # from the model's own memory.
                pending_required = [
                    name
                    for name in required_tools(card.name, context.user_message or instruction)
                    if name not in observed_tools and card.allows(name)
                ]
                if pending_required:
                    forced_name = pending_required[0]
                    forced_args = required_arguments(forced_name, context.user_message)
                    try:
                        guard.check(forced_name, forced_args)
                        context.budget.charge_tool_call()
                    except (RepeatedToolCall, BudgetExceeded) as exc:
                        status, error = "budget_exceeded", exc.message
                        record(step_index=step_index, event_type="guard", text_summary=exc.message, status="rejected")
                        break
                    context.emit(
                        "tool_started",
                        run_id=run_id,
                        agent=card.name,
                        tool=forced_name,
                        arguments=forced_args,
                        step_index=step_index,
                        forced=True,
                    )
                    outcome = await TOOL_REGISTRY.execute(
                        agent=card.name,
                        tool_name=forced_name,
                        arguments=forced_args,
                        timeout=AGENT_TOOL_TIMEOUT_SECONDS,
                    )
                    tool_call_count += 1
                    context.emit(
                        "tool_finished",
                        run_id=run_id,
                        agent=card.name,
                        tool=forced_name,
                        status=outcome.status,
                        latency_ms=outcome.latency_ms,
                        result_summary=outcome.summary(),
                        forced=True,
                    )
                    record(
                        step_index=step_index,
                        event_type="tool_call",
                        tool_name=forced_name,
                        tool_arguments=forced_args,
                        tool_result_summary=outcome.summary(),
                        status=outcome.status if outcome.ok else "error",
                        latency_ms=outcome.latency_ms,
                    )
                    local_observations.append(outcome.as_observation())
                    # OpenAI-compatible providers (including DeepSeek) require every
                    # role=tool message to immediately follow an assistant message
                    # containing the matching tool_call. Forced policy calls must
                    # therefore create the same pair as a model-requested call.
                    forced_id = f"forced_{forced_name}_{iteration}"
                    messages.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "reasoning_content": "先检索与游客问题直接相关的景区资料，再组织回答。",
                            "tool_calls": [
                                {
                                    "id": forced_id,
                                    "type": "function",
                                    "function": {
                                        "name": forced_name,
                                        "arguments": json.dumps(forced_args, ensure_ascii=False),
                                    },
                                }
                            ],
                        }
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": forced_id,
                            "content": outcome.for_model(limit=AGENT_TOOL_OUTPUT_MAX_CHARS),
                        }
                    )
                    if _can_finish_point_to_point_route(card, context, local_observations):
                        answer = render_rule_answer(
                            card.name, local_observations, user_message=context.user_message
                        )
                        break
                    continue
            try:
                decision = await brain.decide(
                    agent=card.name,
                    system_prompt=card.system_prompt,
                    messages=messages,
                    tools=schemas,
                    observations=local_observations,
                    task=instruction,
                    max_steps=card.max_steps,
                    user_message=context.user_message,
                )
            except LLMUnavailable as exc:
                status, error = "model_unavailable", str(exc)
                record(step_index=step_index, event_type="error", text_summary=error, status="error")
                break

            record(
                step_index=step_index,
                event_type="thought",
                text_summary=_clip(decision.content),
                status="ok",
            )
            context.emit(
                "agent_thought",
                run_id=run_id,
                agent=card.name,
                step_index=step_index,
                text=_clip(decision.content, 300),
                source=decision.source,
            )

            if not decision.wants_tools:
                answer = decision.content.strip()
                break

            # Some reasoning models keep issuing slightly different search calls
            # after they already received usable evidence. Cap that behaviour at one
            # follow-up turn and render the grounded deterministic answer instead of
            # burning the request budget or ending with an empty response.
            if local_observations and iteration >= 1 and isinstance(brain, LLMBrain):
                answer = render_rule_answer(card.name, local_observations, user_message=context.user_message)
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": decision.content or None,
                    **({"reasoning_content": str((decision.raw or {}).get("reasoning_content") or "")} if (decision.raw or {}).get("reasoning_content") else {}),
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments, ensure_ascii=False),
                            },
                        }
                        for call in decision.tool_calls
                    ],
                }
            )

            for call in decision.tool_calls:
                step_index += 1
                if not card.allows(call.name):
                    outcome = ToolOutcome(
                        tool=call.name,
                        status="rejected",
                        error=f"{card.name} 的工具白名单不包含 {call.name}",
                        error_code="tool_not_allowed",
                    )
                else:
                    try:
                        guard.check(call.name, call.arguments)
                        context.budget.charge_tool_call()
                    except RepeatedToolCall as exc:
                        outcome = ToolOutcome(
                            tool=call.name,
                            status="rejected",
                            error=exc.message,
                            error_code=exc.code,
                        )
                    except BudgetExceeded as exc:
                        status, error = "budget_exceeded", exc.message
                        record(
                            step_index=step_index,
                            event_type="guard",
                            text_summary=exc.message,
                            status="rejected",
                        )
                        break
                    else:
                        tool_call_count += 1
                        context.emit(
                            "tool_started",
                            run_id=run_id,
                            agent=card.name,
                            tool=call.name,
                            arguments=call.arguments,
                            step_index=step_index,
                        )
                        outcome = await TOOL_REGISTRY.execute(
                            agent=card.name,
                            tool_name=call.name,
                            arguments=call.arguments,
                            timeout=AGENT_TOOL_TIMEOUT_SECONDS,
                        )
                        context.emit(
                            "tool_finished",
                            run_id=run_id,
                            agent=card.name,
                            tool=call.name,
                            status=outcome.status,
                            latency_ms=outcome.latency_ms,
                            result_summary=outcome.summary(),
                        )

                record(
                    step_index=step_index,
                    event_type="tool_call",
                    tool_name=call.name,
                    tool_arguments=call.arguments,
                    tool_result_summary=outcome.summary(),
                    status=outcome.status if outcome.ok else "error",
                    latency_ms=outcome.latency_ms,
                )
                local_observations.append(outcome.as_observation())
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": outcome.for_model(limit=AGENT_TOOL_OUTPUT_MAX_CHARS),
                    }
                )
            else:
                continue
            break

        if not answer:
            # A model can fail after it has already observed useful data.  Return a
            # grounded fallback for every non-cancelled terminal state instead of
            # losing that evidence merely because the final model turn failed.
            answer = render_rule_answer(
                card.name,
                local_observations,
                user_message=context.user_message,
            )
            if not answer.strip() and status == "success":
                status = "no_answer"
        if not answer.strip() and status == "success":
            status = "no_answer"
    except asyncio_cancelled() as exc:  # pragma: no cover - defensive
        status, error = "cancelled", str(exc)
    except Exception as exc:  # a runtime bug must degrade, not explode
        status, error = "error", f"{type(exc).__name__}: {exc}"[:300]
        record(step_index=step_index + 1, event_type="error", text_summary=error, status="error")

    latency_ms = int((time.perf_counter() - started) * 1000)
    citations = context_citations(local_observations)
    result = AgentResult(
        agent=card.name,
        run_id=run_id,
        instruction=instruction,
        answer=_clip(answer, 4000),
        status=status,
        iterations=iterations,
        tool_calls=tool_call_count,
        evidence=[dict(item) for item in local_observations],
        citations=citations,
        error=error,
        latency_ms=latency_ms,
        source=context.mode,
    )
    audit.finish_run(result, role=card.role, mode=context.mode)
    context.emit(
        "agent_finished" if status == "success" else "agent_failed",
        run_id=run_id,
        agent=card.name,
        status=status,
        iterations=iterations,
        tool_calls=tool_call_count,
        latency_ms=latency_ms,
        error=error,
        answer_preview=_clip(result.answer, 200),
    )
    return result


def context_citations(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect citations from tool output, preserving first-seen order."""
    seen: set[str] = set()
    citations: list[dict[str, Any]] = []
    for item in observations:
        result = item.get("result")
        if not isinstance(result, dict):
            continue
        for citation in result.get("citations") or []:
            if not isinstance(citation, dict):
                continue
            key = str(citation.get("document_id") or citation.get("source_id"))
            if key in seen:
                continue
            seen.add(key)
            citations.append(citation)
    return citations


def asyncio_cancelled():  # pragma: no cover - tiny indirection for readability
    import asyncio

    return asyncio.CancelledError


__all__ = [
    "MAX_AGENT_OUTPUT_CHARS",
    "RunContext",
    "build_messages",
    "context_citations",
    "handoff_packet",
    "run_agent",
]
