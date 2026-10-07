"""The supervisor: plan, delegate in parallel, observe, decide whether to continue.

Documented contract (``docs/多Agent入门详解.md`` §4.2, §10.3 Step 3) kept intact:
``FINISH`` sentinel, a hardcoded delegation ceiling the model cannot talk its way past
(``max_steps=3``), a compacted brief instead of the whole transcript, and support for
both single- and multi-delegation per round.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ..core.config import AGENT_MAX_AGENTS, AGENT_MAX_CONCURRENCY, AGENT_MAX_DELEGATION_ROUNDS
from ..core.errors import AgentError, BudgetExceeded
from ..services import audit
from ..services.brains import FINISH, Brain, LLMBrain, Plan, get_default_brain
from .registry import SPECIALIST_AGENTS, delegation_prompt_roster, get_card
from .runtime import RunContext
from .schemas import AgentResult, Delegation, Task
from .specialists import run_specialist

#: Sentinel re-exported so callers can test for it without importing brains.
FINISH_SENTINEL = FINISH


def _is_point_to_point_request(context: RunContext) -> bool:
    message = str(context.user_message or "")
    return (
        context.intent == "recommendation"
        and any(marker in message for marker in ("从", "我在", "人在"))
        and "到" in message
        and any(marker in message for marker in ("怎么走", "怎么去", "如何去"))
        and not any(marker in message for marker in ("今天", "当前", "明天", "余票", "限流", "暴雨", "泥石流"))
    )


def _fast_path_request(context: RunContext, brain: Brain) -> bool:
    """Use the deterministic one-specialist plan for short, low-risk QA turns.

    The specialist, tool policy and evidence path remain unchanged. This only avoids
    spending a separate model round-trip asking the supervisor to rediscover the
    obvious ``knowledge_agent -> retrieval`` plan.
    """
    if not isinstance(brain, LLMBrain):
        return False
    message = str(context.user_message or "")
    if _is_point_to_point_request(context):
        return True
    if context.intent not in {"qa", "smalltalk"}:
        return False
    if len(message) > 100:
        return False
    complex_markers = (
        "路线", "怎么走", "怎么安排", "换乘", "老人", "孩子", "半天", "一天",
        "实时", "今天", "当前", "明天", "余票", "限流", "关闭", "反馈", "投诉",
    )
    return not any(marker in message for marker in complex_markers)


def build_supervisor_input(context: RunContext, *, instruction: str = "") -> str:
    """Progress digest for the supervisor: a summary, never the full transcript."""
    return (
        f"【游客问题】{context.user_message}\n"
        f"【意图】{context.intent}\n"
        f"【已完成】\n{context.summary()}\n"
        f"【可派发专员】{'、'.join(delegation_prompt_roster())}\n"
        f"【下一步该做什么？{'（' + instruction + '）' if instruction else ''}】"
    )


def topo_batches(tasks: list[Task]) -> list[list[Task]]:
    """Group tasks into dependency-ordered batches.

    Tasks in the same batch have no unmet dependency and therefore run concurrently -
    this is what makes the collaboration real rather than a for-loop of fake agents.
    """
    remaining = list(tasks)
    batches: list[list[Task]] = []
    completed: set[str] = set()
    guard = 0
    while remaining and guard <= len(tasks) + 1:
        guard += 1
        ready = [task for task in remaining if all(dep in completed for dep in task.depends_on)]
        if not ready:
            # A dependency cycle: fall back to running everything left in one batch.
            batches.append(remaining)
            break
        batches.append(ready)
        for task in ready:
            completed.add(task.task_id)
            remaining.remove(task)
    return batches


def _dedupe(tasks: list[Task], limit: int) -> list[Task]:
    """Drop all-unknown plans, but retain one unknown task beside valid work.

    Keeping the malformed task in a mixed plan lets the supervisor emit a visible
    ``unknown_agent`` result without allowing a plan made entirely of hallucinated
    agents to spend any execution budget.
    """
    known = set(SPECIALIST_AGENTS)
    seen: set[tuple[str, str]] = set()
    result: list[Task] = []
    unknown: list[Task] = []
    for task in tasks:
        key = (task.agent, task.instruction.strip()[:80])
        if key in seen:
            continue
        seen.add(key)
        if task.agent not in known:
            unknown.append(task)
            continue
        result.append(task)
        if len(result) >= limit:
            break
    if result and len(result) < limit:
        result.extend(unknown[: limit - len(result)])
    return result


async def run_supervisor(
    context: RunContext,
    *,
    brain: Brain | None = None,
    max_steps: int = AGENT_MAX_DELEGATION_ROUNDS,
    agent_limit: int = AGENT_MAX_AGENTS,
    plan: Plan | None = None,
) -> list[AgentResult]:
    """Return every specialist result gathered while working on this request."""
    brain = brain or get_default_brain()
    results: list[AgentResult] = []

    supervisor_card = get_card("supervisor")
    supervisor_run_id = audit.start_run(
        agent_name=supervisor_card.name,
        role=supervisor_card.role,
        conversation_id=context.conversation_id,
        instruction="规划与调度",
        goal=context.artifacts.get("goal", ""),
        trace_id=context.trace_id or None,
        intent=context.intent,
        mode=context.mode,
        model=context.model,
    )
    context.emit("agent_started", run_id=supervisor_run_id, agent="supervisor", role=supervisor_card.role)

    current_plan = plan
    round_index = 0
    try:
        for round_index in range(max_steps):
            if current_plan is None:
                if _fast_path_request(context, brain):
                    current_plan = _safety_plan(context)
                    current_plan.source = "fast-rule"
                else:
                    current_plan = await brain.plan(
                        context.user_message, context.history, delegation_prompt_roster()
                    )
            # A model can incorrectly return an empty/finished plan before retrieval.
            # Always run one deterministic evidence task on the first round.
            if (current_plan.finish or not current_plan.tasks) and round_index == 0:
                current_plan = _safety_plan(context)
            if current_plan.finish or not current_plan.tasks:
                break
            if current_plan.goal:
                context.artifacts["goal"] = current_plan.goal

            tasks = _dedupe(
                [
                    Task(
                        agent=item.get("agent", ""),
                        instruction=item.get("instruction") or context.user_message,
                        round_index=round_index,
                        parent_run_id=supervisor_run_id,
                    )
                    for item in current_plan.tasks
                ],
                agent_limit,
            )
            if not tasks:
                break

            context.emit(
                "plan",
                run_id=supervisor_run_id,
                goal=current_plan.goal,
                round=round_index,
                tasks=[task.as_dict() for task in tasks],
                source=current_plan.source,
            )

            for batch in topo_batches(tasks):
                for task in batch:
                    audit.save_delegation(
                        parent_run_id=supervisor_run_id,
                        from_agent="supervisor",
                        to_agent=task.agent,
                        instruction=task.instruction,
                        round_index=round_index,
                        status="pending",
                    )
                batch_results = await _run_batch(batch, context, brain, supervisor_run_id)
                for task, result in zip(batch, batch_results):
                    if result.run_id:
                        audit.attach_child_run(supervisor_run_id, result.run_id, task.agent)
                    if result.status == "success":
                        context.absorb(result)
                    else:
                        # A failed specialist is recorded, never fatal.
                        context.agent_chain.append(result.as_dict())
                        audit.save_delegation(
                            parent_run_id=supervisor_run_id,
                            from_agent="supervisor",
                            to_agent=task.agent,
                            instruction=task.instruction,
                            round_index=round_index,
                            child_run_id=result.run_id or None,
                            status="failed",
                        )
                    results.append(result)

            if context.budget.exhausted_by_time:
                break
            try:
                current_plan = await _follow_up(brain, context)
            except AgentError:
                break
            if current_plan is None or current_plan.finish or not current_plan.tasks:
                break
    except BudgetExceeded as exc:
        audit.save_step(
            {
                "run_id": supervisor_run_id,
                "agent_name": "supervisor",
                "step_index": round_index + 1,
                "event_type": "guard",
                "text_summary": exc.message,
                "status": "rejected",
            }
        )
        context.emit("agent_guard", run_id=supervisor_run_id, agent="supervisor", reason=exc.message)

    audit.finish_run(
        AgentResult(
            agent="supervisor",
            run_id=supervisor_run_id,
            instruction="规划与调度",
            answer=f"派出 {len(results)} 个专员",
            status="success",
            iterations=round_index + 1,
            tool_calls=0,
            latency_ms=0,
            source=context.mode,
        ),
        role=supervisor_card.role,
        mode=context.mode,
    )
    context.emit(
        "agent_finished",
        run_id=supervisor_run_id,
        agent="supervisor",
        status="success",
        rounds=round_index + 1,
        delegated=len(results),
    )
    return results


def _safety_plan(context: RunContext) -> Plan:
    """Fallback dispatch that prevents an empty supervisor plan from skipping RAG."""
    message = context.user_message
    route_words = ("怎么走", "怎么去", "路线", "行程", "换乘", "安排", "游览")
    ticket_words = ("门票", "预约", "领票", "余票", "门票价格")
    realtime_words = ("今天", "现在", "当前", "限流", "开放情况", "暴雨", "泥石流", "下雪")
    tasks: list[dict[str, str]] = []
    if context.intent == "feedback":
        tasks.append({"agent": "feedback_agent", "instruction": "处理游客反馈，未经明确提交确认不得写入"})
    elif context.intent == "recommendation" or any(word in message for word in route_words):
        tasks.append({"agent": "route_agent", "instruction": "识别起点终点或行程约束，查询资料并给出可执行路线"})
        if any(word in message for word in ticket_words):
            tasks.append({"agent": "knowledge_agent", "instruction": "检索门票、网上预约与优惠规则"})
        if any(word in message for word in realtime_words):
            tasks.append({"agent": "realtime_agent", "instruction": "查询与本次出行相关的实时状态"})
    elif context.intent == "realtime":
        if any(word in message for word in ticket_words):
            tasks.append({"agent": "knowledge_agent", "instruction": "检索门票、网上预约与优惠规则"})
        tasks.append({"agent": "realtime_agent", "instruction": "查询问题涉及的实时状态；没有实时源时明确说明"})
    else:
        tasks.append({"agent": "knowledge_agent", "instruction": "检索景区知识库，回答游客问题并给出引用"})
    return Plan(goal="确定性检索兜底", tasks=tasks, finish=False, reason="empty-supervisor-plan", source="safety")


async def _run_batch(
    batch: list[Task], context: RunContext, brain: Brain, supervisor_run_id: str
) -> list[AgentResult]:
    """Run one dependency-free batch concurrently, within the concurrency ceiling."""
    semaphore = asyncio.Semaphore(max(1, AGENT_MAX_CONCURRENCY))

    async def one(task: Task) -> AgentResult:
        async with semaphore:
            try:
                return await run_specialist(task.agent, task.instruction, context, brain)
            except AgentError as exc:
                return AgentResult(
                    agent=task.agent,
                    run_id="",
                    instruction=task.instruction,
                    status="rejected",
                    error=exc.message,
                    source=context.mode,
                )
            except Exception as exc:  # one bad agent must not sink the batch
                return AgentResult(
                    agent=task.agent,
                    run_id="",
                    instruction=task.instruction,
                    status="error",
                    error=f"{type(exc).__name__}: {exc}"[:300],
                    source=context.mode,
                )

    return list(await asyncio.gather(*(one(task) for task in batch)))


async def _follow_up(brain: Brain, context: RunContext) -> Plan | None:
    """Ask the brain whether more delegation is needed given what came back.

    The rule brain always converges after one round: it cannot judge novelty, and
    guessing would only burn budget on a request that is already answerable.
    """
    if getattr(brain, "mode", "fallback") != "agent":
        return None
    if _fast_path_request(context, brain):
        return None
    return await brain.review(context.user_message, context.summary(), delegation_prompt_roster())


__all__ = [
    "FINISH_SENTINEL",
    "build_supervisor_input",
    "run_supervisor",
    "topo_batches",
]
