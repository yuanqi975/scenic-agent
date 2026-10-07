"""Agent-facing data structures: cards, tasks, steps, delegations, results."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

#: Roles match the documented vocabulary: supervisor / worker / critic / responder.
ROLE_SUPERVISOR = "supervisor"
ROLE_WORKER = "worker"
ROLE_CRITIC = "critic"
ROLE_RESPONDER = "responder"


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


@dataclass(slots=True)
class AgentCard:
    """Declarative definition of one independent agent.

    ``tools`` is a whitelist, not documentation: the runtime refuses anything else.
    """

    name: str
    role: str
    description: str
    system_prompt: str
    tools: tuple[str, ...] = ()
    max_steps: int = 6
    parallelizable: bool = True
    requires_confirmation: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def allows(self, tool_name: str) -> bool:
        return tool_name in self.tools


@dataclass(slots=True)
class Task:
    """One delegated unit of work handed to a specialist."""

    agent: str
    instruction: str
    round_index: int = 0
    depends_on: tuple[str, ...] = ()
    parent_run_id: str | None = None
    task_id: str = field(default_factory=lambda: new_id("task"))

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "agent": self.agent,
            "instruction": self.instruction,
            "round_index": self.round_index,
            "depends_on": list(self.depends_on),
        }


@dataclass(slots=True)
class StepRecord:
    """One auditable step. Deliberately excludes hidden chain-of-thought."""

    run_id: str
    agent_name: str
    step_index: int
    event_type: str
    text_summary: str | None = None
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None
    tool_result_summary: dict[str, Any] | None = None
    status: str = "ok"
    latency_ms: int | None = None
    parent_run_id: str | None = None
    round_index: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "agent_name": self.agent_name,
            "step_index": self.step_index,
            "event_type": self.event_type,
            "text_summary": self.text_summary,
            "tool_name": self.tool_name,
            "tool_arguments": self.tool_arguments,
            "tool_result_summary": self.tool_result_summary,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "parent_run_id": self.parent_run_id,
            "round_index": self.round_index,
        }


@dataclass(slots=True)
class Delegation:
    parent_run_id: str
    from_agent: str
    to_agent: str
    instruction: str
    round_index: int = 0
    child_run_id: str | None = None
    status: str = "pending"

    def as_dict(self) -> dict[str, Any]:
        return {
            "parent_run_id": self.parent_run_id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "instruction": self.instruction,
            "round_index": self.round_index,
            "child_run_id": self.child_run_id,
            "status": self.status,
        }


@dataclass(slots=True)
class AgentResult:
    """What a specialist produced, plus the evidence needed to audit it."""

    agent: str
    run_id: str
    instruction: str = ""
    answer: str = ""
    status: str = "success"
    iterations: int = 0
    tool_calls: int = 0
    evidence: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    latency_ms: int = 0
    source: str = "model"

    @property
    def ok(self) -> bool:
        return self.status == "success" and bool(self.answer)

    def as_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "run_id": self.run_id,
            "instruction": self.instruction,
            "answer": self.answer,
            "status": self.status,
            "iterations": self.iterations,
            "tool_calls": self.tool_calls,
            "citations": self.citations,
            "artifacts": self.artifacts,
            "error": self.error,
            "latency_ms": self.latency_ms,
            "source": self.source,
        }


@dataclass(slots=True)
class RunTrace:
    """Aggregate record of one request: goal, plan, delegation tree, answer."""

    run_id: str
    conversation_id: str
    mode: str
    goal: str = ""
    plan: dict[str, Any] = field(default_factory=dict)
    agent_chain: list[dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    total_steps: int = 0
    total_tool_calls: int = 0
    total_latency_ms: int = 0
    degraded: bool = False
    status: str = "success"
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "conversation_id": self.conversation_id,
            "mode": self.mode,
            "goal": self.goal,
            "plan": self.plan,
            "agent_chain": self.agent_chain,
            "citation_count": len(self.citations),
            "total_steps": self.total_steps,
            "total_tool_calls": self.total_tool_calls,
            "total_latency_ms": self.total_latency_ms,
            "degraded": self.degraded,
            "status": self.status,
            "error": self.error,
        }
