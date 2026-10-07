"""Audit trail for the multi-agent runtime.

Three levels of detail, all queryable after the fact:

* ``agent_runs``        - one row per agent execution (the existing table, extended)
* ``agent_steps``       - think / tool / observe / answer steps, without hidden CoT
* ``agent_delegations`` - who handed what to whom, which proves real collaboration

Two rules keep the audit trail from becoming a liability:

1. **Writers are defensive** - an audit failure must never break a visitor's answer.
2. **Writers never block the event loop** - every statement is handed to the
   background writer in :mod:`app.core.db_write`. psycopg is synchronous, so running
   these inline stalled every concurrent agent task for a whole connect timeout; see
   that module for the full reasoning.

Readers stay synchronous: they are only used by admin endpoints, never on the visitor
hot path.
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import text

from ..core import db_write
from ..core.config import PARK_ID
from ..core.db import enabled, engine
from ..agents.schemas import AgentResult, RunTrace, StepRecord, new_id


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def new_trace_id() -> str:
    """Root id that ties every run, step and delegation of one request together."""
    return new_id("trace")


def _submit(job: Any, key: str) -> None:
    """Queue one write unless the circuit breaker already knows the database is down."""
    if not enabled():
        return
    db_write.enqueue(db_write.AUDIT_QUEUE, job, key=key)


def _write_run_started(
    *,
    run_id: str,
    conversation_id: str | None,
    intent: str,
    agent_name: str,
    trace_id: str,
    parent_run_id: str | None,
    step_index: int,
    role: str,
    goal: str,
    instruction: str,
    model: str | None,
    mode: str,
) -> None:
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    """INSERT INTO agent_runs(
                           run_id, conversation_id, park_id, intent, agent_name,
                           retrieved_document_ids, cache_hit, latency_ms, status, error,
                           trace_id, parent_run_id, step_index, agent_role, tool_calls,
                           goal, instruction, model, mode, iterations, tool_call_count,
                           started_at, created_at
                       ) VALUES (
                           :run_id,:conversation_id,:park_id,:intent,:agent_name,
                           '[]'::jsonb,false,0,'running',NULL,
                           :trace_id,:parent_run_id,:step_index,:agent_role,'[]'::jsonb,
                           :goal,:instruction,:model,:mode,0,0,
                           now(),now()
                       )"""
                ),
                {
                    "run_id": run_id,
                    "conversation_id": conversation_id,
                    "park_id": PARK_ID,
                    "intent": intent,
                    "agent_name": agent_name,
                    "trace_id": trace_id,
                    "parent_run_id": parent_run_id,
                    "step_index": step_index,
                    "agent_role": role,
                    "goal": goal,
                    "instruction": instruction,
                    "model": model,
                    "mode": mode,
                },
            )
            _bump_registry(conn, agent_name, role, calls=1, steps=0)
    except Exception:
        db_failure()


def db_failure() -> None:
    """Record a real connection failure so the writer stops retrying per job.

    Wrapped so audit can never raise: the circuit breaker is an optimisation, and a bug
    in it must not surface to a visitor.
    """
    try:
        db_write.note_write_failure()
    except Exception:  # pragma: no cover - defensive
        pass


def start_run(
    *,
    agent_name: str,
    role: str,
    conversation_id: str | None,
    instruction: str = "",
    goal: str = "",
    parent_run_id: str | None = None,
    trace_id: str | None = None,
    intent: str = "unknown",
    mode: str = "fallback",
    model: str | None = None,
    step_index: int = 0,
    round_index: int = 0,
) -> str:
    """Open one agent run and return its ``run_id``.

    The id is generated locally and returned immediately; persistence happens on the
    writer thread, so a slow database cannot delay the agent it is describing.
    """
    run_id = new_id("run")
    _submit(
        lambda: _write_run_started(
            run_id=run_id,
            conversation_id=conversation_id,
            intent=intent,
            agent_name=agent_name,
            trace_id=trace_id or run_id,
            parent_run_id=parent_run_id,
            step_index=step_index,
            role=role,
            goal=goal,
            instruction=instruction,
            model=model,
            mode=mode,
        ),
        "agent_run_start",
    )
    return run_id


def _write_run_finished(result: AgentResult, role: str) -> None:
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    """UPDATE agent_runs SET
                           status=:status, error=:error, latency_ms=:latency,
                           iterations=:iterations, tool_call_count=:tool_calls,
                           retrieved_document_ids=CAST(:docs AS jsonb),
                           tool_calls=CAST(:tool_names AS jsonb),
                           completed_at=now()
                       WHERE run_id=:run_id"""
                ),
                {
                    "run_id": result.run_id,
                    "status": result.status,
                    "error": result.error,
                    "latency": result.latency_ms,
                    "iterations": result.iterations,
                    "tool_calls": result.tool_calls,
                    "docs": _json([str(c.get("document_id") or c.get("source_id")) for c in result.citations]),
                    "tool_names": _json(
                        sorted({str(step.get("tool_name")) for step in result.evidence if step.get("tool_name")})
                    ),
                },
            )
            _bump_registry(
                conn,
                result.agent,
                role,
                calls=0,
                steps=result.iterations,
                latency=result.latency_ms,
                errors=0 if result.status == "success" else 1,
            )
    except Exception:
        db_failure()


def finish_run(result: AgentResult, *, role: str, mode: str = "fallback") -> None:
    """Close one agent run with its outcome."""
    _submit(lambda: _write_run_finished(result, role), "agent_run_finish")


def _write_step(record: dict[str, Any]) -> None:
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    """INSERT INTO agent_steps(
                           run_id, park_id, agent_name, step_index, event_type, text_summary,
                           tool_name, tool_arguments, tool_result_summary, status, latency_ms, created_at
                       ) VALUES (
                           :run_id,:park_id,:agent_name,:step_index,:event_type,:text_summary,
                           :tool_name,CAST(:tool_arguments AS jsonb),CAST(:tool_result AS jsonb),:status,:latency_ms,now()
                       )"""
                ),
                {
                    "run_id": record.get("run_id"),
                    "park_id": PARK_ID,
                    "agent_name": record.get("agent_name") or "unknown",
                    "step_index": record.get("step_index", 0),
                    "event_type": record.get("event_type", "thought"),
                    "text_summary": (record.get("text_summary") or None),
                    "tool_name": record.get("tool_name"),
                    "tool_arguments": _json(record.get("tool_arguments") or {}),
                    "tool_result": _json(record.get("tool_result_summary") or {}),
                    "status": record.get("status", "ok"),
                    "latency_ms": record.get("latency_ms"),
                },
            )
    except Exception:
        db_failure()


def save_step(step: StepRecord | dict[str, Any]) -> None:
    """Persist one step of an agent's ReAct loop.

    The record is copied *before* it is queued: the caller may keep mutating its own
    dict, and the writer thread runs later.
    """
    record = step.as_dict() if isinstance(step, StepRecord) else dict(step)
    _submit(lambda: _write_step(dict(record)), "agent_step")


def _write_delegation(payload: dict[str, Any]) -> None:
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    """INSERT INTO agent_delegations(
                           park_id, parent_run_id, child_run_id, from_agent, to_agent,
                           instruction, round_index, status, created_at
                       ) VALUES (:park_id,:parent,:child,:from_agent,:to_agent,:instruction,:round,:status,now())"""
                ),
                payload,
            )
    except Exception:
        db_failure()


def save_delegation(
    *,
    parent_run_id: str,
    from_agent: str,
    to_agent: str,
    instruction: str,
    round_index: int = 0,
    child_run_id: str | None = None,
    status: str = "pending",
) -> None:
    payload = {
        "park_id": PARK_ID,
        "parent": parent_run_id,
        "child": child_run_id,
        "from_agent": from_agent,
        "to_agent": to_agent,
        "instruction": instruction,
        "round": round_index,
        "status": status,
    }
    _submit(lambda: _write_delegation(dict(payload)), "agent_delegation")


def _write_attach_child(parent_run_id: str, child_run_id: str, to_agent: str) -> None:
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    """UPDATE agent_delegations SET child_run_id=:child, status='dispatched'
                       WHERE parent_run_id=:parent AND to_agent=:agent AND child_run_id IS NULL"""
                ),
                {"parent": parent_run_id, "child": child_run_id, "agent": to_agent},
            )
    except Exception:
        db_failure()


def attach_child_run(parent_run_id: str, child_run_id: str, to_agent: str) -> None:
    _submit(lambda: _write_attach_child(parent_run_id, child_run_id, to_agent), "agent_delegation_attach")


def _write_trace(payload: dict[str, Any]) -> None:
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    """INSERT INTO run_traces(
                           run_id, park_id, conversation_id, mode, goal, plan, agent_chain,
                           final_answer, citations, total_steps, total_tool_calls,
                           total_latency_ms, degraded, status, error, created_at
                       ) VALUES (
                           :run_id,:park_id,:conversation_id,:mode,:goal,CAST(:plan AS jsonb),
                           CAST(:chain AS jsonb),:answer,CAST(:citations AS jsonb),
                           :steps,:tool_calls,:latency,:degraded,:status,:error,now()
                       )
                       ON CONFLICT (run_id) DO UPDATE SET
                           final_answer=EXCLUDED.final_answer, citations=EXCLUDED.citations,
                           agent_chain=EXCLUDED.agent_chain, plan=EXCLUDED.plan,
                           total_steps=EXCLUDED.total_steps, total_tool_calls=EXCLUDED.total_tool_calls,
                           total_latency_ms=EXCLUDED.total_latency_ms, status=EXCLUDED.status,
                           error=EXCLUDED.error, degraded=EXCLUDED.degraded"""
                ),
                payload,
            )
    except Exception:
        db_failure()


def save_trace(trace: RunTrace) -> None:
    payload = {
        "run_id": trace.run_id,
        "park_id": PARK_ID,
        "conversation_id": trace.conversation_id,
        "mode": trace.mode,
        "goal": trace.goal,
        "plan": _json(trace.plan),
        "chain": _json(trace.agent_chain),
        "answer": trace.final_answer,
        "citations": _json(trace.citations),
        "steps": trace.total_steps,
        "tool_calls": trace.total_tool_calls,
        "latency": trace.total_latency_ms,
        "degraded": trace.degraded,
        "status": trace.status,
        "error": trace.error,
    }
    _submit(lambda: _write_trace(dict(payload)), "run_trace")


def _bump_registry(
    conn: Any,
    agent_name: str,
    role: str,
    *,
    calls: int = 0,
    steps: int = 0,
    latency: int = 0,
    errors: int = 0,
) -> None:
    conn.execute(
        text(
            """INSERT INTO agent_registry(agent_name, park_id, role, enabled, call_count, step_count, error_count, total_latency_ms, updated_at)
               VALUES (:agent_name,:park_id,:role,true,:calls,:steps,:errors,:latency,now())
               ON CONFLICT (agent_name, park_id) DO UPDATE SET
                   call_count = agent_registry.call_count + :calls,
                   step_count = agent_registry.step_count + :steps,
                   error_count = agent_registry.error_count + :errors,
                   total_latency_ms = agent_registry.total_latency_ms + :latency,
                   updated_at = now()"""
        ),
        {
            "agent_name": agent_name,
            "park_id": PARK_ID,
            "role": role,
            "calls": calls,
            "steps": steps,
            "errors": errors,
            "latency": latency,
        },
    )


# --------------------------------------------------------------------------- readers
def run_tree(trace_id: str) -> dict[str, Any]:
    """Reassemble one request: run rows, delegation edges and every step."""
    runs: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    delegations: list[dict[str, Any]] = []
    trace: dict[str, Any] | None = None
    if not enabled():
        return {
            "trace_id": trace_id,
            "trace": None,
            "runs": [],
            "steps": [],
            "delegations": [],
            "tree": {"roots": []},
        }
    try:
        with engine.connect() as conn:
            trace_row = conn.execute(
                text("""SELECT run_id, conversation_id, mode, goal, plan, agent_chain, final_answer,
                               citations, total_steps, total_tool_calls, total_latency_ms,
                               degraded, status, error, created_at
                        FROM run_traces WHERE park_id=:park_id AND run_id=:run_id"""),
                {"park_id": PARK_ID, "run_id": trace_id},
            ).first()
            if trace_row:
                trace = {
                    "run_id": trace_row[0],
                    "conversation_id": trace_row[1],
                    "mode": trace_row[2],
                    "goal": trace_row[3],
                    "plan": trace_row[4],
                    "agent_chain": trace_row[5],
                    "final_answer": trace_row[6],
                    "citations": trace_row[7],
                    "total_steps": trace_row[8],
                    "total_tool_calls": trace_row[9],
                    "total_latency_ms": trace_row[10],
                    "degraded": trace_row[11],
                    "status": trace_row[12],
                    "error": trace_row[13],
                    "created_at": str(trace_row[14]),
                }
            run_rows = conn.execute(
                text(
                    """SELECT run_id, agent_name, agent_role, parent_run_id, instruction, goal,
                              status, iterations, tool_call_count, latency_ms, error, started_at, completed_at
                       FROM agent_runs WHERE park_id=:park_id AND trace_id=:run_id
                       ORDER BY started_at NULLS LAST, run_id"""
                ),
                {"park_id": PARK_ID, "run_id": trace_id},
            ).fetchall()
            runs = [
                {
                    "run_id": row[0],
                    "agent_name": row[1],
                    "agent_role": row[2],
                    "parent_run_id": row[3],
                    "instruction": row[4],
                    "goal": row[5],
                    "status": row[6],
                    "iterations": row[7],
                    "tool_call_count": row[8],
                    "latency_ms": row[9],
                    "error": row[10],
                    "started_at": str(row[11]) if row[11] else None,
                    "completed_at": str(row[12]) if row[12] else None,
                }
                for row in run_rows
            ]
            step_rows = conn.execute(
                text(
                    """SELECT run_id, agent_name, step_index, event_type, text_summary, tool_name,
                              tool_arguments, tool_result_summary, status, latency_ms, created_at
                       FROM agent_steps WHERE park_id=:park_id AND run_id = ANY(:run_ids)
                       ORDER BY step_index, step_id"""
                ),
                {"park_id": PARK_ID, "run_ids": [run["run_id"] for run in runs] or [trace_id]},
            ).fetchall()
            steps = [
                {
                    "run_id": row[0],
                    "agent_name": row[1],
                    "step_index": row[2],
                    "event_type": row[3],
                    "text_summary": row[4],
                    "tool_name": row[5],
                    "tool_arguments": row[6],
                    "tool_result_summary": row[7],
                    "status": row[8],
                    "latency_ms": row[9],
                    "created_at": str(row[10]),
                }
                for row in step_rows
            ]
            delegation_rows = conn.execute(
                text(
                    """SELECT parent_run_id, child_run_id, from_agent, to_agent, instruction,
                              round_index, status, created_at
                       FROM agent_delegations WHERE park_id=:park_id AND parent_run_id=:run_id
                       ORDER BY round_index, delegation_id"""
                ),
                {"park_id": PARK_ID, "run_id": trace_id},
            ).fetchall()
            delegations = [
                {
                    "parent_run_id": row[0],
                    "child_run_id": row[1],
                    "from_agent": row[2],
                    "to_agent": row[3],
                    "instruction": row[4],
                    "round_index": row[5],
                    "status": row[6],
                    "created_at": str(row[7]),
                }
                for row in delegation_rows
            ]
    except Exception:
        pass

    return {
        "trace_id": trace_id,
        "trace": trace,
        "runs": runs,
        "steps": steps,
        "delegations": delegations,
        "tree": _build_tree(runs, delegations),
    }


def _build_tree(runs: list[dict[str, Any]], delegations: list[dict[str, Any]]) -> dict[str, Any]:
    """Nested view used by the admin trace viewer."""
    children: dict[str, list[dict[str, Any]]] = {}
    for delegation in delegations:
        children.setdefault(delegation["parent_run_id"], []).append(delegation)
    by_id = {run["run_id"]: run for run in runs}

    def walk(run_id: str, depth: int = 0, seen: set[str] | None = None) -> dict[str, Any]:
        seen = seen or set()
        if run_id in seen or depth > 6:
            return {"run_id": run_id, "children": []}
        seen.add(run_id)
        run = by_id.get(run_id, {"run_id": run_id})
        nodes = []
        for delegation in children.get(run_id, []):
            child_id = delegation.get("child_run_id")
            node: dict[str, Any] = {
                "to_agent": delegation["to_agent"],
                "instruction": delegation.get("instruction"),
                "status": delegation.get("status"),
                "round_index": delegation.get("round_index"),
                "run": by_id.get(child_id) if child_id else None,
                "children": walk(child_id, depth + 1, seen)["children"] if child_id else [],
            }
            nodes.append(node)
        return {
            "run_id": run_id,
            "agent_name": run.get("agent_name"),
            "agent_role": run.get("agent_role"),
            "status": run.get("status"),
            "latency_ms": run.get("latency_ms"),
            "children": nodes,
        }

    roots = [run for run in runs if not run.get("parent_run_id")]
    if not roots and runs:
        roots = runs[:1]
    return {"roots": [walk(run["run_id"]) for run in roots]}


def conversation_traces(conversation_id: str, limit: int = 20) -> list[dict[str, Any]]:
    if not enabled():
        return []
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """SELECT run_id, mode, goal, agent_chain, total_steps, total_tool_calls,
                              total_latency_ms, degraded, status, created_at
                       FROM run_traces WHERE park_id=:park_id AND conversation_id=:cid
                       ORDER BY created_at DESC LIMIT :limit"""
                ),
                {"park_id": PARK_ID, "cid": conversation_id, "limit": limit},
            ).fetchall()
    except Exception:
        return []
    return [
        {
            "run_id": row[0],
            "mode": row[1],
            "goal": row[2],
            "agent_chain": row[3],
            "total_steps": row[4],
            "total_tool_calls": row[5],
            "total_latency_ms": row[6],
            "degraded": row[7],
            "status": row[8],
            "created_at": str(row[9]),
        }
        for row in rows
    ]


def agent_metrics() -> list[dict[str, Any]]:
    """Per-agent call volume, step counts and error rate for the admin dashboard."""
    from .registry import AGENT_CARDS

    stored: dict[str, dict[str, Any]] = {}
    if enabled():
        try:
            with engine.connect() as conn:
                rows = conn.execute(
                    text(
                        """SELECT agent_name, role, enabled, call_count, step_count, error_count, total_latency_ms
                           FROM agent_registry WHERE park_id=:park_id"""
                    ),
                    {"park_id": PARK_ID},
                ).fetchall()
        except Exception:
            rows = []
        stored = {
            row[0]: {
                "role": row[1],
                "enabled": row[2],
                "call_count": row[3],
                "step_count": row[4],
                "error_count": row[5],
                "avg_latency_ms": int(row[6] / row[3]) if row[3] else 0,
            }
            for row in rows
        }
    metrics = []
    for card in AGENT_CARDS.values():
        record = stored.get(card.name, {})
        calls = record.get("call_count", 0)
        metrics.append(
            {
                "agent_name": card.name,
                "role": card.role,
                "description": card.description,
                "tools": list(card.tools),
                "max_steps": card.max_steps,
                "parallelizable": card.parallelizable,
                "enabled": record.get("enabled", True),
                "call_count": calls,
                "avg_steps": round(record.get("step_count", 0) / calls, 2) if calls else 0,
                "error_count": record.get("error_count", 0),
                "error_rate": round(record.get("error_count", 0) / calls, 3) if calls else 0,
                "avg_latency_ms": record.get("avg_latency_ms", 0),
            }
        )
    return metrics
