"""Admin endpoints: operations dashboard, knowledge review, and agent observability."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel
from sqlalchemy import text

from ..agents.registry import AGENT_CARDS, SPECIALIST_AGENTS, naming_contract
from ..core.config import ENVIRONMENT, PARK_ID, resolved_agent_mode
from ..core.db import engine
from ..core.security import admin_from_token, authenticate, check_login_rate_limit, record_login_failure, reset_login_failures, revoke_token
from ..services import audit
from ..services.chunking import split_document
from ..tools.registry import permission_matrix, tool_names

router = APIRouter(tags=["admin"])


class LoginRequest(BaseModel):
    email: str
    password: str


@router.post("/admin/auth/login")
def admin_login(request: LoginRequest, http_request: Request, response: Response):
    identity = f"{http_request.client.host if http_request.client else 'unknown'}:{request.email.strip().lower()}"
    check_login_rate_limit(identity)
    token = authenticate(request.email, request.password)
    if not token:
        record_login_failure(identity)
        raise HTTPException(401, "邮箱或密码错误")
    reset_login_failures(identity)
    # The browser uses an HttpOnly session cookie; the token remains in the JSON
    # response only for backwards-compatible API clients and is never stored by UI.
    response.set_cookie(
        "admin_session",
        token["access_token"],
        max_age=8 * 60 * 60,
        httponly=True,
        secure=ENVIRONMENT in {"production", "prod"},
        samesite="lax",
        path="/",
    )
    return token


@router.post("/admin/auth/logout")
def admin_logout(response: Response, http_request: Request):
    token = http_request.cookies.get("admin_session")
    authorization = http_request.headers.get("authorization")
    if not token and authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1]
    if token:
        revoke_token(token)
    response.delete_cookie("admin_session", path="/")
    return {"ok": True}


@router.get("/admin/dashboard")
def dashboard(_: dict = Depends(admin_from_token)):
    with engine.connect() as conn:
        counts = {
            table: conn.execute(
                text(f"SELECT count(*) FROM {table} WHERE park_id=:park_id"), {"park_id": PARK_ID}
            ).scalar()
            for table in ("attractions", "facilities", "documents", "feedbacks", "agent_runs")
        }
        trace_count = conn.execute(
            text("SELECT count(*) FROM run_traces WHERE park_id=:park_id"), {"park_id": PARK_ID}
        ).scalar()
        step_count = conn.execute(
            text("SELECT count(*) FROM agent_steps WHERE park_id=:park_id"), {"park_id": PARK_ID}
        ).scalar()
    return {
        "park_id": PARK_ID,
        "counts": counts,
        "traces": trace_count,
        "agent_steps": step_count,
        "llm_mode": resolved_agent_mode(),
    }


@router.get("/admin/feedback-candidates")
def feedback_candidates(_: dict = Depends(admin_from_token), limit: int = Query(50, ge=1, le=200)):
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT payload FROM feedback_candidates WHERE park_id=:park_id ORDER BY (payload->>'created_at') DESC NULLS LAST, candidate_id DESC LIMIT :limit"
            ),
            {"park_id": PARK_ID, "limit": limit},
        ).all()
    return {"items": [row[0] for row in rows]}


@router.post("/admin/feedback-candidates/{candidate_id}/{action}")
def review_candidate(candidate_id: str, action: str, admin: dict = Depends(admin_from_token)):
    """Approve or reject a candidate. Publishing knowledge still requires this human step."""
    if action not in {"approve", "reject"}:
        raise HTTPException(400, "action 必须为 approve 或 reject")
    enqueue_after_commit: tuple[str, str] | None = None
    response: dict[str, Any] | None = None
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT payload FROM feedback_candidates WHERE park_id=:park_id AND candidate_id=:id"),
            {"park_id": PARK_ID, "id": candidate_id},
        ).first()
        if not row:
            raise HTTPException(404, "候选知识不存在")
        candidate = row[0] if isinstance(row[0], dict) else json.loads(row[0])
        next_status = "approved" if action == "approve" else "rejected"
        candidate["status"] = next_status
        conn.execute(
            text(
                "UPDATE feedback_candidates SET payload=CAST(:payload AS jsonb) WHERE park_id=:park_id AND candidate_id=:id"
            ),
            {"payload": json.dumps(candidate, ensure_ascii=False), "park_id": PARK_ID, "id": candidate_id},
        )
        conn.execute(
            text(
                "INSERT INTO knowledge_reviews(candidate_id,park_id,action,reviewer) VALUES (:id,:park_id,:action,:reviewer)"
            ),
            {"id": candidate_id, "park_id": PARK_ID, "action": action, "reviewer": admin.get("sub", "admin")},
        )
        if action == "approve":
            # A previous answer may have been cached before this publication. Clear
            # retrieval results so the newly approved feedback can be seen immediately
            # after its embedding task completes.
            from ..core.cache import cache_clear

            cache_clear(prefix="retrieval:v2:")
            document_id = f"feedback_review_{candidate_id}"
            existing_task = conn.execute(
                text(
                    "SELECT task_id FROM async_tasks WHERE park_id=:park_id AND task_type='embed_document' AND payload->>'document_id'=:document_id ORDER BY created_at ASC LIMIT 1"
                ),
                {"park_id": PARK_ID, "document_id": document_id},
            ).first()
            if existing_task:
                return {
                    "candidate_id": candidate_id,
                    "status": next_status,
                    "document_id": document_id,
                    "task_id": existing_task[0],
                }
            version = f"feedback-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{candidate_id[-4:]}"
            task_id = f"task_embed_{uuid.uuid4().hex[:16]}"
            content = f"审核通过的游客反馈（待人工核实）：{candidate.get('candidate_fact', '')}"
            metadata = {
                "source_type": "feedback_review",
                "source_id": candidate_id,
                "feedback_id": candidate.get("feedback_id"),
                "authority": "community",
                "approved_by": admin.get("sub", "admin"),
                "approved_at": datetime.now(timezone.utc).isoformat(),
            }
            conn.execute(
                text(
                    "INSERT INTO knowledge_versions(version,park_id,status) VALUES (:version,:park_id,'published') ON CONFLICT (version) DO NOTHING"
                ),
                {"version": version, "park_id": PARK_ID},
            )
            conn.execute(
                text(
                    """INSERT INTO documents(park_id,document_id,source_type,source_id,content,metadata,knowledge_version,updated_at)
                       VALUES (:park_id,:document_id,'feedback_review',:source_id,:content,CAST(:metadata AS jsonb),:version,now())
                       ON CONFLICT (park_id,document_id) DO NOTHING"""
                ),
                {
                    "park_id": PARK_ID,
                    "document_id": document_id,
                    "source_id": candidate_id,
                    "content": content,
                    "metadata": json.dumps(metadata, ensure_ascii=False),
                    "version": version,
                },
            )
            chunks = split_document(content)
            conn.execute(
                text(
                    """INSERT INTO document_chunks(park_id,document_id,chunk_index,content,updated_at)
                       VALUES (:park_id,:document_id,:chunk_index,:content,now())
                       ON CONFLICT (park_id,document_id,chunk_index) DO UPDATE SET content=EXCLUDED.content,updated_at=now()"""
                ),
                [
                    {
                        "park_id": PARK_ID,
                        "document_id": document_id,
                        "chunk_index": chunk.index,
                        "content": chunk.content,
                    }
                    for chunk in chunks
                ],
            )
            conn.execute(
                text(
                    "INSERT INTO async_tasks(task_id,park_id,task_type,status,payload) VALUES (:task_id,:park_id,'embed_document','pending',CAST(:payload AS jsonb))"
                ),
                {
                    "task_id": task_id,
                    "park_id": PARK_ID,
                    "payload": json.dumps({"document_id": document_id, "candidate_id": candidate_id}),
                },
            )
            # The endpoint is synchronous and runs in a threadpool, so there is no
            # running event loop to schedule from here. Queue after the transaction
            # commits so the worker can see the durable task row.
            enqueue_after_commit = (task_id, document_id)
            response = {
                "candidate_id": candidate_id,
                "status": next_status,
                "document_id": document_id,
                "task_id": task_id,
            }
    if enqueue_after_commit and response is not None:
        task_id, document_id = enqueue_after_commit
        queued = False
        try:
            from ..core.events import enqueue_embedding_task

            queued = bool(asyncio.run(enqueue_embedding_task(task_id, document_id)))
        except Exception:
            queued = False
        response["queued"] = queued
    return response or {"candidate_id": candidate_id, "status": next_status}


@router.get("/admin/agent-runs")
def agent_runs(_: dict = Depends(admin_from_token), limit: int = Query(50, ge=1, le=200)):
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """SELECT run_id, conversation_id, intent, agent_name, agent_role, parent_run_id,
                          trace_id, tool_call_count, iterations, retrieved_document_ids,
                          cache_hit, latency_ms, status, error, created_at
                   FROM agent_runs WHERE park_id=:park_id ORDER BY created_at DESC LIMIT :limit"""
            ),
            {"park_id": PARK_ID, "limit": limit},
        ).all()
    keys = (
        "run_id",
        "conversation_id",
        "intent",
        "agent_name",
        "agent_role",
        "parent_run_id",
        "trace_id",
        "tool_call_count",
        "iterations",
        "retrieved_document_ids",
        "cache_hit",
        "latency_ms",
        "status",
        "error",
        "created_at",
    )
    return {"items": [dict(zip(keys, row)) for row in rows]}


@router.get("/admin/agents")
def admin_agents(_: dict = Depends(admin_from_token)):
    """Agent roster, tool whitelists and live metrics."""
    return {
        "mode": resolved_agent_mode(),
        "park_id": PARK_ID,
        "naming_contract": naming_contract(),
        "tools": tool_names(),
        "permission_matrix": permission_matrix(),
        "items": audit.agent_metrics(),
        "specialists": list(SPECIALIST_AGENTS),
    }


@router.post("/admin/agents/{agent_name}/toggle")
def toggle_agent(agent_name: str, _: dict = Depends(admin_from_token)):
    """Enable or disable one agent for the *next* deployment of prompts.

    Prompts and tool whitelists live in code (see ``agents/registry.py``), so this only
    flips the operational flag used by dashboards and canary runs.
    """
    if agent_name not in AGENT_CARDS:
        raise HTTPException(404, "未注册的 Agent")
    try:
        with engine.begin() as conn:
            row = conn.execute(
                text("SELECT enabled FROM agent_registry WHERE agent_name=:name AND park_id=:park_id"),
                {"name": agent_name, "park_id": PARK_ID},
            ).first()
            if row is None:
                conn.execute(
                    text(
                        """INSERT INTO agent_registry(agent_name,park_id,role,enabled,updated_at)
                           VALUES (:name,:park_id,:role,false,now())"""
                    ),
                    {"name": agent_name, "park_id": PARK_ID, "role": AGENT_CARDS[agent_name].role},
                )
                enabled = False
            else:
                enabled = not row[0]
                conn.execute(
                    text(
                        "UPDATE agent_registry SET enabled=:enabled, updated_at=now() WHERE agent_name=:name AND park_id=:park_id"
                    ),
                    {"enabled": enabled, "name": agent_name, "park_id": PARK_ID},
                )
    except Exception as exc:
        raise HTTPException(503, f"无法更新 Agent 状态：{exc}") from exc
    return {"agent_name": agent_name, "enabled": enabled}


@router.get("/admin/traces/{trace_id}")
def admin_trace(trace_id: str, _: dict = Depends(admin_from_token)):
    """Full replay of one request: run tree, delegation edges and every step."""
    tree = audit.run_tree(trace_id)
    if not tree.get("trace") and not tree.get("runs"):
        raise HTTPException(404, "轨迹不存在")
    return tree


@router.get("/admin/documents")
def admin_documents(
    _: dict = Depends(admin_from_token),
    limit: int = Query(50, ge=1, le=200),
    source_type: str | None = None,
):
    statement = "SELECT document_id,source_type,source_id,content,metadata,knowledge_version,updated_at FROM documents WHERE park_id=:park_id"
    params: dict[str, Any] = {"park_id": PARK_ID, "limit": limit}
    if source_type:
        statement += " AND source_type=:source_type"
        params["source_type"] = source_type
    statement += " ORDER BY updated_at DESC LIMIT :limit"
    with engine.connect() as conn:
        rows = conn.execute(text(statement), params).all()
    return {
        "items": [
            {
                "document_id": row[0],
                "source_type": row[1],
                "source_id": row[2],
                "content": row[3],
                "metadata": row[4],
                "knowledge_version": row[5],
                "updated_at": row[6],
            }
            for row in rows
        ]
    }


@router.get("/admin/tasks/{task_id}")
def admin_task(task_id: str, _: dict = Depends(admin_from_token)):
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT task_id,task_type,status,payload,attempt_count,max_attempts,last_error,created_at,updated_at FROM async_tasks WHERE task_id=:id AND park_id=:park_id"
            ),
            {"id": task_id, "park_id": PARK_ID},
        ).first()
    if not row:
        raise HTTPException(404, "任务不存在")
    return {
        "task_id": row[0],
        "task_type": row[1],
        "status": row[2],
        "payload": row[3],
        "attempt_count": row[4],
        "max_attempts": row[5],
        "last_error": row[6],
        "created_at": row[7],
        "updated_at": row[8],
    }


@router.post("/admin/documents/{document_id}/publish")
def publish_document(document_id: str, _: dict = Depends(admin_from_token)):
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT knowledge_version FROM documents WHERE park_id=:park_id AND document_id=:document_id"),
            {"park_id": PARK_ID, "document_id": document_id},
        ).first()
        if not row:
            raise HTTPException(404, "文档不存在")
        conn.execute(
            text("UPDATE knowledge_versions SET status='published' WHERE park_id=:park_id AND version=:version"),
            {"park_id": PARK_ID, "version": row[0]},
        )
    return {"document_id": document_id, "status": "published", "knowledge_version": row[0]}


@router.post("/admin/documents/{document_id}/reindex")
def reindex_document(document_id: str, _: dict = Depends(admin_from_token)):
    with engine.begin() as conn:
        row = conn.execute(
            text(
                "SELECT content FROM document_chunks WHERE park_id=:park_id AND document_id=:document_id LIMIT 1"
            ),
            {"park_id": PARK_ID, "document_id": document_id},
        ).first()
        if not row:
            raise HTTPException(404, "文档分块不存在")
        existing = conn.execute(
            text(
                "SELECT task_id FROM async_tasks WHERE park_id=:park_id AND task_type='embed_document' AND payload->>'document_id'=:document_id AND status IN ('pending','running','retry') ORDER BY created_at DESC LIMIT 1"
            ),
            {"park_id": PARK_ID, "document_id": document_id},
        ).first()
        if existing:
            return {"document_id": document_id, "task_id": existing[0], "status": "already_queued"}
        task_id = f"task_embed_{uuid.uuid4().hex[:16]}"
        conn.execute(
            text(
                "INSERT INTO async_tasks(task_id,park_id,task_type,status,payload) VALUES (:task_id,:park_id,'embed_document','pending',CAST(:payload AS jsonb))"
            ),
            {"task_id": task_id, "park_id": PARK_ID, "payload": json.dumps({"document_id": document_id})},
        )
    queued = False
    try:
        from ..core.events import enqueue_embedding_task

        queued = bool(asyncio.run(enqueue_embedding_task(task_id, document_id)))
    except Exception:
        queued = False
    return {"document_id": document_id, "task_id": task_id, "status": "pending", "queued": queued}
