"""Chat endpoints: SSE streaming and the synchronous fallback.

Event contract
--------------
The four historical events (``status`` / ``citations`` / ``token`` / ``result``) keep
working unchanged, so the existing frontend needs no update. The multi-agent events are
strictly additive, and each one also emits a human ``status`` line so an old client
still shows meaningful progress text.
"""

from __future__ import annotations

import asyncio
import json
import contextlib
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..core.cache import enforce_public_rate_limit
from ..core.config import AGENT_REQUEST_TIMEOUT_SECONDS, SSE_MAX_CONCURRENCY
from ..core import db_write
from ..services.conversations import conversation_history
from ..services.orchestrator import answer_async
from ..services import audit

router = APIRouter(tags=["chat"])
_stream_slots = asyncio.Semaphore(max(1, SSE_MAX_CONCURRENCY))

#: Agent name -> visitor-facing progress text (kept for the legacy ``status`` event).
AGENT_LABELS = {
    "supervisor": "正在分析你的需求",
    "knowledge_agent": "正在查阅景区知识库",
    "route_agent": "路线规划 Agent 正在筛选景点",
    "realtime_agent": "实时信息 Agent 正在检查开放状态",
    "feedback_agent": "正在受理你的反馈",
    "response_agent": "正在汇总答案",
    "critic_agent": "正在质检答案准确性",
}

TOOL_LABELS = {
    "search_knowledge": "正在检索景区知识",
    "search_attractions": "正在筛选景点",
    "get_attraction_detail": "正在读取景点详情",
    "get_facilities": "正在查询设施",
    "find_facilities_near": "正在查找景点附近设施",
    "get_faqs": "正在查询官方问答",
    "get_realtime_status": "正在检查实时状态",
    "get_park_notices": "正在查询景区公告",
    "calculate_route": "正在计算路线",
    "validate_route": "正在检查路线时长",
    "create_feedback_candidate": "正在登记反馈",
    "check_pending_feedback": "正在核对反馈状态",
    "get_conversation_history": "正在读取对话上下文",
}


class ChatRequest(BaseModel):
    conversation_id: str | None = None
    message: str = Field(min_length=1, max_length=2000)
    # ---- additive, optional
    mode: str | None = Field(default=None, description="auto / multi / single，用于对照实验")
    debug_trace: bool = Field(default=False, description="在 result 事件中附带完整轨迹")


def _sse(event: str, payload: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _legacy_status(event: str, data: dict[str, Any]) -> str | None:
    """Map a new event onto the historical human-readable progress line."""
    if event == "run_started":
        return "正在识别问题"
    if event == "agent_started":
        label = AGENT_LABELS.get(str(data.get("agent")), "正在处理")
        return f"{label}"
    if event == "tool_started":
        return TOOL_LABELS.get(str(data.get("tool")), "正在查询景区数据")
    if event == "agent_finished" and data.get("agent") == "route_agent":
        return "路线生成完成"
    if event == "agent_failed":
        return f"{data.get('agent')} 处理失败，正在降级处理"
    return None


async def _answer_with_deadline(
    message: str,
    conversation_id: str | None,
    *,
    emitter=None,
    mode: str | None = None,
) -> dict[str, Any]:
    """Bound the whole visitor request, including model queueing and tool loops."""
    try:
        return await asyncio.wait_for(
            answer_async(message, conversation_id, emitter=emitter, mode=mode),
            timeout=AGENT_REQUEST_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        return {
            "conversation_id": conversation_id,
            "message": "模型服务响应超时，已返回降级结果，请稍后重试。",
            "citations": [],
            "intent": "unknown",
            "status": "degraded",
            "error": "request_timeout",
            "mode": "fallback",
            "degraded": True,
        }


@router.post("/chat/messages")
async def chat(request: ChatRequest, _: None = Depends(enforce_public_rate_limit)):
    """Synchronous问答. Still works without a model: the rule brain takes over."""
    return await _answer_with_deadline(request.message, request.conversation_id, mode=request.mode)


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest, _: None = Depends(enforce_public_rate_limit)):
    """Server-sent events: progress, agent轨迹, citations, tokens, final result."""
    try:
        await asyncio.wait_for(_stream_slots.acquire(), timeout=0.1)
    except asyncio.TimeoutError:
        raise HTTPException(503, "流式连接繁忙，请稍后重试", headers={"Retry-After": "1"})

    async def events() -> AsyncIterator[str]:
        queue: asyncio.Queue[tuple[str, dict[str, Any]] | None] = asyncio.Queue()

        # Emit the legacy first progress line synchronously.  Direct structured
        # answers may finish before an async ``run_started`` callback gets a
        # chance to run, which otherwise leaves old clients without any status
        # event at all.
        yield _sse("status", "正在识别问题")

        async def emit(event: str, data: dict[str, Any]) -> None:
            await queue.put((event, data))

        async def run() -> None:
            try:
                result = await _answer_with_deadline(
                    request.message,
                    request.conversation_id,
                    emitter=emit,
                    mode=request.mode,
                )
                await queue.put(("__result__", result))
            except Exception as exc:  # never leave the stream hanging
                await queue.put(
                    (
                        "__result__",
                        {
                            "conversation_id": request.conversation_id,
                            "message": f"请求处理失败：{type(exc).__name__}",
                            "citations": [],
                            "intent": "unknown",
                            "status": "error",
                            "error": str(exc)[:300],
                            "mode": "fallback",
                        },
                    )
                )
            finally:
                await queue.put(None)

        task = asyncio.create_task(run())
        deadline = asyncio.get_running_loop().time() + AGENT_REQUEST_TIMEOUT_SECONDS + 5
        try:
            while True:
                try:
                    remaining = max(0.001, deadline - asyncio.get_running_loop().time())
                    item = await asyncio.wait_for(queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    yield _sse("citations", [])
                    yield _sse("result", {"conversation_id": request.conversation_id, "message": "处理超时，请稍后重试", "citations": [], "status": "error", "error": "request_timeout", "degraded": True})
                    break
                if item is None:
                    break
                event, data = item
                if event == "__result__":
                    result = data
                    yield _sse("citations", result.get("citations", []))
                    for token in str(result.get("message") or "").splitlines(True):
                        yield _sse("token", token)
                    if request.debug_trace:
                        # Audit writes are queued to a background writer, so flush the
                        # queue before rebuilding the tree - otherwise a debug trace
                        # would race its own writes and come back empty.
                        await asyncio.to_thread(db_write.drain, db_write.AUDIT_QUEUE, timeout=2.0)
                        # ``run_tree`` is synchronous and runs four queries; reading it on
                        # the loop would freeze every other in-flight request.
                        trace_id = result.get("trace_id", "")
                        result = {**result, "trace": await asyncio.to_thread(audit.run_tree, trace_id)}
                    yield _sse("result", result)
                    continue
                legacy = _legacy_status(event, data)
                if legacy:
                    yield _sse("status", legacy)
                yield _sse(event, data)
        finally:
            if not task.done():
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            _stream_slots.release()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str):
    history = conversation_history(conversation_id)
    if not history:
        from fastapi import HTTPException

        raise HTTPException(404, "会话不存在")
    return history


@router.get("/conversations/{conversation_id}/trace")
def get_conversation_trace(conversation_id: str, limit: int = 20):
    """Every multi-agent run of one conversation, newest first."""
    return {"conversation_id": conversation_id, "items": audit.conversation_traces(conversation_id, limit)}
