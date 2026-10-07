"""Knowledge base tools (read-only) - the RAG entry point for agents."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..core.config import RETRIEVAL_TOP_K
from ..core.db import payload_rows
from ..services.rag import RagSearchRequest, get_rag_service
from ..services.retrieval import citations_from, retrieve
from ..services.query_planner import plan_queries
from .registry import tool


class SearchKnowledgeArgs(BaseModel):
    query: str = Field(min_length=1, max_length=200, description="要检索的游客问题或关键词，尽量保留原始问法")
    top_k: int = Field(default=RETRIEVAL_TOP_K, ge=1, le=10, description="返回的资料来源条数")


class GetFaqsArgs(BaseModel):
    keyword: str | None = Field(default=None, max_length=60, description="FAQ 关键词，例如「门票」「退票」")
    topic: str | None = Field(default=None, max_length=40, description="FAQ 主题分类")
    official_only: bool = Field(default=True, description="是否只返回官方 FAQ")
    limit: int = Field(default=10, ge=1, le=50)


class RagSearchArgs(BaseModel):
    query: str = Field(min_length=1, max_length=200, description="需要检索的游客问题")
    top_k: int = Field(default=RETRIEVAL_TOP_K, ge=1, le=10)
    filters: dict[str, Any] = Field(default_factory=dict, description="park_id、authority、source_type 等过滤条件")


@tool(
    name="rag_search",
    description=(
        "统一混合 RAG 检索工具：在需要事实依据时检索景区知识，返回带权威性、召回通道、" 
        "引用和冲突标记的证据。门票、开放时间、限流、安全和路线规划必须调用。"
    ),
    args_schema=RagSearchArgs,
    allowed=("knowledge_agent", "route_agent", "realtime_agent", "feedback_agent"),
    tags=("read", "rag", "hybrid"),
)
async def rag_search(args: RagSearchArgs) -> dict[str, Any]:
    planned = plan_queries(args.query)
    filters = dict(args.filters)
    if len(planned) > 1:
        filters["planned_queries"] = planned
    result = await get_rag_service().search(
        RagSearchRequest(query=args.query, top_k=args.top_k, filters=filters)
    )
    items = result.get("items") or []
    return {
        "query": args.query,
        "backend": result.get("backend"),
        "confidence": result.get("confidence"),
        "evidence_score": result.get("evidence_score"),
        "grounding_status": result.get("grounding_status"),
        "abstention_required": result.get("abstention_required", False),
        "conflicts": result.get("conflicts") or [],
        "total": len(items),
        "items": items,
        "citations": citations_from(items),
    }


@tool(
    name="search_knowledge",
    description=(
        "检索九寨沟官方知识库（景点介绍、票价与优惠政策、开放时间、游览规则、官方问答）。"
        "返回带来源标识的原文片段，回答事实性问题必须使用它。"
    ),
    args_schema=SearchKnowledgeArgs,
    allowed=("knowledge_agent", "route_agent", "realtime_agent", "feedback_agent"),
    tags=("read", "rag"),
)
def search_knowledge(args: SearchKnowledgeArgs) -> dict[str, Any]:
    """Backward-compatible name for the canonical ``rag_search`` capability.

    The legacy PostgreSQL path is kept for tests and rollback mode; once shadow or
    Milvus is selected, this name delegates to the exact same hybrid service.
    """
    from ..core import config

    if config.RAG_BACKEND != "postgres":
        from ..services.rag import RagSearchRequest, get_rag_service

        # This sync tool is executed in a worker thread by the registry.
        import asyncio

        result = asyncio.run(get_rag_service().search(RagSearchRequest(query=args.query, top_k=args.top_k)))
        items = result.get("items") or []
        return {
            "query": args.query,
            "total": len(items),
            "items": items,
            "citations": citations_from(items),
            "backend": result.get("backend"),
            "confidence": result.get("confidence"),
            "conflicts": result.get("conflicts") or [],
        }
    items = retrieve(args.query, top_k=args.top_k)
    return {
        "query": args.query,
        "total": len(items),
        "items": [
            {
                "document_id": item.get("document_id"),
                "source_type": item.get("source_type"),
                "source_id": item.get("source_id"),
                "authority": item.get("authority"),
                "retrieval": item.get("retrieval"),
                "topic": (item.get("metadata") or {}).get("topic") if isinstance(item.get("metadata"), dict) else None,
                "content": item.get("content"),
            }
            for item in items
        ],
        "citations": citations_from(items),
    }


@tool(
    name="get_faqs",
    description="查询九寨沟官方问答库（按关键词或主题），适合回答高频票务与游览规定问题。",
    args_schema=GetFaqsArgs,
    allowed=("knowledge_agent", "route_agent"),
    tags=("read", "catalog"),
)
def get_faqs(args: GetFaqsArgs) -> dict[str, Any]:
    items = payload_rows("faqs", 200)
    if args.official_only:
        items = [item for item in items if item.get("official")]
    if args.topic:
        items = [item for item in items if args.topic in str(item.get("topic") or "")]
    if args.keyword:
        keyword = args.keyword
        items = [
            item
            for item in items
            if keyword in str(item.get("question") or "")
            or keyword in str(item.get("answer") or "")
            or keyword in "、".join(item.get("keywords") or [])
        ]
    return {
        "total": len(items),
        "items": [
            {
                "faq_id": item.get("faq_id"),
                "topic": item.get("topic"),
                "question": item.get("question"),
                "answer": item.get("answer"),
                "official": item.get("official"),
                "related_attraction_id": item.get("related_attraction_id"),
            }
            for item in items[: args.limit]
        ],
    }
