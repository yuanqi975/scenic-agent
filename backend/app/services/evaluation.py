"""Offline retrieval and grounded-answer evaluation helpers.

The evaluator deliberately accepts both the legacy single-document field and the
multi-document grading shape.  It returns enough per-question detail to inspect
regressions instead of collapsing every failure into one aggregate number.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable

from .rag import RagSearchRequest, get_rag_service


def _item_aliases(item: dict[str, Any]) -> set[str]:
    """Return document/chunk/source aliases used by different retrievers.

    Milvus normally returns ``document_id``.  Older collections and the PostgreSQL
    fallback may only return a chunk id (for example ``doc_00065:chunk_0``), while
    hand-authored evaluation sets often use a source id.  Treating these as aliases
    keeps retrieval metrics about evidence rather than storage implementation details.
    """
    aliases = {
        str(item.get(key)).strip()
        for key in ("document_id", "chunk_id", "source_id")
        if item.get(key)
    }
    for value in tuple(aliases):
        for separator in (":", "#", "/"):
            if separator in value:
                aliases.add(value.split(separator, 1)[0])
    return aliases


def _retrieval_channels(items: list[dict[str, Any]]) -> set[str]:
    channels: set[str] = set()
    for item in items:
        raw = str(item.get("retrieval") or "")
        channels.update(part for part in raw.split("+") if part)
    return channels


async def evaluate_questions(questions: list[dict[str, Any]], *, top_k: int = 5) -> dict[str, Any]:
    """Evaluate retrieval quality for a list of question records.

    Supported labels are ``relevant_document_ids``, ``expected_document_ids``,
    ``document_ids`` and the generated dataset's ``expected_document_id``.  A
    question with no usable label is reported as invalid rather than silently
    disappearing from the denominator.
    """
    total = 0
    hits = 0
    details: list[dict[str, Any]] = []
    invalid = 0
    abstention_expected = 0
    abstention_evaluated = 0
    abstention_correct = 0
    service = get_rag_service()
    for question in questions:
        query = str(question.get("question") or "")
        expected_values = (
            question.get("relevant_document_ids")
            or question.get("expected_document_ids")
            or question.get("document_ids")
            or ([question.get("expected_document_id")] if question.get("expected_document_id") else [])
        )
        expected = {str(value) for value in expected_values if value}
        should_abstain = bool(question.get("should_abstain", False))
        if not query or (not expected and not should_abstain):
            invalid += 1
            continue
        total += 1
        result = await service.search(RagSearchRequest(query=query, top_k=top_k))
        items = list(result.get("items") or [])
        aliases = [_item_aliases(item) for item in items]
        ranked = [next(iter(alias), "") for alias in aliases]
        matched = any(alias & expected for alias in aliases)
        abstention_expected += int(should_abstain)
        abstention_evaluated += 1
        abstention_correct += int(bool(result.get("abstention_required")) == should_abstain)
        hits += int(matched)
        first_rank = next((index for index, alias in enumerate(aliases, start=1) if alias & expected), None)
        relevance = question.get("graded_relevance") or {document_id: 1 for document_id in expected}
        gains = [
            max((float(relevance.get(document_id, 1)) for document_id in alias if document_id in relevance), default=0.0)
            for alias in aliases
        ]
        ideal = sorted((float(value) for value in relevance.values()), reverse=True)[:top_k]
        dcg = sum(gain / math.log2(index + 1) for index, gain in enumerate(gains[:top_k], start=1))
        idcg = sum(gain / math.log2(index + 1) for index, gain in enumerate(ideal, start=1))
        retrieval_channels = sorted(_retrieval_channels(items))
        expected_channels = {str(value) for value in (question.get("expected_channels") or [])}
        channel_match = sorted(expected_channels & set(retrieval_channels))
        details.append(
            {
                "question_id": question.get("question_id"),
                "question_type": question.get("question_type") or question.get("intent") or "unknown",
                "matched": matched,
                "rank": first_rank,
                "precision_at_k": sum(bool(alias & expected) for alias in aliases[:top_k]) / max(1, top_k),
                "recall_at_k": int(matched),
                "mrr_at_k": 1 / first_rank if first_rank else 0.0,
                "ndcg_at_k": dcg / idcg if idcg else 0.0,
                "should_abstain": should_abstain,
                "abstention_correct": bool(result.get("abstention_required")) == should_abstain,
                "expected_channels": sorted(expected_channels),
                "matched_channels": channel_match,
                "evaluation_split": question.get("evaluation_split", "baseline"),
                "backend": result.get("backend"),
                "retrieval_channels": retrieval_channels,
            }
        )

    def mean(key: str, rows: Iterable[dict[str, Any]]) -> float:
        values = [float(row[key]) for row in rows]
        return round(sum(values) / len(values), 6) if values else 0.0

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for detail in details:
        groups[str(detail["question_type"])].append(detail)
    return {
        "questions": total,
        "invalid_questions": invalid,
        "top_k": top_k,
        "recall_at_k": mean("recall_at_k", details),
        "precision_at_k": mean("precision_at_k", details),
        "mrr_at_k": mean("mrr_at_k", details),
        "ndcg_at_k": mean("ndcg_at_k", details),
        "abstention_questions": abstention_expected,
        "abstention_evaluated": abstention_evaluated,
        "abstention_accuracy": round(abstention_correct / abstention_evaluated, 6) if abstention_evaluated else None,
        "channel_hits": {
            channel: sum(channel in row.get("matched_channels", []) for row in details)
            for channel in ("dense", "sparse", "entity", "structured")
        },
        "by_question_type": {
            name: {
                "questions": len(rows),
                "recall_at_k": mean("recall_at_k", rows),
                "precision_at_k": mean("precision_at_k", rows),
                "mrr_at_k": mean("mrr_at_k", rows),
                "ndcg_at_k": mean("ndcg_at_k", rows),
            }
            for name, rows in sorted(groups.items())
        },
        "details": details,
    }


def keyword_coverage(answer: str, keywords: Iterable[str]) -> float:
    """Return the fraction of required answer keywords explicitly present."""
    required = [str(keyword).strip() for keyword in keywords if str(keyword).strip()]
    if not required:
        return 1.0
    text = answer or ""
    return round(sum(keyword in text for keyword in required) / len(required), 6)
