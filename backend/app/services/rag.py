"""Backend-independent hybrid RAG service.

The public contract is deliberately small: agents ask for evidence through this
module and never know whether the evidence came from PostgreSQL, Milvus, or a
fallback.  Milvus is optional so local/fallback development remains useful.
"""

from __future__ import annotations

import asyncio
import ast
import hashlib
import inspect
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Protocol

from ..core import config
from ..core.cache import cache_get, cache_set


@dataclass(slots=True)
class RagSearchRequest:
    query: str
    top_k: int = config.RETRIEVAL_TOP_K
    filters: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RagHit:
    chunk_id: str
    content: str
    score: float
    channel: str
    document_id: str | None = None
    source_type: str | None = None
    source_id: str | None = None
    authority: str = "reference"
    metadata: dict[str, Any] = field(default_factory=dict)
    channels: list[str] = field(default_factory=list)
    updated_at: str | None = None

    def __post_init__(self) -> None:
        if not self.channels:
            self.channels = [self.channel]
        if self.updated_at is None and isinstance(self.metadata, dict):
            raw = self.metadata.get("updated_at") or self.metadata.get("published_at")
            if raw is not None:
                self.updated_at = str(raw)


class Retriever(Protocol):
    def search(self, request: RagSearchRequest) -> Any: ...


def _as_hit(value: RagHit | dict[str, Any], channel: str) -> RagHit:
    if isinstance(value, RagHit):
        if channel not in value.channels:
            value.channels.append(channel)
        return value
    metadata = value.get("metadata") if isinstance(value.get("metadata"), dict) else {}
    return RagHit(
        chunk_id=str(value.get("chunk_id") or value.get("document_id") or value.get("source_id") or id(value)),
        document_id=value.get("document_id"),
        content=str(value.get("content") or ""),
        score=float(value.get("score") or value.get("distance") or 0.0),
        channel=channel,
        source_type=value.get("source_type"),
        source_id=value.get("source_id"),
        authority=str(value.get("authority") or metadata.get("authority") or "reference"),
        metadata=metadata,
        channels=[channel],
        updated_at=value.get("updated_at") or metadata.get("updated_at"),
    )


def _rrf_score(rank: int, channels: int, *, k: int = 60) -> float:
    return (1.0 / (k + rank)) * (1.0 + 0.15 * max(0, channels - 1))


#: Authority weighting is intentionally a small nudge. It must never let a stale
#: official page outrank a current one, which is what freshness weighting is for.
AUTHORITY_WEIGHT = {"official": 1.15, "community": 0.9, "reference": 1.0}

#: Freshness is a bounded multiplier: at most +/-10 %, so it reorders near-ties
#: (the same fact published twice) without overturning a much better textual match.
FRESHNESS_MIN_WEIGHT = 0.9
FRESHNESS_MAX_WEIGHT = 1.1
#: Half-life of the freshness signal, in days.
FRESHNESS_HALF_LIFE_DAYS = 365.0

_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def parse_hit_date(value: Any) -> date | None:
    """Best-effort date extraction from the free-form metadata the loaders produce."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    match = _DATE_RE.search(str(value or ""))
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except (TypeError, ValueError):
        return None


def freshness_weight(hit: RagHit, *, today: date | None = None) -> float:
    """Exponential decay on ``updated_at``, clamped so it only breaks near-ties.

    Undated documents get a neutral 1.0: penalising a document for missing metadata
    would silently bury hand-curated content that has no publication date.
    """
    updated = parse_hit_date(hit.updated_at)
    if updated is None:
        return 1.0
    age_days = max(0, ((today or date.today()) - updated).days)
    decay = 0.5 ** (age_days / FRESHNESS_HALF_LIFE_DAYS)
    return FRESHNESS_MIN_WEIGHT + (FRESHNESS_MAX_WEIGHT - FRESHNESS_MIN_WEIGHT) * decay


def fuse_hits(
    channel_hits: dict[str, list[RagHit | dict[str, Any]]],
    *,
    top_k: int,
    today: date | None = None,
) -> list[RagHit]:
    """Fuse channels with reciprocal rank fusion and content deduplication.

    Two adjustments on top of plain RRF, both aimed at a knowledge base that repeats the
    same fact across many document ids:

    * **authority weighting** separates official pages from visitor commentary;
    * **freshness weighting** separates two official pages that disagree because one is
      out of date. Without it, a stale price page and its replacement score identically
      and the winner is decided by insertion order.
    """
    merged: dict[str, RagHit] = {}
    ranks: dict[str, list[int]] = {}
    for channel, values in channel_hits.items():
        for rank, raw in enumerate(values, start=1):
            hit = _as_hit(raw, channel)
            key = str(hit.document_id or hit.chunk_id or " ".join(hit.content.split()))
            existing = merged.get(key)
            if existing is None:
                merged[key] = hit
            else:
                existing.channels = sorted(set(existing.channels + hit.channels))
                if not existing.content and hit.content:
                    existing.content = hit.content
                if existing.authority == "reference" and hit.authority != "reference":
                    existing.authority = hit.authority
                if not existing.updated_at and hit.updated_at:
                    existing.updated_at = hit.updated_at
            ranks.setdefault(key, []).append(rank)
    scored: list[tuple[float, RagHit]] = []
    for key, hit in merged.items():
        base = sum(_rrf_score(rank, len(hit.channels)) for rank in ranks[key])
        weight = AUTHORITY_WEIGHT.get(hit.authority, 1.0) * freshness_weight(hit, today=today)
        scored.append((base * weight, hit))
    scored.sort(key=lambda item: item[0], reverse=True)
    for score, hit in scored:
        hit.score = round(score, 6)
    return [hit for _, hit in scored[:top_k]]


# --------------------------------------------------------------------------- conflicts
#: ``(fact, label, pattern)`` triples. Only facts where a wrong answer has real
#: consequences are checked: a visitor who reads a stale ticket price or opening time
#: has been misled in a way that an apology cannot fix.
#:
#: The label matters more than the number. 「门票 190 元、观光车票 90 元」 is one sentence
#: with two *different facts*, not a contradiction, so a rule that simply collects every
#: "N 元" in a chunk reports a conflict on almost every official page and destroys the
#: credibility of the signal. A conflict therefore requires the **same label** to carry
#: different values.
CONFLICT_PATTERNS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("ticket_price", "门票", re.compile(r"门票[^0-9]{0,10}?(\d{2,4})\s*元")),
    ("ticket_price", "观光车票", re.compile(r"观光车票[^0-9]{0,10}?(\d{2,4})\s*元")),
    ("opening_time", "开放时间", re.compile(r"(?:开放时间|开园时间|售票时间)[^0-9]{0,10}?(\d{1,2}:\d{2})")),
    ("elevation", "海拔", re.compile(r"海拔[^0-9]{0,6}(\d{3,5})\s*米")),
)

#: Labels that restate a total rather than a fact, and would otherwise look like a
#: third, conflicting price.
CONFLICT_LABELS = {
    "ticket_price": "门票价格",
    "opening_time": "开放时间",
    "elevation": "海拔",
}


def _labelled_values(content: str) -> dict[tuple[str, str], set[str]]:
    found: dict[tuple[str, str], set[str]] = {}
    for fact, label, pattern in CONFLICT_PATTERNS:
        matches = {match.group(1) for match in pattern.finditer(content or "")}
        if matches:
            found.setdefault((fact, label), set()).update(matches)
    return found


def detect_conflicts(hits: list[RagHit], *, limit: int = 5) -> list[dict[str, Any]]:
    """Report labelled facts that disagree across the retrieved evidence.

    Conflicts are only meaningful within one subject: two attractions legitimately have
    different elevations, so hits are grouped by ``source_id`` (the entity they
    describe) and compared only inside a group. A hit with no entity is compared against
    hits of the same ``source_type``, which still catches duplicated park-level pages.

    Each conflict carries a ``severity``:

    * ``conflict``  - the same labelled fact is given different values by **different
      documents**. This is the dangerous case: evidence genuinely disagrees and the
      model must not silently pick one.
    * ``ambiguous`` - one document lists several values for one label (a peak and an
      off-peak price, for example). The fact is conditional rather than contradictory,
      so it warns the model to state the condition instead of lowering confidence.
    """
    groups: dict[str, list[RagHit]] = {}
    for hit in hits:
        subject = str(hit.source_id or hit.source_type or hit.document_id or "")
        if not subject:
            continue
        groups.setdefault(subject, []).append(hit)

    conflicts: list[dict[str, Any]] = []
    for subject, group in groups.items():
        values: dict[tuple[str, str], set[str]] = {}
        owners: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for hit in group:
            for (fact, label), found in _labelled_values(hit.content).items():
                values.setdefault((fact, label), set()).update(found)
                for value in found:
                    owners.setdefault((fact, label, value), []).append(
                        {
                            "document_id": hit.document_id or hit.chunk_id,
                            "updated_at": hit.updated_at,
                            "authority": hit.authority,
                        }
                    )
        for (fact, label), distinct in sorted(values.items()):
            if len(distinct) < 2:
                continue
            document_ids = {
                owner["document_id"]
                for (owner_fact, owner_label, _), entries in owners.items()
                if (owner_fact, owner_label) == (fact, label)
                for owner in entries
            }
            severity = "conflict" if len(document_ids) > 1 else "ambiguous"
            conflicts.append(
                {
                    "subject": subject,
                    "fact": fact,
                    "label": label,
                    "severity": severity,
                    "values": sorted(distinct),
                    "sources": [
                        {"value": value, **entries[0]}
                        for (owner_fact, owner_label, value), entries in sorted(owners.items())
                        if (owner_fact, owner_label) == (fact, label)
                    ],
                    "advice": (
                        f"检索到多个不一致的{label}（{'、'.join(sorted(distinct))}），"
                        "必须以最新官方公告为准，并在回答中说明存在不同说法。"
                        if severity == "conflict"
                        else f"资料中的{label}存在多个取值（{'、'.join(sorted(distinct))}），"
                        "属于旺淡季等条件差异，回答时必须说明适用条件，不要只报其中一个。"
                    ),
                }
            )
            if len(conflicts) >= limit:
                return conflicts
    return conflicts


class PostgresRetriever:
    """Adapter around the project's proven PostgreSQL retrieval implementation."""

    def search(self, request: RagSearchRequest) -> dict[str, list[dict[str, Any]]]:
        from .retrieval import retrieve
        channels: dict[str, list[dict[str, Any]]] = {"dense": [], "sparse": [], "entity": [], "structured": []}
        queries = request.filters.get("planned_queries") if isinstance(request.filters, dict) else None
        query_texts = [str(item.get("text")) for item in queries if isinstance(item, dict) and item.get("text")] if isinstance(queries, list) else [request.query]
        for query in query_texts[:4]:
            items = retrieve(query, top_k=max(request.top_k, config.RETRIEVAL_TOP_K))
            for item in items:
                method = str(item.get("retrieval") or "")
                if method == "entity":
                    channels["entity"].append(item)
                elif method == "vector":
                    channels["dense"].append(item)
                elif method in {"fts", "like"}:
                    channels["sparse"].append(item)
                else:
                    channels["structured"].append(item)
        return channels


class MilvusRetriever:
    """Milvus Dense + native BM25 adapter.

    The collection is built with a Milvus ``BM25`` function.  That means the sparse
    channel is searched by Milvus itself (``data=[query_text]``), rather than by a
    second PostgreSQL full-text implementation.  PostgreSQL is still queried by the
    hybrid service for entity/structured evidence and remains the safe fallback.
    """

    def __init__(self, uri: str | None = None, collection: str | None = None) -> None:
        self.uri = uri or config.MILVUS_URI
        self.collection = collection or config.MILVUS_COLLECTION
        self._client: Any | None = None

    def _client_or_none(self) -> Any | None:
        if self._client is not None:
            return self._client
        try:
            from pymilvus import MilvusClient

            # The timeout matters: without it a Milvus that is down but whose TCP port is
            # open leaves the health probe and the first retrieval waiting on the client
            # default, which is far longer than the request budget.
            self._client = MilvusClient(
                uri=self.uri,
                token=config.MILVUS_TOKEN or None,
                timeout=config.MILVUS_TIMEOUT_SECONDS,
            )
            return self._client
        except Exception:
            return None

    def search(self, request: RagSearchRequest) -> dict[str, list[dict[str, Any]]]:
        client = self._client_or_none()
        if client is None:
            raise RuntimeError("Milvus client is unavailable")
        from .retrieval import embed_query

        channels: dict[str, list[dict[str, Any]]] = {"dense": [], "sparse": [], "entity": []}
        planned = request.filters.get("planned_queries") if isinstance(request.filters, dict) else None
        queries = [
            str(item.get("text"))
            for item in planned
            if isinstance(item, dict) and item.get("text")
        ] if isinstance(planned, list) else []
        queries = list(dict.fromkeys([request.query, *queries]))[: max(1, config.RAG_MAX_QUERY_VARIANTS)]
        output_fields = [
            "chunk_id", "content", "document_id", "source_type", "source_id",
            "authority", "park_id", "metadata",
        ]
        filter_expr = self._filter_expression(request.filters)
        search_errors: list[str] = []
        for query in queries:
            vector_text = embed_query(query)
            if vector_text:
                try:
                    vector = self._parse_vector(vector_text)
                    rows = client.search(
                        collection_name=self.collection,
                        data=[vector],
                        anns_field="vector",
                        limit=max(config.RAG_DENSE_TOP_K, request.top_k),
                        output_fields=output_fields,
                        filter=filter_expr,
                        timeout=config.MILVUS_TIMEOUT_SECONDS,
                    )
                    channels["dense"].extend(self._normalize_milvus_rows(rows, "dense"))
                except Exception as exc:
                    # Continue into BM25: either retrieval channel can still provide
                    # useful evidence on its own.
                    search_errors.append(f"dense: {exc}")

            try:
                sparse_rows = client.search(
                    collection_name=self.collection,
                    data=[query],
                    anns_field="sparse_vector",
                    limit=max(config.RAG_SPARSE_TOP_K, request.top_k),
                    output_fields=output_fields,
                    filter=filter_expr,
                    timeout=config.MILVUS_TIMEOUT_SECONDS,
                )
                channels["sparse"].extend(self._normalize_milvus_rows(sparse_rows, "sparse"))
            except Exception as exc:
                search_errors.append(f"BM25: {exc}")

        if not channels["dense"] and not channels["sparse"]:
            detail = "; ".join(search_errors[-4:]) or "no candidates"
            raise RuntimeError(f"Milvus Dense and BM25 retrieval unavailable: {detail}")
        return channels

    @staticmethod
    def _parse_vector(value: str | list[float]) -> list[float]:
        if isinstance(value, list):
            return [float(part) for part in value]
        parsed: Any
        try:
            parsed = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            parsed = ast.literal_eval(value)
        if not isinstance(parsed, (list, tuple)):
            raise ValueError("embedding response is not a vector")
        return [float(part) for part in parsed]

    @staticmethod
    def _normalize_milvus_rows(rows: Any, channel: str) -> list[dict[str, Any]]:
        """Convert MilvusClient rows and Hit-like objects to the RAG evidence shape."""
        if not rows:
            return []
        first = rows[0] if isinstance(rows, list) else rows
        candidates = first if isinstance(first, list) else rows
        normalized: list[dict[str, Any]] = []
        for row in candidates or []:
            if isinstance(row, dict):
                entity = row.get("entity") or row
                distance = row.get("distance", row.get("score", 0.0))
                row_id = row.get("id")
            else:
                entity = getattr(row, "entity", None) or {}
                distance = getattr(row, "distance", getattr(row, "score", 0.0))
                row_id = getattr(row, "id", None)
            if not isinstance(entity, dict):
                try:
                    entity = dict(entity)
                except (TypeError, ValueError):
                    entity = {}
            item = dict(entity)
            if row_id is not None and not item.get("chunk_id"):
                item["chunk_id"] = str(row_id)
            metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            item["metadata"] = metadata
            item["score"] = float(distance or 0.0)
            item["retrieval"] = channel
            normalized.append(item)
        return normalized

    @staticmethod
    def _filter_expression(filters: dict[str, Any]) -> str | None:
        parts = []
        for key in ("park_id", "authority", "source_type"):
            value = filters.get(key)
            if isinstance(value, str) and value:
                parts.append(f'{key} == "{value.replace(chr(34), "")}"')
        return " and ".join(parts) or None


class HybridRagService:
    def __init__(self, *, primary: Retriever, fallback: Retriever, backend: str) -> None:
        self.primary = primary
        self.fallback = fallback
        self.backend = backend if backend in {"postgres", "shadow", "milvus"} else "postgres"
        self._rerank_client: Any | None = None

    @staticmethod
    def _cacheable(query: str) -> bool:
        # Never cache questions whose answer depends on current operating state.
        return not any(token in query for token in ("今天", "现在", "当前", "实时", "明天", "余票", "限流"))

    def _cache_key(self, request: RagSearchRequest) -> str:
        filters = request.filters if isinstance(request.filters, dict) else {}
        payload = json.dumps(
            {"query": request.query.strip().lower(), "top_k": request.top_k, "filters": filters},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"rag:v3:{config.PARK_ID}:{self.backend}:{digest}"

    async def search(self, request: RagSearchRequest) -> dict[str, Any]:
        cache_key = self._cache_key(request)
        # Only cache the real Milvus adapter. Test/fallback retrievers may be
        # deliberately blocking or stateful and must always be exercised.
        cache_enabled = self.backend == "milvus" and isinstance(self.primary, MilvusRetriever)
        if cache_enabled and self._cacheable(request.query):
            cached = cache_get(cache_key)
            if isinstance(cached, dict) and isinstance(cached.get("items"), list):
                cached["cache_hit"] = True
                return cached
        if self.backend == "postgres":
            raw = await _maybe_await(_offload(self.fallback.search, request))
            result = await self._rerank_result(self._result(raw, backend="postgres"), request.query)
            if cache_enabled and self._cacheable(request.query):
                cache_set(cache_key, result)
            return result
        try:
            # Both retrievers are synchronous: ``embed_query`` performs a real HTTP call
            # and pymilvus blocks on the network. Calling either directly from this
            # coroutine would stall every concurrent agent task for the duration of the
            # embedding request plus the ANN search, so both run on a worker thread.
            #
            # They are also started *together*. The dense/ANN channel and the
            # lexical/entity channel share no state, so awaiting them in sequence simply
            # added their latencies; the hybrid backend must not be slower than the sum
            # of its parts for no reason.
            primary_task = asyncio.ensure_future(_maybe_await(_offload(self.primary.search, request)))
            lexical_task = asyncio.ensure_future(_maybe_await(_offload(self.fallback.search, request)))
            try:
                raw = await primary_task
                lexical = await lexical_task
            except BaseException:
                lexical_task.cancel()
                raise
            if isinstance(raw, dict):
                # PostgreSQL contributes lexical/entity/structured candidates even
                # when Milvus is primary; this is the hybrid part of the contract.
                channels = {key: list(raw.get(key) or []) for key in ("dense", "sparse", "entity", "structured")}
                if isinstance(lexical, dict):
                    for key in ("sparse", "entity", "structured"):
                        channels[key].extend(list(lexical.get(key) or []))
                else:
                    channels["sparse"] = list(lexical or [])
                result = self._result(channels, backend="milvus")
            else:
                result = self._result(raw, backend="milvus")
        except Exception:
            raw = await _maybe_await(_offload(self.fallback.search, request))
            result = self._result(raw, backend="postgres-fallback")
        if self.backend == "shadow":
            try:
                fallback_raw = await _maybe_await(_offload(self.fallback.search, request))
                if isinstance(fallback_raw, dict):
                    result["shadow_count"] = sum(
                        len(fallback_raw.get(key) or [])
                        for key in ("dense", "sparse", "entity", "structured")
                    )
                else:
                    result["shadow_count"] = len(fallback_raw or [])
            except Exception:
                result["shadow_count"] = 0
        result = await self._rerank_result(result, request.query)
        if cache_enabled and self._cacheable(request.query):
            cache_set(cache_key, result)
        return result

    @staticmethod
    def _lexical_score(query: str, content: str) -> float:
        q = set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{1,2}", query or ""))
        c = set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{1,2}", content or ""))
        return len(q & c) / max(1, len(q))

    async def _rerank_result(self, result: dict[str, Any], query: str) -> dict[str, Any]:
        """Cross-encoder rerank after hybrid fusion, with lexical/offline fallback."""
        items = [item for item in result.get("items") or [] if isinstance(item, dict)]
        if len(items) < 2:
            result["rerank"] = {"enabled": False, "provider": "none", "count": len(items)}
            return result
        scores: list[float] | None = None
        provider = "lexical"
        if config.RERANK_BASE_URL and config.RERANK_API_KEY:
            try:
                import httpx
                if self._rerank_client is None:
                    self._rerank_client = httpx.AsyncClient(
                        timeout=config.RERANK_TIMEOUT_SECONDS,
                        limits=httpx.Limits(max_keepalive_connections=8, max_connections=16),
                    )
                res = await self._rerank_client.post(
                    f"{config.RERANK_BASE_URL}/rerank",
                    headers={"Authorization": f"Bearer {config.RERANK_API_KEY}"},
                    json={"model": config.RERANK_MODEL, "query": query, "documents": [str(i.get("content") or "") for i in items], "top_n": len(items), "return_documents": False},
                )
                res.raise_for_status()
                data = res.json().get("results") or []
                scores = [0.0] * len(items)
                for row in data:
                    if isinstance(row, dict) and isinstance(row.get("index"), int):
                        scores[row["index"]] = float(row.get("relevance_score") or row.get("score") or 0.0)
                provider = "api"
            except Exception:
                scores = None
        if scores is None:
            scores = [self._lexical_score(query, str(item.get("content") or "")) for item in items]
        for item, score in zip(items, scores):
            item["rerank_score"] = round(float(score), 6)
        result["items"] = sorted(items, key=lambda item: float(item.get("rerank_score") or 0), reverse=True)
        result["rerank"] = {"enabled": True, "provider": provider, "model": config.RERANK_MODEL if provider == "api" else "lexical-overlap", "count": len(items)}
        return result

    @staticmethod
    def _result(raw: Any, *, backend: str) -> dict[str, Any]:
        if isinstance(raw, dict) and any(key in raw for key in ("dense", "sparse", "entity", "structured")):
            channels = {key: list(raw.get(key) or []) for key in ("dense", "sparse", "entity", "structured")}
            hits = fuse_hits(channels, top_k=config.RAG_FINAL_TOP_K)
        else:
            # The PostgreSQL path already returns de-duplicated, ordered evidence, so it
            # is not re-fused - but its hits still carry content and a subject, and
            # skipping conflict detection here would mean the safety check silently does
            # nothing in the only backend that is actually deployed today.
            hits = [
                _as_hit(item, "postgres")
                for item in list(raw or [])[: config.RAG_FINAL_TOP_K]
            ]
        items = [HybridRagService._item(hit) for hit in hits]
        conflicts = detect_conflicts(hits)
        # Confidence already has to account for disagreement: two official pages that
        # contradict each other are weaker evidence than one page that says nothing
        # else, and reporting high confidence there is how a hallucination gets
        # laundered through a citation. Conditional ambiguity only costs a little,
        # because the fact is still usable once its condition is stated.
        confidence = min(1.0, 0.35 + 0.12 * len(items))
        hard = sum(1 for item in conflicts if item.get("severity") == "conflict")
        soft = sum(1 for item in conflicts if item.get("severity") == "ambiguous")
        if hard or soft:
            confidence = round(max(0.2, confidence - 0.15 * hard - 0.05 * soft), 4)
        authority_hits = sum(1 for item in items if item.get("authority") == "official")
        evidence_score = min(1.0, 0.25 + 0.1 * len(items) + 0.08 * authority_hits)
        if hard:
            evidence_score = max(0.0, evidence_score - 0.25 * hard)
        grounding_status = "insufficient" if not items else ("conflicted" if hard else "grounded")
        return {
            "backend": backend,
            "items": items,
            "conflicts": conflicts,
            "confidence": confidence,
            "evidence_score": round(evidence_score, 4),
            "grounding_status": grounding_status,
            "abstention_required": not items or bool(hard),
        }

    @staticmethod
    def _item(hit: RagHit) -> dict[str, Any]:
        return {
            "document_id": hit.document_id or hit.chunk_id,
            "chunk_id": hit.chunk_id,
            "source_type": hit.source_type,
            "source_id": hit.source_id,
            "content": hit.content,
            "authority": hit.authority,
            "updated_at": hit.updated_at,
            "metadata": hit.metadata,
            "score": hit.score,
            "retrieval": "+".join(hit.channels),
        }


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


async def _offload(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Run one synchronous retriever call on a worker thread.

    Retrieval touches an HTTP embedding endpoint and a network vector store. Both are
    synchronous clients, so awaiting them on the event loop blocks every other agent in
    the request - and the embedding provider's timeout is 12 s.
    """
    if inspect.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    return await asyncio.to_thread(func, *args, **kwargs)


_service: HybridRagService | None = None


def get_rag_service() -> HybridRagService:
    global _service
    if _service is None:
        _service = HybridRagService(
            primary=MilvusRetriever(), fallback=PostgresRetriever(), backend=config.RAG_BACKEND
        )
    return _service


def set_rag_service(service: HybridRagService | None) -> None:
    global _service
    _service = service


__all__ = [
    "AUTHORITY_WEIGHT",
    "CONFLICT_PATTERNS",
    "HybridRagService",
    "MilvusRetriever",
    "PostgresRetriever",
    "RagHit",
    "RagSearchRequest",
    "detect_conflicts",
    "freshness_weight",
    "fuse_hits",
    "get_rag_service",
    "set_rag_service",
]
