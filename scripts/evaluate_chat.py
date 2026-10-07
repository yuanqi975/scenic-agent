"""Evaluate the visitor-facing chat API on the complex Jiuzhaigou set.

This is intentionally answer-level evaluation: retrieval can find a document while
the final response still omits a requested constraint or invents a live status.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

import httpx


NO_EVIDENCE = ("暂无相关信息", "没有收录", "无法回答", "没有找到", "请重新发")


def _norm(value: Any) -> str:
    # Ignore punctuation/spacing differences (e.g. "树正沟、日则沟" versus a
    # compact checklist in the expected facts).
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", str(value or "")).lower()


def _contains(answer: str, value: Any) -> bool:
    needle = _norm(value)
    haystack = _norm(answer)
    if not needle:
        return False
    if needle in haystack:
        return True
    # Chinese facts are often interrupted by qualifiers ("四川省阿坝藏族羌族
    # 自治州九寨沟县" vs. the compact label "四川阿坝九寨沟县").  Accept a
    # high overlap of the fact's character bigrams while keeping short labels exact.
    if len(needle) >= 5 and all(needle[index].strip() for index in range(len(needle))):
        bigrams = {needle[index : index + 2] for index in range(len(needle) - 1)}
        hit_rate = sum(pair in haystack for pair in bigrams) / max(1, len(bigrams))
        return hit_rate >= 0.65
    return False


def score_answer(answer: str, item: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    required = [str(value) for value in item.get("required_facts") or [] if str(value).strip()]
    entities = [str(value) for value in item.get("expected_route_entities") or [] if str(value).strip()]
    must = [str(value) for value in item.get("must_answer") or [] if str(value).strip()]
    forbidden = [str(value) for value in item.get("forbidden_claims") or [] if str(value).strip()]
    fact_hits = [value for value in required if _contains(answer, value)]
    entity_hits = [value for value in entities if _contains(answer, value)]
    must_hits = [value for value in must if _contains(answer, value)]
    forbidden_hits = [value for value in forbidden if _contains(answer, value)]
    fact_coverage = len(fact_hits) / max(1, len(required))
    entity_coverage = len(entity_hits) / max(1, len(entities))
    must_coverage = len(must_hits) / max(1, len(must))
    no_evidence = any(_contains(answer, phrase) for phrase in NO_EVIDENCE)
    citations = response.get("citations") or []
    grounded = bool(answer.strip()) and not no_evidence and not forbidden_hits
    # Required facts carry the most weight; must-answer labels are a softer signal
    # because a model can phrase "换乘" as "在诺日朗中心站转车".
    coverage = round(0.55 * fact_coverage + 0.25 * entity_coverage + 0.20 * must_coverage, 6)
    return {
        "question_id": item.get("question_id"),
        "intent": item.get("intent"),
        "answer": answer,
        "fact_coverage": round(fact_coverage, 6),
        "entity_coverage": round(entity_coverage, 6),
        "must_answer_coverage": round(must_coverage, 6),
        "coverage": coverage,
        "required_facts_hit": fact_hits,
        "required_facts_missing": [value for value in required if value not in fact_hits],
        "route_entities_hit": entity_hits,
        "route_entities_missing": [value for value in entities if value not in entity_hits],
        "forbidden_claims_hit": forbidden_hits,
        "citation_count": len(citations),
        "grounded": grounded,
        "complete": grounded and coverage >= 0.70 and (not required or fact_coverage >= 0.60),
        "status": response.get("status"),
        "degraded": response.get("degraded"),
        "trace_id": response.get("trace_id"),
    }


async def ask(client: httpx.AsyncClient, base_url: str, message: str, conversation_id: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"message": message, "mode": "agent"}
    if conversation_id:
        payload["conversation_id"] = conversation_id
    response = await client.post(f"{base_url.rstrip('/')}/api/v1/chat/messages", json=payload)
    response.raise_for_status()
    return response.json()


async def evaluate(dataset: list[dict[str, Any]], base_url: str, concurrency: int) -> dict[str, Any]:
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    timeout = httpx.Timeout(120.0, connect=10.0)
    semaphore = asyncio.Semaphore(concurrency)
    # Ignore workstation HTTP(S)_PROXY settings: the benchmark must exercise the
    # local Docker service directly, otherwise a corporate proxy can return 502.
    async with httpx.AsyncClient(limits=limits, timeout=timeout, trust_env=False) as client:
        async def one(item: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                try:
                    started = time.perf_counter()
                    # cx_021 explicitly tests anaphora across turns.
                    if item.get("question_id") == "cx_021":
                        first = await ask(client, base_url, "五花海在哪里？")
                        response = await ask(client, base_url, "它适合老人吗？", first.get("conversation_id"))
                    else:
                        response = await ask(client, base_url, str(item.get("question") or ""))
                    row = score_answer(str(response.get("message") or ""), item, response)
                    row["wall_latency_ms"] = int((time.perf_counter() - started) * 1000)
                    row["response_latency_ms"] = response.get("latency_ms")
                    row["timings"] = response.get("timings") or {}
                    return row
                except Exception as exc:  # keep the report complete if one request fails
                    return {
                        "question_id": item.get("question_id"),
                        "intent": item.get("intent"),
                        "error": f"{type(exc).__name__}: {exc}",
                        "coverage": 0.0,
                        "complete": False,
                        "grounded": False,
                    }

        rows = await asyncio.gather(*(one(item) for item in dataset))
    valid = [row for row in rows if "error" not in row]
    mean = lambda key: round(sum(float(row.get(key, 0.0)) for row in valid) / max(1, len(valid)), 6)
    latency_values = sorted(
        int(row.get("wall_latency_ms") or 0) for row in valid if row.get("wall_latency_ms") is not None
    )

    def percentile(percent: float) -> int:
        if not latency_values:
            return 0
        index = min(len(latency_values) - 1, round((len(latency_values) - 1) * percent))
        return latency_values[index]

    return {
        "questions": len(dataset),
        "successful_requests": len(valid),
        "failed_requests": len(rows) - len(valid),
        "complete_answers": sum(bool(row.get("complete")) for row in rows),
        "grounded_answers": sum(bool(row.get("grounded")) for row in rows),
        "answer_complete_rate": round(sum(bool(row.get("complete")) for row in rows) / max(1, len(rows)), 6),
        "grounded_rate": round(sum(bool(row.get("grounded")) for row in rows) / max(1, len(rows)), 6),
        "mean_fact_coverage": mean("fact_coverage"),
        "mean_entity_coverage": mean("entity_coverage"),
        "mean_must_answer_coverage": mean("must_answer_coverage"),
        "mean_coverage": mean("coverage"),
        "citation_rate": round(sum(bool(row.get("citation_count")) for row in valid) / max(1, len(valid)), 6),
        "forbidden_claim_answers": sum(bool(row.get("forbidden_claims_hit")) for row in rows),
        "latency_ms": {
            "mean": round(sum(latency_values) / max(1, len(latency_values)), 2),
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "max": max(latency_values, default=0),
        },
        "details": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--ids", nargs="*", default=[], help="optional question ids to evaluate")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.ids:
        wanted = set(args.ids)
        dataset = [item for item in dataset if str(item.get("question_id")) in wanted]
    report = asyncio.run(evaluate(dataset, args.base_url, max(1, args.concurrency)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
