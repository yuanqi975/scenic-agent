"""Focused load scenarios for browse, recommendation, cache, model and SSE."""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from locust import HttpUser, between, tag, task, events

_observed = defaultdict(lambda: {"http": Counter(), "business": Counter(), "success_ms": [], "sse_complete": 0, "cache_hits": 0, "agent_runs": 0})


def _check(response, *, streaming=False):
    """Validate business status as well as HTTP; count full SSE read time."""
    name = response.request_meta["name"]
    observed = _observed[name]
    observed["http"][str(response.status_code)] += 1
    if response.status_code != 200:
        response.failure(f"HTTP {response.status_code}")
        return
    try:
        if streaming:
            frames = response.content.decode("utf-8").replace("\r\n", "\n").split("\n\n")
            event_data = {}
            for frame in frames:
                lines = frame.splitlines()
                event = next((line[7:] for line in lines if line.startswith("event: ")), None)
                data = "\n".join(line[6:] for line in lines if line.startswith("data: "))
                if event:
                    event_data[event] = json.loads(data)
            if "citations" not in event_data or "result" not in event_data:
                response.failure("SSE incomplete")
                return
            observed["sse_complete"] += 1
            payload = event_data["result"]
        else:
            payload = response.json()
        status = payload.get("status", "success")
        observed["business"][status] += 1
        if status not in {"success"} or payload.get("degraded"):
            response.failure(f"business {status}; degraded={bool(payload.get('degraded'))}")
            return
        if name.startswith("POST chat") and not payload.get("message"):
            response.failure("empty answer")
            return
        if name == "POST chat agent" and not payload.get("agent_chain"):
            response.failure("model scenario used a direct answer")
            return
        observed["cache_hits"] += int(bool(payload.get("cache_hit")))
        observed["agent_runs"] += int(bool(payload.get("agent_chain")))
        observed["success_ms"].append(response.request_meta["response_time"])
    except (ValueError, KeyError, TypeError) as exc:
        response.failure(f"invalid payload: {type(exc).__name__}")


@events.quitting.add_listener
def final_report(environment, **kwargs):
    """Persist shutdown totals, which can differ from the last periodic CSV flush."""
    prefix = environment.parsed_options.csv_prefix
    if not prefix:
        return
    rows = []
    for entry in [*environment.stats.entries.values(), environment.stats.total]:
        rows.append({"name": entry.name, "requests": entry.num_requests, "failures": entry.num_failures, "rps": entry.total_rps, "average_ms": entry.avg_response_time, "p95_ms": entry.get_response_time_percentile(.95), "p99_ms": entry.get_response_time_percentile(.99)})
    observed = {}
    for name, data in _observed.items():
        times = sorted(data["success_ms"])
        observed[name] = {**data, "success_ms": {"count": len(times), "average": sum(times) / len(times) if times else None, "p95": times[min(len(times)-1, int((len(times)-1)*.95))] if times else None}}
    Path(prefix + "_final.json").write_text(json.dumps({"stats": rows, "observed": observed}, ensure_ascii=False, indent=2), encoding="utf-8")


class ScenicVisitor(HttpUser):
    wait_time = between(0.5, 1.5)

    @tag("browse")
    @task(4)
    def browse_attractions(self):
        self.client.get("/api/v1/attractions?limit=20", name="GET attractions")

    @tag("recommend")
    @task(2)
    def route_recommendation(self):
        with self.client.post("/api/v1/recommendations", json={"duration_minutes": 240, "groups": ["老年游客"], "preferences": ["湖泊/海子"], "weather": "晴"}, name="POST recommendations", catch_response=True, timeout=65) as response:
            _check(response)

    @tag("cache")
    @task(3)
    def cached_question(self):
        with self.client.post("/api/v1/chat/messages", json={"message": "九寨沟门票多少钱"}, name="POST chat cached", catch_response=True, timeout=65) as response:
            _check(response)

    @tag("model")
    @task(1)
    def model_question(self):
        # Historical scenario: structured lookup, not guaranteed to call an LLM.
        with self.client.post("/api/v1/chat/messages", json={"message": "五花海开放时间和游玩建议"}, name="POST chat structured", catch_response=True, timeout=65) as response:
            _check(response)

    @tag("sse")
    @task(1)
    def streamed_question(self):
        started = time.perf_counter()
        with self.client.post("/api/v1/chat/stream", json={"message": "诺日朗瀑布怎么去"}, name="POST chat stream", stream=True, catch_response=True, timeout=65) as response:
            try:
                body = response.content
                response.request_meta["response_time"] = (time.perf_counter() - started) * 1000
                response.request_meta["response_length"] = len(body)
                _check(response, streaming=True)
            except Exception as exc:
                response.request_meta["response_time"] = (time.perf_counter() - started) * 1000
                response.failure(f"SSE read error: {type(exc).__name__}")


class ModelVisitor(HttpUser):
    """Run separately: Locust ... ModelVisitor; validate an actual agent chain."""
    wait_time = between(1, 2)

    @task
    def agent_question(self):
        with self.client.post("/api/v1/chat/messages", json={"message": "请比较树正沟、日则沟和则查洼沟的观景侧重点，并说明如何按体力取舍。"}, name="POST chat agent", catch_response=True, timeout=65) as response:
            _check(response)
