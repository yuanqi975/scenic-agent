"""Small Prometheus-compatible metrics endpoint.

The counters are intentionally process-local for now.  They provide a stable
contract for dashboards and can later be replaced by a Prometheus client without
changing API consumers.
"""

from __future__ import annotations

import time
from collections import Counter
from threading import Lock

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse

router = APIRouter(tags=["observability"])
_lock = Lock()
_counters: Counter[str] = Counter()
_latencies: list[float] = []


def record_request(path: str, status: int, elapsed_seconds: float) -> None:
    with _lock:
        _counters["http_requests_total"] += 1
        _counters[f'http_requests_total{{path="{path}",status="{status}"}}'] += 1
        _latencies.append(elapsed_seconds)
        del _latencies[:-1000]


@router.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    with _lock:
        lines = ["# TYPE http_requests_total counter"]
        lines.extend(f"{key} {value}" for key, value in _counters.items())
        if _latencies:
            values = sorted(_latencies)
            index = min(len(values) - 1, int(len(values) * 0.95))
            lines.append(f"http_request_duration_seconds_p95 {values[index]:.6f}")
        return "\n".join(lines) + "\n"


async def metrics_middleware(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    record_request(request.url.path, response.status_code, time.perf_counter() - started)
    return response
