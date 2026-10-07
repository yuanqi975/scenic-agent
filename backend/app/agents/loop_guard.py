"""Budget and loop guards.

Every failure mode the multi-agent design has to survive is enforced here, in one
place, instead of being spread across prompts:

* per-agent step ceiling
* whole-request tool-call ceiling
* whole-request wall-clock deadline
* identical-call detection (prevents the classic think/observe ping-pong)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from ..core.config import (
    AGENT_MAX_STEPS,
    AGENT_MAX_TOTAL_TOOL_CALLS,
    AGENT_REQUEST_TIMEOUT_SECONDS,
)
from ..core.errors import BudgetExceeded, RepeatedToolCall


@dataclass
class Budget:
    """Shared, request-scoped budget. One instance per visitor request."""

    max_steps: int = AGENT_MAX_STEPS
    max_tool_calls: int = AGENT_MAX_TOTAL_TOOL_CALLS
    max_seconds: float = AGENT_REQUEST_TIMEOUT_SECONDS
    tool_calls: int = 0
    steps: int = 0
    started_at: float = field(default_factory=time.monotonic)

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def remaining_seconds(self) -> float:
        return self.max_seconds - self.elapsed_seconds

    @property
    def exhausted_by_time(self) -> bool:
        return self.remaining_seconds <= 0

    def charge_step(self) -> int:
        if self.exhausted_by_time:
            raise BudgetExceeded(
                f"整次请求已超过 {self.max_seconds:.0f} 秒上限",
                elapsed_seconds=round(self.elapsed_seconds, 2),
            )
        if self.steps >= self.max_steps * 4:  # 4 agents x max_steps safety net
            raise BudgetExceeded("整次请求步骤数超限", steps=self.steps)
        self.steps += 1
        return self.steps

    def charge_tool_call(self) -> int:
        if self.exhausted_by_time:
            raise BudgetExceeded("整次请求超时，已停止继续调用工具")
        if self.tool_calls + 1 > self.max_tool_calls:
            raise BudgetExceeded(
                f"整次请求工具调用超过 {self.max_tool_calls} 次上限", tool_calls=self.tool_calls
            )
        self.tool_calls += 1
        return self.tool_calls


class LoopGuard:
    """Detects a model that keeps issuing the same call instead of making progress."""

    def __init__(self, *, max_repeats: int = 1) -> None:
        self.max_repeats = max_repeats
        self._seen: dict[str, int] = {}

    @staticmethod
    def fingerprint(tool_name: str, arguments: dict | None) -> str:
        return f"{tool_name}:{json.dumps(arguments or {}, sort_keys=True, ensure_ascii=False, default=str)}"

    def check(self, tool_name: str, arguments: dict | None) -> None:
        key = self.fingerprint(tool_name, arguments)
        self._seen[key] = self._seen.get(key, 0) + 1
        if self._seen[key] > self.max_repeats:
            raise RepeatedToolCall(
                f"{tool_name} 已用相同参数调用过，请改用其它工具或直接给出结论",
                tool=tool_name,
                repeat=self._seen[key],
            )

    def notes(self) -> dict[str, int]:
        return dict(self._seen)
