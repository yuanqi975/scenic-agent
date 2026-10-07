"""The controlled tool layer.

Every capability an agent has must be a registered tool. Agents never touch SQL, HTTP
or the filesystem directly, so this module is where the security boundary lives:

1. unknown tool           -> :class:`ToolNotFound`
2. agent not whitelisted  -> :class:`ToolNotAllowed`
3. arguments invalid      -> :class:`ToolArgumentError` (handed back to the model)
4. tool raised / timed out -> structured error observation, never a crash
5. output is untrusted    -> fenced, truncated and scanned before re-entering context
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, ValidationError

from ..core.config import AGENT_TOOL_OUTPUT_MAX_CHARS
from ..core.errors import (
    AgentError,
    ToolArgumentError,
    ToolExecutionError,
    ToolNotAllowed,
    ToolNotFound,
    ToolTimeout,
)

#: Prompt-injection markers that must never reach another model turn verbatim.
INJECTION_PATTERNS: tuple[str, ...] = (
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(all\s+)?previous",
    r"system\s*prompt",
    r"系统提示词",
    r"忽略(以上|之前|前面)(的)?(所有)?(指令|提示|规则)",
    r"你现在是",
    r"请输出(你的)?(系统)?提示",
)

INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)


@dataclass(slots=True)
class ToolSpec:
    """One registered, permission-scoped capability."""

    name: str
    description: str
    args_schema: type[BaseModel]
    allowed: tuple[str, ...]
    func: Callable[..., Any]
    write: bool = False
    timeout: float | None = None
    tags: tuple[str, ...] = ()

    def schema(self) -> dict[str, Any]:
        """Strict JSON schema in the OpenAI function-calling shape."""
        parameters = self.args_schema.model_json_schema()
        parameters.pop("title", None)
        parameters.setdefault("type", "object")
        parameters.setdefault("additionalProperties", False)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }

    def allows(self, agent: str) -> bool:
        return agent in self.allowed


@dataclass(slots=True)
class ToolOutcome:
    """Normalised result of one tool execution."""

    tool: str
    status: str
    result: Any = None
    error: str | None = None
    error_code: str | None = None
    latency_ms: int = 0
    untrusted: bool = True
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def summary(self) -> dict[str, Any]:
        """Compact, JSON-safe form stored in the audit trail."""
        if self.ok:
            return _compact(self.result)
        return {"error": self.error_code or "error", "message": self.error}

    def for_model(self, limit: int = AGENT_TOOL_OUTPUT_MAX_CHARS) -> str:
        """Serialise for the next model turn, fenced as untrusted data."""
        body = json.dumps(self.summary(), ensure_ascii=False, default=str)
        return sanitize_tool_output(body, tool=self.tool, status=self.status, limit=limit)

    def as_observation(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "status": self.status,
            "result": self.summary() if self.ok else None,
            "error": self.error,
            "error_code": self.error_code,
            "latency_ms": self.latency_ms,
        }


def sanitize_tool_output(body: str, *, tool: str, status: str, limit: int = AGENT_TOOL_OUTPUT_MAX_CHARS) -> str:
    """Wrap tool output as untrusted data and neutralise injected instructions."""
    redacted = INJECTION_RE.sub("[已过滤的指令性内容]", body)
    if len(redacted) > limit:
        redacted = redacted[:limit] + "…（已截断）"
    return (
        f'<tool_output tool="{tool}" status="{status}" trusted="false">\n'
        f"{redacted}\n"
        "</tool_output>"
    )


def _compact(value: Any, *, depth: int = 0, max_items: int = 10) -> Any:
    """Shrink a tool result to something safe to store and to re-prompt with."""
    if depth > 4:
        return "…"
    if isinstance(value, dict):
        return {key: _compact(item, depth=depth + 1, max_items=max_items) for key, item in list(value.items())[:40]}
    if isinstance(value, (list, tuple)):
        return [_compact(item, depth=depth + 1, max_items=max_items) for item in list(value)[:max_items]]
    if isinstance(value, str):
        return value if len(value) <= 600 else value[:600] + "…"
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)[:200]


class ToolRegistry:
    """Holds every tool and enforces the permission matrix on execution."""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> ToolSpec:
        if spec.name in self._specs:
            raise ValueError(f"tool already registered: {spec.name}")
        self._specs[spec.name] = spec
        return spec

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def require(self, name: str, agent: str) -> ToolSpec:
        spec = self._specs.get(name)
        if spec is None:
            raise ToolNotFound(f"未注册的工具：{name}", tool=name, agent=agent)
        if not spec.allows(agent):
            raise ToolNotAllowed(
                f"{agent} 无权调用 {name}", tool=name, agent=agent, allowed=list(spec.allowed)
            )
        return spec

    def names(self) -> list[str]:
        return sorted(self._specs)

    def specs(self) -> list[ToolSpec]:
        return [self._specs[name] for name in self.names()]

    def for_agent(self, agent: str) -> list[ToolSpec]:
        return [spec for spec in self.specs() if spec.allows(agent)]

    def schemas_for(self, agent: str) -> list[dict[str, Any]]:
        return [spec.schema() for spec in self.for_agent(agent)]

    def write_tools(self) -> list[ToolSpec]:
        return [spec for spec in self.specs() if spec.write]

    async def execute(
        self,
        *,
        agent: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> ToolOutcome:
        """Validate, run and normalise one tool call. Never raises for tool faults."""
        started = time.perf_counter()

        def finish(status: str, result: Any = None, error: str | None = None, code: str | None = None, truncated: bool = False) -> ToolOutcome:
            return ToolOutcome(
                tool=tool_name,
                status=status,
                result=result,
                error=error,
                error_code=code,
                latency_ms=int((time.perf_counter() - started) * 1000),
                truncated=truncated,
            )

        try:
            spec = self.require(tool_name, agent)
        except AgentError as exc:
            return finish("rejected", error=exc.message, code=exc.code)

        raw_arguments = dict(arguments or {})
        raw_arguments.pop("__raw__", None)
        try:
            parsed = spec.args_schema.model_validate(raw_arguments)
        except ValidationError as exc:
            detail = "; ".join(
                f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}" for error in exc.errors()[:5]
            )
            return finish("invalid_arguments", error=detail, code=ToolArgumentError.code)

        call_timeout = timeout if timeout is not None else spec.timeout
        try:
            if inspect.iscoroutinefunction(spec.func):
                outcome = await asyncio.wait_for(spec.func(parsed), timeout=call_timeout)
            else:
                # Blocking database and HTTP work must not hold the event loop.
                outcome = await asyncio.wait_for(asyncio.to_thread(spec.func, parsed), timeout=call_timeout)
        except asyncio.TimeoutError:
            return finish("timeout", error=f"{tool_name} 执行超时", code=ToolTimeout.code)
        except AgentError as exc:
            return finish("error", error=exc.message, code=exc.code)
        except (ValidationError, ValueError) as exc:
            return finish("invalid_arguments", error=str(exc)[:300], code=ToolArgumentError.code)
        except Exception as exc:  # a tool bug must not take down the request
            return finish("error", error=f"{type(exc).__name__}: {exc}"[:300], code=ToolExecutionError.code)

        return finish("ok", result=outcome)


TOOL_REGISTRY = ToolRegistry()


def tool(
    *,
    name: str,
    description: str,
    args_schema: type[BaseModel],
    allowed: list[str] | tuple[str, ...],
    write: bool = False,
    timeout: float | None = None,
    tags: tuple[str, ...] = (),
    registry: ToolRegistry | None = None,
):
    """Register one tool. ``allowed`` is the security boundary, not documentation."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        (registry or TOOL_REGISTRY).register(
            ToolSpec(
                name=name,
                description=description,
                args_schema=args_schema,
                allowed=tuple(allowed),
                func=func,
                write=write,
                timeout=timeout,
                tags=tags,
            )
        )
        return func

    return decorator


def tools_for_agent(agent: str) -> list[ToolSpec]:
    return TOOL_REGISTRY.for_agent(agent)


def tool_schemas(agent: str) -> list[dict[str, Any]]:
    return TOOL_REGISTRY.schemas_for(agent)


def tool_names() -> list[str]:
    return TOOL_REGISTRY.names()


def permission_matrix() -> dict[str, list[str]]:
    """name -> agents allowed; powers the admin Agent detail page."""
    from ..services.prompts import agent_system_prompts

    agents = set(agent_system_prompts())
    matrix: dict[str, list[str]] = {}
    for spec in TOOL_REGISTRY.specs():
        matrix[spec.name] = list(spec.allowed)
    for agent in agents:
        matrix.setdefault(agent, [])
    return matrix


__all__ = [
    "TOOL_REGISTRY",
    "ToolOutcome",
    "ToolRegistry",
    "ToolSpec",
    "permission_matrix",
    "sanitize_tool_output",
    "tool",
    "tool_names",
    "tool_schemas",
    "tools_for_agent",
]
