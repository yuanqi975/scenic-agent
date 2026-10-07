"""Typed failures shared by the tool layer, the agent runtime and the API layer.

Every error carries a ``code`` so the agent runtime can hand a *structured* failure
back to the model instead of raising a 500, and so the audit trail records why a
step was rejected.
"""

from __future__ import annotations

from typing import Any


class AgentError(Exception):
    """Base class for every recoverable multi-agent failure."""

    code = "agent_error"

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def as_payload(self) -> dict[str, Any]:
        return {"error": self.code, "message": self.message, **self.details}


class ToolNotFound(AgentError):
    code = "tool_not_found"


class ToolNotAllowed(AgentError):
    """The agent tried to call a tool outside its whitelist."""

    code = "tool_not_allowed"


class ToolArgumentError(AgentError):
    """Pydantic rejected the arguments; the model should retry with valid input."""

    code = "tool_argument_error"


class ToolExecutionError(AgentError):
    code = "tool_execution_error"


class ToolTimeout(AgentError):
    code = "tool_timeout"


class BudgetExceeded(AgentError):
    """Step, tool-call or wall-clock budget for the whole request is exhausted."""

    code = "budget_exceeded"


class RepeatedToolCall(AgentError):
    """Loop guard: the identical tool call was already observed."""

    code = "repeated_tool_call"


class DegradedMode(AgentError):
    """A capability is unavailable (no model, no embedding key, no realtime data)."""

    code = "degraded_mode"
