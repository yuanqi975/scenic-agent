"""Controlled tool layer.

Importing this package registers every tool exactly once. Nothing else in the
codebase may reach a datasource directly on behalf of an agent.
"""

from . import attractions, conversation, facilities, feedback, knowledge, realtime, route  # noqa: F401
from .registry import (  # noqa: F401
    TOOL_REGISTRY,
    ToolOutcome,
    ToolRegistry,
    ToolSpec,
    permission_matrix,
    sanitize_tool_output,
    tool,
    tool_names,
    tool_schemas,
    tools_for_agent,
)

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
