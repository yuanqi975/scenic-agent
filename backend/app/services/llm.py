"""OpenAI-compatible chat client.

Only ``httpx`` plus the standard library: the existing deployment already exposes a
``/chat/completions`` endpoint, so the multi-agent runtime can speak the documented
function-calling protocol without pulling in an agent framework.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..core.config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MAX_CONCURRENCY,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
)


@dataclass(slots=True)
class ToolCall:
    """One function call requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Any) -> "ToolCall":
        """Parse one tool-call object, tolerating every malformed shape.

        A model (or a proxy) can return ``function`` as a string, ``arguments`` as a
        list, or the whole entry as something that is not a mapping at all. Treating a
        malformed call as an empty one is safe: the registry rejects it on validation
        instead of the runtime raising ``AttributeError`` out of the ReAct loop.
        """
        if not isinstance(payload, dict):
            return cls(id="call_malformed", name="", arguments={})
        function = payload.get("function")
        if not isinstance(function, dict):
            function = {}
        raw_arguments = function.get("arguments")
        if isinstance(raw_arguments, str):
            try:
                parsed = json.loads(raw_arguments) if raw_arguments.strip() else {}
            except json.JSONDecodeError:
                parsed = {"__raw__": raw_arguments}
            if not isinstance(parsed, dict):
                parsed = {"__raw__": raw_arguments}
        elif isinstance(raw_arguments, dict):
            parsed = raw_arguments
        else:
            parsed = {}
        return cls(
            id=str(payload.get("id") or f"call_{function.get('name') or 'unknown'}"),
            name=str(function.get("name") or ""),
            arguments=parsed,
        )


@dataclass(slots=True)
class ModelReply:
    """A normalised model turn: free text plus zero or more tool calls."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    reasoning_content: str = ""

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLMUnavailable(RuntimeError):
    """Raised when the model endpoint is missing, unreachable or answers garbage."""


class LLMClient:
    """Thin, testable wrapper around one OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        base_url: str = LLM_BASE_URL,
        api_key: str = LLM_API_KEY,
        model: str = LLM_MODEL,
        timeout: float = LLM_TIMEOUT_SECONDS,
        temperature: float = LLM_TEMPERATURE,
        max_concurrency: int = LLM_MAX_CONCURRENCY,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self._semaphore = asyncio.Semaphore(max(1, max_concurrency))
        self._transport = transport
        # Reuse one client per long-lived LLMClient. Creating a new AsyncClient for
        # every model turn forced a fresh TCP/TLS path even though one visitor request
        # commonly makes several turns (router, specialist and reviewer).
        self._http_client: httpx.AsyncClient | None = None
        self.call_count = 0

    def _client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=self.timeout,
                transport=self._transport,
                limits=httpx.Limits(max_keepalive_connections=16, max_connections=32),
            )
        return self._http_client

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.api_key)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ModelReply:
        """One model turn. Raises :class:`LLMUnavailable` instead of returning junk."""
        if not self.configured:
            raise LLMUnavailable("LLM_BASE_URL/LLM_API_KEY 未配置")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if response_format:
            payload["response_format"] = response_format

        async with self._semaphore:
            self.call_count += 1
            try:
                response = await self._client().post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                )
                response.raise_for_status()
                body = response.json()
            except httpx.HTTPStatusError as exc:
                detail = (exc.response.text or "")[:800]
                raise LLMUnavailable(f"{exc}; provider={detail}") from exc
            except (httpx.HTTPError, ValueError) as exc:  # network, status or JSON failure
                raise LLMUnavailable(str(exc)) from exc

        try:
            choice = body["choices"][0]
            if not isinstance(choice, dict):
                raise TypeError("choice is not an object")
            message = choice.get("message")
            if not isinstance(message, dict):
                message = {}
            if not isinstance(body.get("usage") or {}, dict):
                raise TypeError("usage is not an object")
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMUnavailable("模型返回结构不符合 OpenAI 兼容格式") from exc

        raw_calls = message.get("tool_calls")
        if not isinstance(raw_calls, list):
            raw_calls = []
        content = message.get("content")
        return ModelReply(
            content=(content if isinstance(content, str) else "").strip(),
            tool_calls=[ToolCall.from_payload(call) for call in raw_calls],
            finish_reason=choice.get("finish_reason"),
            usage=body.get("usage") or {},
            reasoning_content=(message.get("reasoning_content") if isinstance(message.get("reasoning_content"), str) else ""),
        )

    async def complete_json(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        retries: int = 1,
    ) -> dict[str, Any]:
        """Ask for a JSON object and tolerate one malformed answer."""
        attempt = 0
        last_error: Exception | None = None
        working = list(messages)
        while attempt <= retries:
            reply = await self.complete(
                working,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            parsed = _loads_object(reply.content)
            if parsed is not None:
                return parsed
            last_error = ValueError("模型未返回可解析的 JSON")
            working = working + [
                {"role": "assistant", "content": reply.content},
                {"role": "user", "content": "上一次输出不是合法 JSON 对象，请只输出 JSON。"},
            ]
            attempt += 1
        raise LLMUnavailable(str(last_error) if last_error else "JSON 解析失败")


def _loads_object(raw: str) -> dict[str, Any] | None:
    """Parse a JSON object, tolerating ```json fences and surrounding prose."""
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text[3:]
        text = text.removeprefix("json").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


_default_client: LLMClient | None = None


def get_client() -> LLMClient:
    """Process-wide default client (kept lazy so tests can inject their own)."""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client


def set_client(client: LLMClient | None) -> None:
    """Test helper: replace the process-wide client."""
    global _default_client
    _default_client = client
