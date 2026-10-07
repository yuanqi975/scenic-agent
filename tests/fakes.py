"""In-memory fakes: the whole multi-agent pipeline can be verified without PostgreSQL.

The production code deliberately reads its catalog through injectable callables, so a
test can supply the Jiuzhaigou records directly. A scripted LLM client drives the ReAct
loop deterministically, which is what makes "the model decided to call a tool twice"
assertable.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

import httpx

# --------------------------------------------------------------------------- catalog
ATTRACTIONS: list[dict[str, Any]] = [
    {
        "attraction_id": "attr_001",
        "name": "五花海",
        "valley": "日则沟",
        "category": "湖泊/海子",
        "description": "五花海以水色斑斓著称。",
        "visit_duration_minutes": 40,
        "difficulty": "轻松",
        "suitable_for": ["普通游客", "摄影爱好者", "亲子家庭"],
        "opening_hours": "旺季 08:00-18:00",
        "ticket_note": "含于景区门票（旺季 190 元 / 淡季 80 元）",
        "status": "open",
        "in_standard_tour": True,
        "seasonal_closure": None,
        "elevation_meters": 2472,
    },
    {
        "attraction_id": "attr_014",
        "name": "诺日朗瀑布",
        "valley": "日则沟",
        "category": "瀑布",
        "description": "诺日朗瀑布是景区内最宽的瀑布。",
        "visit_duration_minutes": 40,
        "difficulty": "轻松",
        "suitable_for": ["普通游客", "摄影爱好者"],
        "opening_hours": "旺季 08:00-18:00",
        "ticket_note": "含于景区门票（旺季 190 元 / 淡季 80 元）",
        "status": "open",
        "in_standard_tour": True,
        "seasonal_closure": None,
        "elevation_meters": 2365,
    },
    {
        "attraction_id": "attr_028",
        "name": "原始森林",
        "valley": "日则沟",
        "category": "森林",
        "description": "原始森林海拔较高。",
        "visit_duration_minutes": 35,
        "difficulty": "中等",
        "suitable_for": ["户外爱好者"],
        "opening_hours": "旺季 08:00-18:00",
        "ticket_note": "含于景区门票（旺季 190 元 / 淡季 80 元）",
        "status": "open",
        "in_standard_tour": True,
        "seasonal_closure": {"range": "11-16 至次年 03-31", "reason": "季节性轮休保育"},
        "elevation_meters": 3060,
    },
    {
        "attraction_id": "attr_039",
        "name": "扎依扎嘎神山",
        "valley": "扎如沟",
        "category": "山峰/神山",
        "description": "扎如沟内的神山。",
        "visit_duration_minutes": 60,
        "difficulty": "挑战",
        "suitable_for": ["户外爱好者"],
        "opening_hours": "旺季 08:00-18:00",
        "ticket_note": "含于景区门票（旺季 190 元 / 淡季 80 元）",
        "status": "open",
        "in_standard_tour": False,
        "seasonal_closure": None,
        "elevation_meters": None,
    },
]

FACILITIES: list[dict[str, Any]] = [
    {
        "facility_id": "facility_010",
        "name": "诺日朗中心站",
        "facility_type": "观光车站",
        "status": "available",
        "inside_park": True,
        "nearby_attraction_id": "attr_014",
        "phone": None,
        "accessibility": True,
        "distance_to_gate_meters": None,
    },
    {
        "facility_id": "facility_006",
        "name": "诺日朗服务中心（诺日朗餐厅）",
        "facility_type": "餐饮点",
        "status": "available",
        "inside_park": True,
        "nearby_attraction_id": "attr_014",
        "phone": "0837-7739753",
        "accessibility": True,
        "distance_to_gate_meters": None,
    },
]

FAQS: list[dict[str, Any]] = [
    {
        "faq_id": "faq_001",
        "topic": "门票",
        "question": "九寨沟门票多少钱？",
        "answer": "旺季 190 元，淡季 80 元，观光车票 90 元。",
        "keywords": ["门票", "190", "80"],
        "official": True,
        "related_attraction_id": None,
    }
]

DOCUMENTS: list[dict[str, Any]] = [
    {
        "document_id": "doc_00001",
        "source_type": "park",
        "source_id": "jiuzhaigou_scenic_area",
        "content": "九寨沟风景名胜区旺季门票 190 元、淡季 80 元，观光车票 90 元。",
        "metadata": {"topic": "门票与开放", "version": "v1.0", "source_type": "park"},
    },
    {
        "document_id": "doc_00021",
        "source_type": "attraction",
        "source_id": "attr_001",
        "content": "【五花海】五花海位于日则沟，海拔约 2472 米，建议游览 40 分钟。",
        "metadata": {"topic": "五花海", "version": "v1.0", "source_type": "attraction"},
    },
    {
        "document_id": "doc_00022",
        "source_type": "attraction",
        "source_id": "attr_001",
        "content": "游览提示：五花海（日则沟 · 湖泊/海子）。请沿栈道行走，勿进入未开放区域。",
        "metadata": {"topic": "五花海", "version": "v1.0", "source_type": "attraction"},
    },
    {
        "document_id": "doc_00041",
        "source_type": "attraction",
        "source_id": "attr_014",
        "content": "【诺日朗瀑布】诺日朗瀑布位于日则沟，海拔约 2365 米，建议游览 40 分钟。",
        "metadata": {"topic": "诺日朗瀑布", "version": "v1.0", "source_type": "attraction"},
    },
    {
        "document_id": "doc_00250",
        "source_type": "faq",
        "source_id": "faq_001",
        "content": "问：九寨沟门票多少钱？ 答：旺季 190 元，淡季 80 元，观光车票 90 元。",
        "metadata": {"topic": "门票", "version": "v1.0", "source_type": "faq"},
    },
]

TABLES: dict[str, list[dict[str, Any]]] = {
    "attractions": ATTRACTIONS,
    "facilities": FACILITIES,
    "faqs": FAQS,
}


def fake_payload_rows(table: str, limit: int = 100, offset: int = 0, park_id: str | None = None):
    """Drop-in replacement for ``core.db.payload_rows``."""
    rows = TABLES.get(table, [])
    return [dict(item) for item in rows[offset : offset + limit]]


class FakeConnection:
    """Minimal connection double: enough for the retrieval SQL paths.

    It supports the context-manager protocol because production reads ``with
    engine.connect()``. Without ``__enter__`` every query raised ``TypeError`` into a
    bare ``except Exception: return []`` - so the whole retrieval layer silently
    returned nothing and any assertion of the form "no results came back" passed for the
    wrong reason.
    """

    def __init__(self, documents: list[dict[str, Any]] | None = None) -> None:
        self.documents = documents if documents is not None else DOCUMENTS

    def __enter__(self) -> "FakeConnection":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
        return False

    def execute(self, statement, params: dict[str, Any] | None = None):  # noqa: ANN001
        sql = " ".join(str(statement).split()).lower()
        params = params or {}
        if "from documents" in sql and "source_id=:source_id" in sql:
            source_id = params.get("source_id")
            rows = [item for item in self.documents if item["source_id"] == source_id]
        elif "from documents" in sql and "to_tsvector" in sql:
            message = str(params.get("message") or "")
            rows = [item for item in self.documents if message and message in item["content"]]
        elif "from documents" in sql and "ilike" in sql:
            needle = str(params.get("q") or "").strip("%")
            rows = [item for item in self.documents if needle and needle in item["content"]]
        elif "from document_chunks" in sql:
            rows = []
        elif "from conversation_messages" in sql:
            rows = self._conversation_rows(params)
        else:
            rows = []
        return _FakeResult(rows)

    def _conversation_rows(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Conversation history is per-test state, so it starts empty by design."""
        return []


class _FakeResult:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def fetchall(self) -> list[tuple]:
        return [
            (
                item.get("document_id"),
                item.get("source_type"),
                item.get("source_id"),
                item.get("content"),
                item.get("metadata"),
            )
            for item in self._rows
        ]

    def first(self):
        rows = self.fetchall()
        return rows[0] if rows else None

    def scalar(self):  # noqa: ANN201
        rows = self.fetchall()
        return rows[0][0] if rows and rows[0] else None


def install_fake_database(monkeypatch, *, tables: dict[str, list[dict[str, Any]]] | None = None) -> None:
    """Route every read path to the in-memory catalog for the duration of a test."""
    from app.core import db
    from app.services import conversations, retrieval
    from app.tools import attractions, facilities, knowledge, realtime, route

    catalog = tables or TABLES

    def rows(table: str, limit: int = 100, offset: int = 0, park_id: str | None = None):
        items = catalog.get(table, [])
        return [dict(item) for item in items[offset : offset + limit]]

    for module in (db, retrieval, attractions, facilities, knowledge, realtime, route):
        if hasattr(module, "payload_rows"):
            monkeypatch.setattr(module, "payload_rows", rows, raising=False)

    # ``engine`` must be replaced everywhere a *read* still opens a connection inline.
    # Missing one module here is not a harmless omission: the engine stays real, the
    # ``enabled()`` patch below reports the database as up, and the test then blocks on
    # a genuine connect attempt for the full connect timeout.
    for module in (retrieval, realtime, conversations):
        monkeypatch.setattr(module, "engine", _FakeEngine(), raising=False)
    monkeypatch.setattr(db, "database_available", lambda: True, raising=False)
    monkeypatch.setattr(db, "enabled", lambda: True, raising=False)
    monkeypatch.setattr(db, "db_ready", lambda: True, raising=False)


class FakeEngine:
    """Engine double whose connections understand the retrieval SQL paths."""

    def connect(self) -> FakeConnection:
        return FakeConnection()

    def begin(self) -> FakeConnection:
        return FakeConnection()


#: Backwards-compatible alias.
_FakeEngine = FakeEngine


# --------------------------------------------------------------------------- llm
class FakeLLMClient:
    """Scripted OpenAI-compatible client.

    ``script`` is consumed one entry per ``complete()`` call; each entry is either a
    list of ``(tool_name, arguments)`` pairs (a tool-calling turn) or a string (a final
    answer). ``complete_json`` pops from ``json_script``.
    """

    def __init__(
        self,
        script: Iterable[Any] | None = None,
        *,
        json_script: Iterable[dict[str, Any]] | None = None,
        fail_after: int | None = None,
    ) -> None:
        self.script = list(script or [])
        self.json_script = list(json_script or [])
        self.fail_after = fail_after
        self.calls: list[dict[str, Any]] = []
        self.mode = "agent"

    @property
    def configured(self) -> bool:
        return True

    async def complete(self, messages, *, tools=None, temperature=None, response_format=None):  # noqa: ANN001
        from app.services.llm import ModelReply, ToolCall

        self.calls.append({"messages": messages, "tools": tools})
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            from app.services.llm import LLMUnavailable

            raise LLMUnavailable("scripted model failure")
        step = self.script.pop(0) if self.script else "（脚本已用尽）"
        if isinstance(step, str):
            return ModelReply(content=step, finish_reason="stop")
        calls = [
            ToolCall(id=f"call_{len(self.calls)}_{index}", name=name, arguments=arguments)
            for index, (name, arguments) in enumerate(step)
        ]
        return ModelReply(content="需要调用工具", tool_calls=calls, finish_reason="tool_calls")

    async def complete_json(self, messages, *, temperature=None, retries=1):  # noqa: ANN001
        self.calls.append({"messages": messages, "json": True})
        if self.json_script:
            return self.json_script.pop(0)
        return {"ok": True, "unsupported": [], "fixed": ""}


class ScriptedLLMClient(FakeLLMClient):
    """Alias kept for readability in tests that only script ``complete_json``."""


def fake_transport(payloads: list[dict[str, Any]]) -> httpx.MockTransport:
    """A real HTTP transport, so :class:`LLMClient` is exercised end to end."""

    queue = list(payloads)

    def handler(request: httpx.Request) -> httpx.Response:
        body = queue.pop(0) if queue else {"choices": [{"message": {"content": "（无更多脚本）"}}]}
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler)


def openai_tool_call(name: str, arguments: dict[str, Any], call_id: str = "call_1") -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }


def openai_text(content: str) -> dict[str, Any]:
    return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
