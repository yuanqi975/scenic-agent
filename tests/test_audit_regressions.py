"""Regressions for the defects found by a follow-up audit of the runtime.

Each one was **reproduced end-to-end** before being fixed, and each has the same shape:
the system kept working, so nothing failed, but the visitor got a worse answer than the
data supported. That is why these assert on the visitor-visible result and on the exact
argument passed to the tool, not on internal state.
"""

from __future__ import annotations

import asyncio

import pytest

from tests.conftest import async_test


# --------------------------------------------------------------------------- route params
@async_test
async def test_route_parameters_come_from_the_visitor_not_the_instruction(fake_catalog):
    """The fallback brain must plan with the visitor's stated duration and audience.

    The rule brain used to extract route parameters from its own instruction
    ("按游客的时长、人群、天气与设施要求规划并校验路线"), which contains no duration, audience
    or preference - so every fallback-mode route silently planned on the 240-minute
    default and then validated that plan against itself, hiding the loss.
    """
    from app.agents.registry import AGENT_CARDS
    from app.agents.runtime import RunContext, run_agent
    from app.services.brains import RuleBrain
    from app.tools.registry import TOOL_REGISTRY

    captured: list[dict] = []
    real_execute = TOOL_REGISTRY.execute

    async def capture(*, agent, tool_name, arguments=None, timeout=None):  # noqa: ANN001
        if tool_name == "calculate_route":
            captured.append(dict(arguments or {}))
        return await real_execute(agent=agent, tool_name=tool_name, arguments=arguments, timeout=timeout)

    TOOL_REGISTRY.execute = capture
    try:
        context = RunContext(
            conversation_id="c1",
            user_message="带老人玩整整一天，想看瀑布",
            mode="fallback",
        )
        await run_agent(
            AGENT_CARDS["route_agent"],
            instruction="按游客的时长、人群、天气与设施要求规划并校验路线",
            context=context,
            brain=RuleBrain(),
        )
    finally:
        TOOL_REGISTRY.execute = real_execute

    assert captured, "calculate_route was never called"
    arguments = captured[0]
    assert arguments["duration_minutes"] == 480, arguments
    assert "老年游客" in (arguments.get("groups") or []), arguments
    assert "瀑布" in (arguments.get("preferences") or []), arguments


@async_test
async def test_a_full_day_elderly_route_is_not_planned_as_the_default(fake_catalog):
    """The visitor-visible budget must reflect what they asked for."""
    from app.services.orchestrator import answer_async

    result = await answer_async("带老人玩整整一天，想看瀑布", mode="fallback")

    assert result["status"] == "success", result.get("error")
    assert "480" in result["message"], result["message"]


# --------------------------------------------------------------------------- retrieval policy
def test_retrieval_policy_is_keyed_on_the_visitor_question():
    """The mandated-evidence rule must classify the visitor's question.

    Keyed on the instruction, a generically worded delegation ("回答游客的问题") skipped
    the forced retrieval entirely, so a ticket-price question could be answered from the
    model's own memory.
    """
    from app.agents.policy import required_tools

    generic_instruction = "回答游客的问题"
    assert required_tools("knowledge_agent", generic_instruction) == []
    assert "rag_search" in required_tools("knowledge_agent", "九寨沟门票多少钱")


# --------------------------------------------------------------------------- object shapes
def test_agent_cards_serialises_slotted_dataclasses():
    """``AgentCard`` is ``@dataclass(slots=True)`` and has no ``__dict__``."""
    from app.main import agent_cards

    cards = agent_cards()
    assert cards, "no agent cards were returned"
    for name, card in cards.items():
        assert isinstance(card, dict), name
        assert card.get("name") == name
        assert "system_prompt" in card


@pytest.mark.parametrize(
    "payload",
    [
        "not-a-dict",
        {},
        {"function": "not-a-dict"},
        {"function": {"name": "search_knowledge", "arguments": "not json"}},
        {"function": {"arguments": ["a", "list"]}},
        {"function": {"arguments": "[1, 2]"} },
    ],
)
def test_malformed_tool_call_payloads_never_raise(payload):
    """A malformed call must degrade to an empty call, not an ``AttributeError``.

    The registry rejects an empty name or bad arguments on validation, which is a visible
    and recoverable outcome; an ``AttributeError`` escapes the ReAct loop and collapses
    the whole run to the catch-all salvage path.
    """
    from app.services.llm import ToolCall

    call = ToolCall.from_payload(payload)
    assert isinstance(call.arguments, dict)
    assert isinstance(call.name, str)


@async_test
async def test_malformed_model_response_becomes_llm_unavailable():
    """The documented contract is ``LLMUnavailable``, not a raw ``AttributeError``."""
    from app.services.llm import LLMClient, LLMUnavailable
    from tests.fakes import fake_transport

    # Structurally unusable bodies: a non-object choice, or a non-list ``choices``.
    for body in ({"choices": ["oops"]}, {"choices": "oops"}):
        client = LLMClient(
            base_url="https://example.invalid/v1",
            api_key="k",
            transport=fake_transport([body]),
        )
        with pytest.raises(LLMUnavailable):
            await client.complete([{"role": "user", "content": "hi"}])


@async_test
async def test_a_list_typed_message_degrades_to_empty_content():
    """A wrong-typed ``message`` is junk, but it is still a *reply*: return, don't raise.

    The runtime already handles empty content through its ``no_answer`` path, so
    degrading here is both safer and consistent with ``ToolCall.from_payload``.
    """
    from app.services.llm import LLMClient
    from tests.fakes import fake_transport

    client = LLMClient(
        base_url="https://example.invalid/v1",
        api_key="k",
        transport=fake_transport([{"choices": [{"message": ["oops"]}]}]),
    )
    reply = await client.complete([{"role": "user", "content": "hi"}])
    assert reply.content == ""
    assert reply.tool_calls == []


# --------------------------------------------------------------------------- evidence plumbing
@async_test
async def test_retrieved_documents_reach_the_legacy_composer(fake_catalog):
    """Tool evidence must be reachable from the run, not stranded in a local list.

    ``run_agent`` returns the evidence, and the supervisor calls ``context.absorb`` - so
    the test has to absorb the result exactly as the orchestrator does, otherwise it is
    not exercising the production plumbing.
    """
    from app.agents.registry import AGENT_CARDS
    from app.agents.runtime import RunContext, run_agent
    from app.services.brains import RuleBrain
    from app.services.orchestrator import retrieval_items

    context = RunContext(conversation_id="c1", user_message="九寨沟门票多少钱", mode="fallback")
    result = await run_agent(
        AGENT_CARDS["knowledge_agent"],
        instruction="依据景区知识库回答问题并给出引用",
        context=context,
        brain=RuleBrain(),
    )
    context.absorb(result)

    items = retrieval_items(context)
    assert items, "retrieved documents were not reachable from the run context"
    assert any("190" in str(item.get("content") or "") for item in items)


@async_test
async def test_legacy_composer_uses_the_retrieved_evidence(fake_catalog):
    """The documented 「根据景区资料：…」 wording must be able to quote real evidence."""
    from app.agents.registry import AGENT_CARDS
    from app.agents.runtime import RunContext, run_agent
    from app.services.brains import RuleBrain
    from app.services.orchestrator import compose_legacy_answer

    context = RunContext(conversation_id="c1", user_message="九寨沟门票多少钱", mode="fallback")
    context.intent = "qa"
    result = await run_agent(
        AGENT_CARDS["knowledge_agent"],
        instruction="依据景区知识库回答问题并给出引用",
        context=context,
        brain=RuleBrain(),
    )
    context.absorb(result)

    answer = await compose_legacy_answer(context)
    assert "190" in answer, answer


# --------------------------------------------------------------------------- realtime honesty
@async_test
async def test_a_broken_realtime_query_is_an_error_not_a_missing_data_source(fake_catalog):
    """A query bug must not be reported to the visitor as "no data source exists"."""
    from app.tools import realtime

    def broken_engine():
        raise RuntimeError("simulated driver failure")

    original = realtime.engine
    realtime.engine = broken_engine
    try:
        from app.tools.registry import TOOL_REGISTRY

        outcome = await TOOL_REGISTRY.execute(
            agent="realtime_agent", tool_name="get_realtime_status", arguments={"subject": "天气"}
        )
    finally:
        realtime.engine = original

    assert outcome.status == "error", outcome.status
    assert outcome.error_code == "tool_execution_error"
    assert outcome.result is None


# --------------------------------------------------------------------------- event loop safety
@async_test
async def test_reading_history_does_not_block_the_event_loop(fake_catalog):
    """The per-request history read is synchronous, so it must be offloaded."""
    from app.services import conversations, orchestrator

    ticks = 0

    async def heartbeat() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.005)
            ticks += 1

    def slow_history(cid: str, limit: int = 6):  # noqa: ANN001, ANN202
        import time

        time.sleep(0.15)
        return []

    original = conversations.recent_history
    orchestrator.recent_history = slow_history
    beat = asyncio.create_task(heartbeat())
    try:
        await orchestrator.answer_async("九寨沟门票多少钱", mode="fallback")
    finally:
        beat.cancel()
        orchestrator.recent_history = original

    assert ticks >= 5, f"the event loop was blocked by the history read ({ticks} ticks)"
