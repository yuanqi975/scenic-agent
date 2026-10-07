"""End-to-end request regressions for the deterministic (``fallback``) path.

These are the tests whose absence let three request-breaking bugs survive in a suite of
53 passing tests, because every one of them was invisible to a unit test that injected
its own collaborators:

1. ``_multi_agent_pass`` called ``.get()`` on :class:`AgentResult` objects, raising
   ``AttributeError`` on **every** request - so ``status`` was always ``error``.
2. The rule brain searched the knowledge base with the *supervisor's instruction*
   ("依据景区知识库回答问题并给出引用") instead of the visitor's question, so the
   deterministic answer path never retrieved anything and always replied "暂未查询到
   足够信息" - even for questions the knowledge base answers.
3. ``retrieve`` and ``known_entity_ids`` bound their catalog reader as a **default
   argument**, capturing the real database reader at import time. The injected fake
   catalog was therefore ignored and every call paid ~12 s of connect timeouts.

The rule that makes these testable is simple: drive ``answer_async`` the way the API
does, assert on the visitor-visible result, and never accept the "no information"
notice as a valid answer for an answerable question.
"""

from __future__ import annotations

import time

from tests.conftest import async_test

#: The degraded-path notice that must NOT be the answer for an answerable question.
NO_INFO_NOTICE = "暂未在九寨沟景区官方知识库中查询到足够信息"

#: The visitor's question must survive into the tool call - this is the exact string the
#: supervisor instruction used to be mistaken for.
SUPERVISOR_INSTRUCTION = "依据景区知识库回答问题并给出引用"


def test_pronoun_resolution_carries_an_attraction_through_five_turns(fake_catalog):
    """A fifth-turn “它” must still resolve to the earlier named attraction."""
    from app.services.orchestrator import _resolve_context_message

    history = [
        {"role": "user", "content": "五花海在哪里？"},
        {"role": "assistant", "content": "五花海位于日则沟。"},
        {"role": "user", "content": "它有哪些看点？"},
        {"role": "assistant", "content": "湖水色彩丰富。"},
        {"role": "user", "content": "明天去需要准备什么？"},
        {"role": "assistant", "content": "注意当天公告；行程还会经过大小金铃海。"},
        {"role": "user", "content": "我会带老人。"},
        {"role": "assistant", "content": "建议短距离步行，可游览大小金铃海、珍珠滩。"},
    ]

    assert _resolve_context_message("它适合老人吗？", history) == "五花海适合老人吗？"


@async_test
async def test_an_answerable_question_is_answered_with_citations(fake_catalog):
    """The deterministic path must actually retrieve, not just report failure."""
    from app.services.orchestrator import answer_async

    result = await answer_async("九寨沟门票多少钱", mode="fallback")

    assert result["status"] == "success", result.get("error")
    assert not result.get("error")
    assert NO_INFO_NOTICE not in result["message"], result["message"]
    assert "190" in result["message"]
    assert result["citations"], "an answered factual question must carry citations"
    assert result["tool_calls"] >= 1


@async_test
async def test_narrow_questions_do_not_echo_unrelated_retrieved_faqs(fake_catalog):
    """A focused question must not display neighbouring FAQ pairs from one chunk."""
    from app.services.orchestrator import answer_async

    rest = await answer_async("哪里可以休息的地方吗", mode="fallback")
    assert rest["status"] == "success"
    assert "游客服务中心" in rest["message"] or "诺日朗服务中心" in rest["message"]
    assert "九寨沟一天能逛完吗" not in rest["message"]
    assert "能住在九寨沟内吗" not in rest["message"]

    direction = await answer_async("五花海怎么去？", mode="fallback")
    assert direction["status"] == "success"
    assert "五花海" in direction["message"]
    assert "九寨沟一天能逛完吗" not in direction["message"]
    assert "能住在九寨沟内吗" not in direction["message"]


@async_test
async def test_the_rule_brain_searches_with_the_visitor_question(fake_catalog):
    """The retrieval query is the visitor's wording, never the internal instruction."""
    from app.agents.registry import AGENT_CARDS
    from app.agents.runtime import RunContext, run_agent
    from app.services.brains import RuleBrain

    captured: list[dict] = []
    from app.tools.registry import TOOL_REGISTRY

    real_execute = TOOL_REGISTRY.execute

    async def capture(*, agent, tool_name, arguments=None, timeout=None):  # noqa: ANN001
        captured.append({"tool": tool_name, "arguments": dict(arguments or {})})
        return await real_execute(agent=agent, tool_name=tool_name, arguments=arguments, timeout=timeout)

    TOOL_REGISTRY.execute = capture
    try:
        context = RunContext(conversation_id="c1", user_message="九寨沟门票多少钱", mode="fallback")
        await run_agent(
            AGENT_CARDS["knowledge_agent"],
            instruction=SUPERVISOR_INSTRUCTION,
            context=context,
            brain=RuleBrain(),
        )
    finally:
        TOOL_REGISTRY.execute = real_execute

    search_calls = [item for item in captured if item["tool"] in {"search_knowledge", "rag_search"}]
    assert search_calls, "the knowledge agent must call a retrieval tool"
    query = search_calls[0]["arguments"].get("query", "")
    assert query == "九寨沟门票多少钱"
    assert query != SUPERVISOR_INSTRUCTION


@async_test
async def test_every_request_succeeds_without_a_database_and_stays_fast(fake_catalog):
    """No request may fail, and none may pay a connect timeout.

    The 2 s ceiling is far above real work (measured at ~0.02 s) and far below the 3 s
    ``connect_timeout``, so it fails specifically when a synchronous database call slips
    back onto the event loop.
    """
    from app.services.orchestrator import answer_async

    questions = [
        "九寨沟门票多少钱",
        "明天下雨，带老人玩半天，沿途最好有厕所",
        "游客中心的厕所不能使用",
        "五花海海拔多少",
    ]
    for question in questions:
        started = time.perf_counter()
        result = await answer_async(question, mode="fallback")
        elapsed = time.perf_counter() - started

        assert result["status"] == "success", (question, result.get("error"))
        assert not result.get("error"), (question, result.get("error"))
        assert str(result["message"]).strip(), question
        assert elapsed < 2.0, f"{question} took {elapsed:.2f}s"


@async_test
async def test_a_reported_problem_asks_for_confirmation_instead_of_claiming_a_write(fake_catalog):
    """The fallback path must not invent a submitted ticket."""
    from app.services.orchestrator import answer_async

    result = await answer_async("游客中心的厕所不能使用", mode="fallback")

    assert result["status"] == "success"
    assert result["intent"] == "feedback"
    assert "feedback_agent" in [entry.get("agent") for entry in result["agent_chain"]]
    # It must ask, and it must not assert that anything was submitted.
    assert "提交" in result["message"]
    assert "已提交" not in result["message"]
    assert "正在等待景区人工审核" not in result["message"]


@async_test
async def test_retrieval_respects_an_injected_catalog_reader(fake_catalog):
    """A caller-supplied reader must win.

    This is the regression for the default-argument capture: the injected reader is
    ignored if ``rows`` is bound at import time, and the call silently goes to the
    database instead.
    """
    from app.services.retrieval import known_entity_ids, retrieve

    calls: list[str] = []

    def reader(table: str, limit: int = 100, offset: int = 0, park_id: str | None = None):
        calls.append(table)
        from tests.fakes import TABLES

        return [dict(item) for item in TABLES.get(table, [])[offset : offset + limit]]

    assert known_entity_ids("五花海海拔多少", rows=reader) == ["attr_001"]
    assert calls, "the injected reader was ignored"

    calls.clear()
    items = retrieve("门票多少钱", use_cache=False, rows=reader)
    assert calls, "retrieve ignored the injected reader"
    assert items, "the injected catalog contains an answer to this question"
