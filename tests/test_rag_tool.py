from tests.conftest import async_test


@async_test
async def test_rag_search_is_registered_and_returns_citations(fake_catalog):
    from app.tools.registry import TOOL_REGISTRY

    outcome = await TOOL_REGISTRY.execute(
        agent="knowledge_agent", tool_name="rag_search", arguments={"query": "五花海"}
    )
    assert outcome.ok
    assert "items" in outcome.result
    assert "citations" in outcome.result
