def test_high_risk_questions_require_rag():
    from app.agents.policy import required_tools

    assert "rag_search" in required_tools("knowledge_agent", "门票多少钱")
    assert "rag_search" in required_tools("realtime_agent", "今天是否限流")


def test_route_planning_requires_structured_and_rag_evidence():
    from app.agents.policy import required_tools

    assert required_tools("route_agent", "带老人安排半天路线") == [
        "search_attractions",
        "calculate_route",
    ]
