"""Agent registry: the code is the source of truth for prompts, tools and roles.

Why code rather than a database table: tool whitelists and prompts are security- and
correctness-critical, so they must be reviewable in a diff. A database row can flip an
agent's ``enabled`` flag or widen a whitelist without any review at all. The
``agent_registry`` table therefore stores only operational state (enabled + metrics).

Naming contract - the project's own multi-agent guide (``docs/多Agent入门详解.md``)
names its experts differently from the implementation, so the mapping is stated here
once, and exposed through :func:`naming_contract` so the admin UI can show it:

| guide vocabulary     | implemented agent  | note                                          |
|----------------------|--------------------|-----------------------------------------------|
| ``knowledge_expert`` | ``knowledge_agent``| also the guide's ``retriever_agent``           |
| ``route_expert``     | ``route_agent``    |                                                |
| ``facility_expert``  | ``knowledge_agent``| folded in: facility lookups are knowledge work |
| ``ticket_expert``    | ``knowledge_agent``| ticketing is answered from the same knowledge  |
| ``realtime_expert``  | ``realtime_agent`` |                                                |
| ``feedback_expert``  | ``feedback_agent`` |                                                |
| (guide's responder)  | ``response_agent`` | the guide's ``answer_agent``                   |
| ``critic``           | ``critic_agent``   |                                                |
| ``router``           | (not an agent)     | intent routing lives in ``services/intent.py`` |
"""

from __future__ import annotations

from .schemas import (
    ROLE_CRITIC,
    ROLE_RESPONDER,
    ROLE_SUPERVISOR,
    ROLE_WORKER,
    AgentCard,
)

#: Tools each specialist is allowed to call. Anything else is rejected at runtime.
KNOWLEDGE_TOOLS: tuple[str, ...] = (
    "rag_search",
    "search_knowledge",
    "search_attractions",
    "get_attraction_detail",
    "get_facilities",
    "get_faqs",
    "get_conversation_history",
)
ROUTE_TOOLS: tuple[str, ...] = (
    "rag_search",
    "search_attractions",
    "get_attraction_detail",
    "get_facilities",
    "find_facilities_near",
    "get_realtime_status",
    "calculate_route",
    "validate_route",
    "search_knowledge",
)
REALTIME_TOOLS: tuple[str, ...] = (
    "rag_search",
    "get_realtime_status",
    "get_park_notices",
    "get_facilities",
    "search_knowledge",
)
FEEDBACK_TOOLS: tuple[str, ...] = (
    "search_attractions",
    "find_facilities_near",
    "get_conversation_history",
    "check_pending_feedback",
    "create_feedback_candidate",
)
SUPERVISOR_TOOLS: tuple[str, ...] = ("get_conversation_history",)
RESPONSE_TOOLS: tuple[str, ...] = ()


def _card(
    name: str,
    role: str,
    description: str,
    tools: tuple[str, ...],
    *,
    max_steps: int = 6,
    parallelizable: bool = True,
    requires_confirmation: bool = False,
) -> AgentCard:
    from ..services.prompts import agent_system_prompts

    return AgentCard(
        name=name,
        role=role,
        description=description,
        system_prompt=agent_system_prompts()[name],
        tools=tools,
        max_steps=max_steps,
        parallelizable=parallelizable,
        requires_confirmation=requires_confirmation,
    )


def build_agent_cards() -> dict[str, AgentCard]:
    """Construct every agent card. Imports prompts lazily to avoid a cycle."""
    return {
        "supervisor": _card(
            "supervisor",
            ROLE_SUPERVISOR,
            "理解游客目标、拆分任务、并行派发专员，并判断何时收敛",
            SUPERVISOR_TOOLS,
            max_steps=1,
            parallelizable=False,
        ),
        "knowledge_agent": _card(
            "knowledge_agent",
            ROLE_WORKER,
            "景区知识问答：景点、票价、开放时间、优惠政策、游览规定，必须给出引用",
            KNOWLEDGE_TOOLS,
        ),
        "route_agent": _card(
            "route_agent",
            ROLE_WORKER,
            "游览路线规划：提取约束 → 调用确定性算法 → 校验 → 不通过则重算",
            ROUTE_TOOLS,
        ),
        "realtime_agent": _card(
            "realtime_agent",
            ROLE_WORKER,
            "实时信息：天气、客流、当前开放状态、临时关闭、设施维护；无数据必须如实声明",
            REALTIME_TOOLS,
        ),
        "feedback_agent": _card(
            "feedback_agent",
            ROLE_WORKER,
            "游客反馈受理：提取事实、追加确认、确认后才创建待审核知识候选",
            FEEDBACK_TOOLS,
            parallelizable=False,
            requires_confirmation=True,
        ),
        "response_agent": _card(
            "response_agent",
            ROLE_RESPONDER,
            "汇总各专员结果、消除重复与冲突、区分官方/实时/推断、生成面向游客的答复",
            RESPONSE_TOOLS,
            max_steps=1,
            parallelizable=False,
        ),
        "critic_agent": _card(
            "critic_agent",
            ROLE_CRITIC,
            "质检：只依据原始资料核对答案依据，不参考其它专员的结论",
            RESPONSE_TOOLS,
            max_steps=1,
            parallelizable=False,
        ),
    }


AGENT_CARDS: dict[str, AgentCard] = build_agent_cards()

#: Specialists the supervisor may delegate to, in stable order.
SPECIALIST_AGENTS: tuple[str, ...] = (
    "knowledge_agent",
    "route_agent",
    "realtime_agent",
    "feedback_agent",
)

#: Registry used by the supervisor loop.
ORCHESTRATORS: tuple[str, ...] = ("supervisor",)

#: Registry used after delegation.
SYNTHESIZERS: tuple[str, ...] = ("response_agent", "critic_agent")


def get_card(name: str) -> AgentCard:
    try:
        return AGENT_CARDS[name]
    except KeyError as exc:
        raise KeyError(f"未注册的 Agent：{name}") from exc


def agent_names() -> list[str]:
    return list(AGENT_CARDS)


def specialist_names() -> list[str]:
    return list(SPECIALIST_AGENTS)


def delegation_prompt_roster() -> list[str]:
    """The roster string handed to the supervisor model."""
    return list(SPECIALIST_AGENTS)


def naming_contract() -> dict[str, str]:
    """Guide vocabulary -> implemented agent, for the docs and the admin panel."""
    return {
        "knowledge_expert": "knowledge_agent",
        "retriever_agent": "knowledge_agent",
        "facility_expert": "knowledge_agent",
        "ticket_expert": "knowledge_agent",
        "route_expert": "route_agent",
        "realtime_expert": "realtime_agent",
        "feedback_expert": "feedback_agent",
        "answer_agent": "response_agent",
        "critic": "critic_agent",
        "router": "services/intent.py::classify_with_cache",
    }


__all__ = [
    "AGENT_CARDS",
    "ORCHESTRATORS",
    "SPECIALIST_AGENTS",
    "SYNTHESIZERS",
    "agent_names",
    "delegation_prompt_roster",
    "get_card",
    "naming_contract",
    "specialist_names",
]
