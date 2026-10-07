"""Response agent: merge every specialist result into one visitor-facing answer.

Design decision (fused plan): when exactly one specialist answered and produced no
conflicts, the answer is passed through with citations instead of paying a second model
round-trip. The agent still runs - its run row, timing and citations are recorded - so
the collaboration chain stays visible in the audit trail.
"""

from __future__ import annotations

import json
from typing import Any

from ..core.config import SERVICE_PHONE
from ..services import audit
from ..services.brains import Brain, LLMBrain
from ..services.prompts import RESPONSE_AGENT_SYSTEM
from .registry import get_card
from .runtime import RunContext, _clip, run_agent
from .schemas import AgentResult

AUTHORITY_LABEL = {
    "official": "官方知识",
    "community": "游客反馈（已人工审核）",
    "reference": "参考资料",
    "realtime": "实时数据",
    "inferred": "推断",
}

CONFLICT_MARKERS = ("冲突", "不一致", "两个口径", "另一个说法")


def _authority_counts(citations: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for citation in citations:
        key = str(citation.get("authority") or "reference")
        counts[key] = counts.get(key, 0) + 1
    return counts


def needs_synthesis(context: RunContext) -> bool:
    """Is a model round-trip actually required to merge the results?"""
    if context.artifacts.get("structured_route_ready"):
        return False
    completed = [entry for entry in context.agent_chain if entry.get("status") == "success"]
    if len(completed) != 1:
        return True
    answer = str(completed[0].get("answer") or "")
    if not answer.strip():
        return True
    if any(marker in answer for marker in CONFLICT_MARKERS):
        return True
    # A single specialist that explicitly reports missing realtime data still needs the
    # response agent to frame it for the visitor.
    return False


def _unusable_model_answer(answer: str) -> bool:
    """Reject provider outputs that merely echo citation/source identifiers."""
    text = str(answer or "").strip()
    if len(text) < 18:
        return True
    source_only = ("仅包含以下引用来源", "仅包含引用来源", "原始资料仅提供引用来源", "只有以下引用来源", "引用来源：", "来源：", "source_id", "document_id", "没有任何工具返回资料", "无法核实")
    if any(marker in text for marker in source_only) and not any(
        marker in text for marker in ("位于", "门票", "路线", "开放", "景点", "建议", "海拔", "预约")
    ):
        return True
    return False


def passthrough(context: RunContext) -> str:
    """Direct answer from the single successful specialist, plus citation framing."""
    completed = [entry for entry in context.agent_chain if entry.get("status") == "success"]
    if not completed:
        return (
            "抱歉，暂时无法确认这个问题的答案。建议咨询九寨沟景区游客服务中心"
            f"（{SERVICE_PHONE}）获取准确信息。"
        )
    answer = str(completed[0].get("answer") or "").strip()
    citations = context.citations()
    if citations:
        labels = {AUTHORITY_LABEL.get(str(item.get("authority") or "reference"), "参考资料") for item in citations}
        answer = f"{answer}\n\n资料来源：{'、'.join(sorted(labels))}。"
    else:
        answer = f"{answer}\n\n（本条回答未附引用来源，请以景区现场公告为准。）"
    return answer


def build_synthesis_brief(context: RunContext) -> str:
    """Give the response agent the specialist results and the evidence they used."""
    lines: list[str] = []
    for entry in context.agent_chain:
        lines.append(
            f"- 专员 {entry.get('agent')}（{entry.get('status')}）：{_clip(str(entry.get('answer') or ''), 900)}"
        )
    evidence: list[str] = []
    for item in context.observations:
        result = item.get("result") or {}
        if item.get("tool", "").startswith("agent:"):
            for citation in (result.get("citations") or [])[:3]:
                evidence.append(
                    f"  · [{citation.get('authority', 'reference')}] {citation.get('title') or citation.get('source_id')}"
                )
        else:
            body = json.dumps(result, ensure_ascii=False, default=str)
            evidence.append(f"  · {item.get('tool')}: {_clip(body, 400)}")
    return (
        f"【游客问题】{context.user_message}\n"
        f"【各专员结论】\n{chr(10).join(lines) or '（无）'}\n"
        f"【可用证据】\n{chr(10).join(evidence[:20]) or '（无）'}\n"
        "请汇总成一段面向游客的最终回答。"
    )


async def response_agent(
    context: RunContext,
    brain: Brain,
    *,
    legacy_composer: Any | None = None,
) -> AgentResult:
    """Compose the final answer and record a run row either way.

    ``legacy_composer`` lets the deterministic path reuse the project's historical
    answer wording (``LLM_MODE=fallback`` behaviour that the existing tests assert)
    instead of the generic merge below.
    """
    card = get_card("response_agent")
    # In fallback mode the legacy composer performs the question-focused rendering
    # (structured POI answers and filtered RAG evidence).  Passing a single specialist
    # answer through here would expose the raw retrieval chunk and undo that filtering.
    if not needs_synthesis(context) and not (
        context.mode == "fallback" and legacy_composer is not None
    ):
        started_answer = passthrough(context)
        run_id = audit.start_run(
            agent_name=card.name,
            role=card.role,
            conversation_id=context.conversation_id,
            instruction="直通汇总（单一专员结果）",
            trace_id=context.trace_id or None,
            intent=context.intent,
            mode=context.mode,
            model=context.model,
        )
        result = AgentResult(
            agent=card.name,
            run_id=run_id,
            instruction="直通汇总（单一专员结果）",
            answer=started_answer,
            status="success",
            iterations=0,
            tool_calls=0,
            citations=context.citations(),
            artifacts={"passthrough": True, "authority_counts": _authority_counts(context.citations())},
            source=context.mode,
        )
        audit.save_step(
            {
                "run_id": run_id,
                "agent_name": card.name,
                "step_index": 1,
                "event_type": "answer",
                "text_summary": _clip(started_answer, 400),
                "status": "ok",
            }
        )
        audit.finish_run(result, role=card.role, mode=context.mode)
        context.emit(
            "agent_finished",
            run_id=run_id,
            agent=card.name,
            status="success",
            passthrough=True,
        )
        return result

    # Model path: reuse the generic ReAct loop with no tools, so one round produces the
    # merged answer. The fallback brain returns the deterministic composition instead.
    if isinstance(brain, LLMBrain):
        instruction = build_synthesis_brief(context)
        # The brief already contains the whole context; keep history out of the way.
        original_history, context.history = context.history, []
        try:
            result = await run_agent(card, instruction=instruction, context=context, brain=brain)
        finally:
            context.history = original_history
        if not result.answer.strip() or result.status != "success":
            result.answer = passthrough(context)
            result.status = "success"
        if legacy_composer is not None and _unusable_model_answer(result.answer):
            try:
                grounded = await legacy_composer(context)
            except Exception:
                grounded = ""
            if grounded.strip():
                result.answer = grounded
                result.artifacts["grounding_fallback"] = True
        result.citations = result.citations or context.citations()
        result.artifacts.setdefault("authority_counts", _authority_counts(result.citations))
        return result

    run_id = audit.start_run(
        agent_name=card.name,
        role=card.role,
        conversation_id=context.conversation_id,
        instruction="规则汇总",
        trace_id=context.trace_id or None,
        intent=context.intent,
        mode=context.mode,
        model=None,
    )
    answer = ""
    if legacy_composer is not None:
        try:
            answer = await legacy_composer(context)
        except Exception:
            answer = ""
    if not answer.strip():
        answer = compose_rule_answer(context)
    result = AgentResult(
        agent=card.name,
        run_id=run_id,
        instruction="规则汇总",
        answer=answer,
        status="success",
        citations=context.citations(),
        artifacts={
            "rule_composed": legacy_composer is None,
            "legacy_composed": legacy_composer is not None,
            "authority_counts": _authority_counts(context.citations()),
        },
        source=context.mode,
    )
    audit.save_step(
        {
            "run_id": run_id,
            "agent_name": card.name,
            "step_index": 1,
            "event_type": "answer",
            "text_summary": _clip(answer, 400),
            "status": "ok",
        }
    )
    audit.finish_run(result, role=card.role, mode=context.mode)
    context.emit("agent_finished", run_id=run_id, agent=card.name, status="success", rule_composed=True)
    return result


def compose_rule_answer(context: RunContext) -> str:
    """Deterministic merge: keep every specialist's grounded text, label the sources."""
    sections: list[str] = []
    for entry in context.agent_chain:
        if entry.get("status") != "success":
            continue
        answer = str(entry.get("answer") or "").strip()
        if answer:
            sections.append(answer)
    if not sections:
        return passthrough(context)
    if len(sections) == 1:
        return passthrough(context)

    body = "\n\n".join(dict.fromkeys(sections))
    counts = _authority_counts(context.citations())
    if counts:
        labels = "、".join(
            f"{AUTHORITY_LABEL.get(key, key)} {value} 条" for key, value in sorted(counts.items())
        )
        body = f"{body}\n\n资料来源：{labels}。"
    return body


__all__ = [
    "RESPONSE_AGENT_SYSTEM",
    "build_synthesis_brief",
    "compose_rule_answer",
    "needs_synthesis",
    "passthrough",
    "response_agent",
]
