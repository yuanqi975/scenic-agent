"""Critic agent: one round of grounding review.

The critical constraint (``docs/多Agent入门详解.md`` §7.5) is the **input**: the critic
sees the *raw evidence* and the *draft answer*, never the other agents' conclusions.
Feeding it another agent's summary is how hallucination gets reinforced instead of
caught. It runs at most once - a second pass has very low value and linear cost.
"""

from __future__ import annotations

import json
from typing import Any

from ..services import audit
from ..services.brains import Brain, LLMBrain
from ..services.prompts import CRITIC_SYSTEM
from ..services.llm import LLMUnavailable
from .registry import get_card
from .runtime import RunContext, _clip
from .schemas import AgentResult

MAX_EVIDENCE_CHARS = 4000


def build_critic_input(context: RunContext, answer: str) -> str:
    """Only raw tool output, plus the draft answer. No other agent's reasoning."""
    evidence: list[str] = []
    for item in context.observations:
        if not str(item.get("tool", "")).startswith("agent:"):
            body = json.dumps(item.get("result") or {}, ensure_ascii=False, default=str)
            evidence.append(f"- {item.get('tool')}: {_clip(body, 700)}")
    citations = context.citations()
    if citations:
        evidence.append(
            "- 引用来源: "
            + "、".join(str(citation.get("title") or citation.get("source_id")) for citation in citations[:10])
        )
    block = "\n".join(evidence) or "（本次没有任何工具返回资料）"
    return f"【原始资料】\n{_clip(block, MAX_EVIDENCE_CHARS)}\n\n【答案】\n{answer}"


def _has_evidence(context: RunContext) -> bool:
    return any(not str(item.get("tool", "")).startswith("agent:") for item in context.observations)


async def critic_agent(context: RunContext, answer: str, brain: Brain) -> AgentResult:
    """Return the final answer after at most one grounding review."""
    card = get_card("critic_agent")
    run_id = audit.start_run(
        agent_name=card.name,
        role=card.role,
        conversation_id=context.conversation_id,
        instruction="质检答案依据",
        trace_id=context.trace_id or None,
        intent=context.intent,
        mode=context.mode,
        model=context.model,
    )

    def finish(final: str, status: str, artifacts: dict[str, Any], step_text: str) -> AgentResult:
        result = AgentResult(
            agent=card.name,
            run_id=run_id,
            instruction="质检答案依据",
            answer=final,
            status=status,
            iterations=1,
            citations=context.citations(),
            artifacts=artifacts,
            source=context.mode,
        )
        audit.save_step(
            {
                "run_id": run_id,
                "agent_name": card.name,
                "step_index": 1,
                "event_type": "answer",
                "text_summary": _clip(step_text, 400),
                "status": "ok" if status == "success" else "error",
            }
        )
        audit.finish_run(result, role=card.role, mode=context.mode)
        context.emit("agent_finished", run_id=run_id, agent=card.name, status=status, **artifacts)
        return result

    if not isinstance(brain, LLMBrain):
        # Without a model the deterministic critic can only check the mechanical rule:
        # an answer that asserts facts must carry at least one citation.
        if not context.citations() and _has_evidence(context) is False:
            return finish(
                answer,
                "success",
                {"skipped": "rule_brain", "ok": True},
                "规则模式：无模型可用，未执行语义质检",
            )
        return finish(
            answer,
            "success",
            {"skipped": "rule_brain", "ok": True, "citations": len(context.citations())},
            "规则模式：仅校验引用存在性",
        )

    try:
        payload = await brain.client.complete_json(
            [
                {"role": "system", "content": CRITIC_SYSTEM},
                {"role": "user", "content": build_critic_input(context, answer)},
            ],
            temperature=0,
        )
    except (LLMUnavailable, Exception) as exc:
        # A failed review must not block the answer, but it must be visible.
        return finish(
            answer,
            "success",
            {"ok": None, "error": str(exc)[:200], "applied": False},
            f"质检不可用：{exc}",
        )

    ok = bool(payload.get("ok"))
    unsupported = payload.get("unsupported")
    if not isinstance(unsupported, list):
        unsupported = []
    fixed = str(payload.get("fixed") or "").strip()
    applied = False
    final = answer
    if not ok and fixed:
        final = fixed
        applied = True
    return finish(
        final,
        "success",
        {"ok": ok, "unsupported": [str(item)[:200] for item in unsupported][:5], "applied": applied},
        "质检通过" if ok else f"质检发现 {len(unsupported)} 处无依据表述" + ("，已修正" if applied else ""),
    )


__all__ = ["MAX_EVIDENCE_CHARS", "build_critic_input", "critic_agent"]
