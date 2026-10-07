"""Question-focused answers for the deterministic fallback path.

Retrieval returns *evidence*, not ready-to-display prose.  In fallback mode there is
no language model to summarise that evidence, so rendering complete chunks directly
leaks unrelated FAQ entries into the visitor answer.  This module keeps the displayed
answer scoped to the question while leaving the original evidence available for audit
and citations.
"""

from __future__ import annotations

import re
from typing import Any


_DIRECTION_WORDS = ("怎么去", "怎么走", "如何去", "怎样去", "怎么到", "如何到", "前往")
_REST_WORDS = ("休息", "歇脚", "坐一会", "休息区", "休息点")
_QUESTION_SPLIT = re.compile(r"(?=问[：:])")
_ANSWER_SPLIT = re.compile(r"(?<=[。！？；])\s*|\n+")
_IGNORED_BIGRAMS = {
    "九寨", "寨沟", "景区", "请问", "可以", "哪里", "怎么", "如何", "怎样", "一下", "请问",
    "能否", "是否", "需要", "想去", "游客", "地方", "什么", "有关", "问题",
}


def _citations(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "source_type": item.get("source_type")
            or ("attraction" if item.get("attraction_id") else "facility"),
            "source_id": item.get("facility_id") or item.get("attraction_id") or item.get("source_id"),
            "title": item.get("name") or item.get("title") or item.get("source_id"),
            "authority": "official",
            "updated_at": item.get("updated_at"),
        }
        for item in items
    ]


def rest_answer(
    message: str,
    *,
    facilities: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]] | None:
    """Answer a rest-stop question only with known service locations."""
    if not any(word in message for word in _REST_WORDS):
        return None
    by_name = {str(item.get("name") or ""): item for item in facilities}
    gate = by_name.get("九寨沟游客服务中心")
    nuorilang = by_name.get("诺日朗服务中心（诺日朗餐厅）")
    locations = [item for item in (gate, nuorilang) if item]
    if not locations:
        return (
            "公开资料未收录明确的休息点位；建议就近向观光车站或工作人员询问。",
            [],
        )

    parts: list[str] = []
    if gate:
        parts.append("沟口可到九寨沟游客服务中心询问并短暂停留")
    if nuorilang:
        parts.append("诺日朗中心可到诺日朗服务中心（诺日朗餐厅）用餐、休息")
    return (
        "可以优先去：" + "；".join(parts) + "。"
        "景区公开资料未提供完整休息座椅或休息区清单；若你告诉我当前所在景点，我可以继续帮你查附近已公开的服务设施。",
        _citations(locations),
    )


def attraction_direction_answer(
    message: str,
    *,
    attractions: list[dict[str, Any]],
    facilities: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]] | None:
    """Answer a named attraction's access question from catalogued stops only."""
    if not any(word in message for word in _DIRECTION_WORDS):
        return None
    matches = [
        item for item in attractions
        if item.get("name") and str(item["name"]) in message
    ]
    if not matches:
        return None
    # Two named attractions are a point-to-point navigation request. Let the route
    # tool use the route graph instead of answering with a generic single-POI stop.
    if len(matches) >= 2:
        return None
    attraction = max(matches, key=lambda item: len(str(item.get("name") or "")))
    attraction_id = attraction.get("attraction_id")
    linked = [
        item for item in facilities
        if (item.get("nearby_attraction_id") or item.get("related_attraction_id")) == attraction_id
    ]
    stop = next((item for item in linked if item.get("facility_type") == "观光车站"), None)
    platform = next((item for item in linked if item.get("facility_type") == "观景台"), None)
    name = str(attraction["name"])
    if stop:
        answer = f"前往{name}：乘景区观光车至{stop.get('name')}下车，再按当天开放的栈道和现场指引游览。"
    else:
        answer = f"公开资料未收录{name}对应的下车点；请在诺日朗中心站向工作人员确认当日观光车停靠和栈道开放情况。"
    if platform:
        answer += f"观景可前往{platform.get('name')}。"
    return answer, _citations([attraction, *([stop] if stop else []), *([platform] if platform else [])])


def _query_bigrams(message: str) -> set[str]:
    text = "".join(re.findall(r"[\u4e00-\u9fff]", message))
    return {
        text[index : index + 2]
        for index in range(max(0, len(text) - 1))
        if text[index : index + 2] not in _IGNORED_BIGRAMS
    }


def _faq_answers(content: str, query_terms: set[str]) -> list[tuple[int, str]]:
    matches: list[tuple[int, str]] = []
    for block in _QUESTION_SPLIT.split(content):
        if not block.startswith("问") or "答" not in block:
            continue
        question, answer = re.split(r"答[：:]", block, maxsplit=1)
        score = sum(1 for term in query_terms if term in question)
        if score:
            matches.append((score * 4, answer.strip()))
    return matches


def focused_evidence_answer(message: str, items: list[dict[str, Any]]) -> str:
    """Render at most two directly relevant statements from retrieved evidence.

    Raw chunks can contain several FAQ pairs.  A sentence is eligible only if it
    overlaps the visitor's wording, or is the answer paired with a matching FAQ
    question.  This deliberately prefers a precise "not enough information" reply to
    showing a plausible but unrelated paragraph.
    """
    terms = _query_bigrams(message)
    ranked: list[tuple[int, int, str]] = []
    seen: set[str] = set()
    for source_index, item in enumerate(items):
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        for score, answer in _faq_answers(content, terms):
            for sentence in _ANSWER_SPLIT.split(answer):
                sentence = sentence.strip(" -•\t")
                if sentence:
                    ranked.append((score, -source_index, sentence))
        for sentence in _ANSWER_SPLIT.split(content):
            sentence = sentence.strip(" -•\t")
            if not sentence or sentence.startswith("问") or sentence.startswith("答"):
                continue
            score = sum(1 for term in terms if term in sentence)
            if score:
                ranked.append((score, -source_index, sentence))

    ranked.sort(reverse=True)
    selected: list[str] = []
    for _, _, sentence in ranked:
        normalised = re.sub(r"\s+", "", sentence)
        if normalised in seen:
            continue
        seen.add(normalised)
        selected.append(sentence)
        if len(selected) == 2:
            break
    if not selected:
        return "知识库暂未检索到与这个问题直接相关的公开信息。"
    answer = "".join(selected)
    return answer if len(answer) <= 360 else answer[:359].rstrip("，、；") + "。"


__all__ = ["attraction_direction_answer", "focused_evidence_answer", "rest_answer"]
