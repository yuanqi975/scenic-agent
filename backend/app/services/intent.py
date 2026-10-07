"""Intent recognition: model-driven routing with an explicit keyword fallback.

The keyword classifier is no longer the *primary* decision maker; it survives as the
degraded path (``LLM_MODE=fallback``) and as the last resort when the model answers
something unusable. Its behaviour is frozen so the documented contract still holds.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ..core.cache import cache_get, cache_set
from ..core.config import CACHE_TTL_SECONDS, LLM_MODE, PARK_ID, llm_configured
from .prompts import ROUTER_SYSTEM

#: Canonical intent set. ``ticket`` is recognised by the model but mapped onto
#: ``qa`` for dispatching because ticketing is answered from the same knowledge base.
INTENTS = {"qa", "recommendation", "realtime", "feedback", "ticket", "smalltalk"}

INTENT_ALIASES = {"ticket": "qa"}


#: Explicit feedback vocabulary ("I want to complain") - the frozen contract's signals.
_FEEDBACK_MARKERS = ("\u53cd\u9988", "\u6295\u8bc9", "\u5efa\u8bae", "\u635f\u574f")

#: A reported *problem with a service object* is also feedback, even without the word
#: "反馈": "游客中心的厕所不能使用" is a facility complaint that needs human review, not a
#: knowledge-base lookup. Both halves are required, so "厕所在哪里" stays a normal
#: question and "今天下雨吗" stays a realtime question.
_ISSUE_WORDS = (
    "\u4e0d\u80fd\u4f7f\u7528",  # 不能使用
    "\u65e0\u6cd5\u4f7f\u7528",  # 无法使用
    "\u4e0d\u80fd\u7528",        # 不能用
    "\u65e0\u6cd5\u7528",        # 无法用
    "\u574f\u4e86",              # 坏了
    "\u635f\u574f",              # 损坏
    "\u6545\u969c",              # 故障
    "\u6ca1\u6c34",              # 没水
    "\u65e0\u6c34",              # 无水
    "\u5f88\u810f",              # 很脏
    "\u592a\u810f",              # 太脏
    "\u4e0d\u5e72\u51c0",        # 不干净
    "\u592a\u5dee",              # 太差
    "\u4e0d\u597d\u7528",        # 不好用
    "\u6392\u961f\u592a\u4e45",  # 排队太久
    "\u503c\u73ed\u4eba\u5458\u4e0d",  # 值班人员不…
)

#: Service objects a visitor can report a problem with. Requires an explicit 景点/沟
#: qualifier for the ambiguous one- and two-character nouns, so plain "海子" prose does
#: not accidentally become a complaint.
_ISSUE_OBJECTS = (
    "\u5395\u6240",      # 厕所
    "\u536b\u751f\u95f4",  # 卫生间
    "\u6d17\u624b\u95f4",  # 洗手间
    "\u89c2\u5149\u8f66",  # 观光车
    "\u666f\u533a\u5de5\u4f5c\u4eba\u5458",  # 景区工作人员
    "\u5de5\u4f5c\u4eba\u5458",  # 工作人员
    "\u670d\u52a1\u4e2d\u5fc3",  # 服务中心
    "\u505c\u8f66\u573a",  # 停车场
    "\u6808\u9053",      # 栈道
    "\u6807\u8bc6",      # 标识
    "\u6307\u793a\u724c",  # 指示牌
    "\u76d6\u7ae0",      # 盖章
    "\u9910\u5385",      # 餐厅
    "\u666f\u70b9",      # 景点
)
_ISSUE_OBJECT_QUALIFIERS = ("\u666f\u533a", "\u6c9f")  # 景区 / 沟


def _contains_any(message: str, words: tuple[str, ...]) -> bool:
    return any(word in message for word in words)


# Short social turns must never start retrieval. Keep this conservative so a real
# travel question that happens to contain “谢谢” is still handled normally.
_SMALLTALK_EXACT = {
    "你好", "您好", "嗨", "哈喽", "hello", "hi", "在吗", "在不在", "谢谢", "感谢", "再见", "拜拜",
}


def is_smalltalk(message: str) -> bool:
    """Return true for a short greeting/acknowledgement with no travel request."""
    normalised = "".join(str(message or "").strip().lower().split())
    return normalised in _SMALLTALK_EXACT


def looks_like_service_complaint(message: str) -> bool:
    """A concrete problem with a concrete facility, without the word "反馈"."""
    if not _contains_any(message, _ISSUE_WORDS):
        return False
    if _contains_any(message, _ISSUE_OBJECTS):
        return True
    # 海子 / 栈道-style nouns are ambiguous on their own, so require a location word.
    return _contains_any(message, ("\u6d77\u5b50", "\u666f\u70b9")) and _contains_any(
        message, _ISSUE_OBJECT_QUALIFIERS
    )


def classify(message: str) -> str:
    """Deterministic keyword router. Frozen: existing tests and traces depend on it.

    The four documented families keep their original relative priority
    (recommendation -> feedback -> realtime -> qa). The only addition is that an
    explicit "problem with a service object" report is recognised as feedback before
    the realtime check, because such a message needs human review rather than a
    knowledge-base answer.
    """
    if is_smalltalk(message):
        return "smalltalk"
    # Keep these literals escaped: some historical files were saved through a
    # lossy console encoding, while visitor messages are valid UTF-8 Chinese.
    recommendation_words = (
        "\u8def\u7ebf", "\u600e\u4e48\u73a9", "\u8001\u4eba", "\u513f\u7ae5", "\u5b69\u5b50",
        "\u534a\u5929", "\u4e00\u5929", "\u4e00\u65e5", "\u4e24\u65e5", "\u4e09\u65e5",
        "\u6362\u4e58", "\u600e\u4e48\u5b89\u6392",
    )
    # Suitability questions are factual follow-ups, not route-generation requests.
    # This is especially important after context resolution turns「它适合老人吗」into
    #「五花海适合老人吗」.
    if _contains_any(message, ("适合", "是否适合", "能不能")) and not _contains_any(
        message, ("路线", "怎么安排", "半天", "一日游", "一天")
    ):
        return "qa"
    realtime_words = (
        "\u5929\u6c14", "\u5ba2\u6d41", "\u5173\u95ed", "\u5f00\u653e\u72b6\u6001", "\u9650\u6d41",
        "\u8f6e\u4f11", "\u4fdd\u80b2", "\u4f59\u7968", "\u6ce5\u77f3\u6d41",
    )
    if _contains_any(message, recommendation_words):
        return "recommendation"
    if _contains_any(message, realtime_words):
        return "realtime"
    # 「建议」 is both a feedback marker and part of normal visitor questions such
    # as「五花海游玩建议」。Recognise the latter as knowledge lookup after realtime
    # checks, while leaving concrete maintenance/suggestion reports in feedback.
    qa_question_words = (
        "开放时间", "游玩建议", "怎么去", "怎么走", "如何去", "海拔", "多少钱",
        "票价", "门票", "预约规则", "购票", "实名预约", "哪里", "哪些", "多少", "注意什么", "介绍",
    )
    if _contains_any(message, qa_question_words) and not looks_like_service_complaint(message):
        return "qa"
    if _contains_any(message, _FEEDBACK_MARKERS) or looks_like_service_complaint(message):
        return "feedback"
    return "qa"

def keyword_route(message: str) -> dict[str, Any]:
    """The degraded router result, shaped like the model's answer."""
    intent = classify(message)
    return {
        "intent": intent,
        "confidence": 0.0,
        "reason": "keyword-fallback",
        "entities": [],
        "source": "keyword",
    }


def can_use_fast_keyword_route(message: str) -> bool:
    """Whether a short, unambiguous factual question can skip model classification.

    This is deliberately narrower than the keyword router itself. Complex planning,
    time-sensitive status and feedback keep model classification; ordinary factual QA
    has exactly the same dispatch target either way and need not pay one LLM round-trip.
    """
    text = str(message or "").strip()
    if not text or len(text) > 100 or keyword_route(text).get("intent") != "qa":
        return False
    complex_markers = (
        "路线", "怎么走", "怎么安排", "换乘", "老人", "孩子", "半天", "一天",
        "实时", "今天", "现在", "当前", "明天", "余票", "限流", "关闭",
        "比较", "区别", "推荐", "反馈", "投诉",
    )
    return not _contains_any(text, complex_markers)


def _normalise(payload: dict[str, Any], message: str) -> dict[str, Any] | None:
    raw_intent = str(payload.get("intent") or "").strip()
    if raw_intent not in INTENTS:
        return None
    try:
        confidence = float(payload.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0
    entities = payload.get("entities")
    if not isinstance(entities, list):
        entities = []
    intent = INTENT_ALIASES.get(raw_intent, raw_intent)
    # A model that says "ticket" is really answering from the knowledge base; keep the
    # raw label for observability but dispatch on the mapped intent.
    return {
        "intent": intent,
        "raw_intent": raw_intent,
        "confidence": max(0.0, min(1.0, confidence)),
        "reason": str(payload.get("reason") or "")[:120],
        "entities": [str(entity) for entity in entities][:8],
        "source": "model",
    }


async def classify_llm(message: str, *, client: Any | None = None) -> dict[str, Any]:
    """Model routing. Any failure returns the keyword fallback instead of raising."""
    fallback = keyword_route(message)
    if is_smalltalk(message):
        return fallback
    if not (LLM_MODE == "agent" and llm_configured()):
        return fallback
    from .llm import LLMUnavailable, get_client

    try:
        llm = client or get_client()
        payload = await llm.complete_json(
            [
                {"role": "system", "content": ROUTER_SYSTEM},
                {"role": "user", "content": message},
            ],
            temperature=0,
        )
    except (LLMUnavailable, Exception):
        return fallback
    result = _normalise(payload, message) or fallback
    # Suitability follow-ups are factual QA even if the router model overweights
    # “老人/儿童” and calls them a route recommendation.
    if _contains_any(message, ("适合", "是否适合", "能不能")) and result.get("intent") != "qa":
        return fallback
    return result


async def classify_with_cache(message: str, *, client: Any | None = None) -> dict[str, Any]:
    """Intent recognition with a cache: visitors repeat the same phrasings constantly."""
    if is_smalltalk(message):
        # Bypass stale cached QA classifications from older releases.
        return keyword_route(message)
    if _contains_any(message, ("适合", "是否适合", "能不能")):
        # A factual suitability follow-up must not reuse an old cached route intent.
        return keyword_route(message)
    if can_use_fast_keyword_route(message):
        result = keyword_route(message)
        result["source"] = "fast-keyword"
        return result
    key = f"intent:{PARK_ID}:{hashlib.sha256(message.strip().lower().encode()).hexdigest()}"
    cached = cache_get(key)
    if cached:
        cached["cache_hit"] = True
        return cached
    result = await classify_llm(message, client=client)
    cache_set(key, result, ttl=CACHE_TTL_SECONDS)
    return result


def intent_payload(route: dict[str, Any]) -> str:
    return json.dumps(route, ensure_ascii=False)
