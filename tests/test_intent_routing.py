"""Intent routing regressions.

The router is the *degraded* path (``LLM_MODE=fallback``) but it is also the last-resort
fallback whenever the model answers something unusable, so a misroute here means a
visitor's complaint is answered as a knowledge-base question and never reaches human
review.

The four original families keep their documented priorities; the additions are
adversarial - each case was checked for false positives as well as false negatives,
because "answer every message with feedback_agent" would pass a one-sided test.
"""

from __future__ import annotations

import pytest

from app.services.intent import classify, looks_like_service_complaint


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        # ---- the frozen contract: these existed before and must not drift
        ("一日游路线怎么安排", "recommendation"),
        ("观光车怎么换乘", "recommendation"),
        ("今天限流多少人，会不会轮休保育", "realtime"),
        ("建议增加休息座椅", "feedback"),
        ("五花海海拔多少", "qa"),
        # ---- reported facility problems are feedback even without the word 反馈
        ("游客中心的厕所不能使用", "feedback"),
        ("景区观光车坏了", "feedback"),
        ("厕所太脏了，没人打扫", "feedback"),
        ("栈道故障，无法使用", "feedback"),
        ("停车场标识不干净", "feedback"),
        # ---- ...but a plain question about the same facility is not a complaint
        ("厕所在哪里", "qa"),
        ("请问诺日朗服务中心有餐厅吗", "qa"),
        ("五花海海拔多少米", "qa"),
        ("五花海开放时间和游玩建议", "qa"),
        # ---- realtime still wins when an explicit realtime word is present
        ("今天天气怎么样", "realtime"),
        ("景区现在客流大吗", "realtime"),
    ],
)
def test_classify_routes_each_message_to_the_right_agent(message: str, expected: str) -> None:
    assert classify(message) == expected


def test_service_complaint_detection_requires_both_halves():
    """A problem word alone, or an object alone, must not be enough.

    This is the guard that keeps "今天下雨吗" out of the feedback queue while still
    catching "厕所不能用".
    """
    assert looks_like_service_complaint("厕所不能使用") is True
    assert looks_like_service_complaint("今天下雨吗") is False
    assert looks_like_service_complaint("厕所在哪里") is False
    assert looks_like_service_complaint("不能使用") is False


def test_the_rule_brain_actually_dispatches_the_complaint_to_feedback_agent():
    """Routing is only useful if the supervisor acts on it."""
    import asyncio

    from app.services.brains import RuleBrain

    plan = asyncio.run(RuleBrain().plan("游客中心的厕所不能使用", [], ["feedback_agent", "knowledge_agent"]))
    assert [task["agent"] for task in plan.tasks] == ["feedback_agent"]
