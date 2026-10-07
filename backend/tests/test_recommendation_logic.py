"""Database-free contract tests for the Jiuzhaigou recommendation and routing logic."""

import asyncio
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, "backend")

from app import main
from app.services.route_algo import REMOTE_UPPER_RIZE_IDS, greedy_itinerary


def attraction(attraction_id, name, category, minutes, groups, in_standard_tour=True, status="open"):
    return {
        "attraction_id": attraction_id,
        "name": name,
        "valley": "日则沟",
        "category": category,
        "visit_duration_minutes": minutes,
        "suitable_for": groups,
        "in_standard_tour": in_standard_tour,
        "status": status,
        "ticket_price": 0,
        "ticket_note": "含于景区门票",
        "opening_hours": "旺季 08:00-18:00",
    }


CATALOG = [
    attraction("attr_021", "五花海", "湖泊/海子", 40, ["普通游客", "摄影爱好者", "亲子家庭"]),
    attraction("attr_014", "诺日朗瀑布", "瀑布", 40, ["普通游客", "摄影爱好者"]),
    attraction("attr_028", "原始森林", "森林", 35, ["户外爱好者"], status="closed"),
    attraction("attr_039", "扎依扎嘎神山", "山峰/神山", 60, ["户外爱好者"], in_standard_tour=False),
]


FACILITIES = [
    {
        "facility_id": "facility_010",
        "name": "诺日朗中心站",
        "facility_type": "观光车站",
        "status": "available",
    },
    {
        "facility_id": "facility_006",
        "name": "诺日朗服务中心（诺日朗餐厅）",
        "facility_type": "餐饮点",
        "status": "available",
    },
]


def test_entity_matching_finds_jiuzhaigou_pois_in_a_chinese_question(monkeypatch):
    """Chinese has no word delimiters, so retrieval must match known POI names itself."""
    monkeypatch.setattr(
        main,
        "payload_rows",
        lambda table, *args, **kwargs: CATALOG if table == "attractions" else FACILITIES,
    )
    assert main.mentioned_entity_ids("五花海怎么去？") == ["attr_021"]
    assert main.mentioned_entity_ids("诺日朗瀑布和五花海怎么安排") == ["attr_021", "attr_014"]
    assert main.mentioned_entity_ids("诺日朗中心站怎么换乘") == ["facility_010"]
    assert main.mentioned_entity_ids("九寨沟门票多少钱") == []


def test_recommendation_skips_closed_and_off_itinerary_spots(monkeypatch):
    monkeypatch.setattr(main, "payload_rows", lambda *args, **kwargs: CATALOG)
    result = main.build_recommendation(main.RecommendationRequest(duration_minutes=240, groups=["普通游客"]))
    names = [item["name"] for item in result["attractions"]]
    assert names == ["五花海", "诺日朗瀑布"]
    assert result["total_minutes"] == 80
    assert "九寨沟" in result["reason"]
    assert [citation["source_id"] for citation in result["citations"]] == ["attr_021", "attr_014"]


def test_classify_recognises_jiuzhaigou_route_and_realtime_intent():
    assert main.classify("一日游路线怎么安排") == "recommendation"
    assert main.classify("观光车怎么换乘") == "recommendation"
    assert main.classify("今天限流多少人，会不会轮休保育") == "realtime"
    assert main.classify("建议增加休息座椅") == "feedback"
    assert main.classify("五花海海拔多少") == "qa"


def test_fallback_answer_uses_the_jiuzhaigou_knowledge_base_wording():
    answer = asyncio.run(main.generate_answer("未知问题", []))
    assert "九寨沟" in answer
    assert "游客服务中心" in answer


def test_map_routes_cover_four_and_eight_hours_without_remote_upper_rize_for_older_visitors():
    dataset = Path(__file__).resolve().parents[2] / "data" / "jiuzhaigou" / "attractions.jsonl"
    with dataset.open(encoding="utf-8") as source:
        attractions = [json.loads(line) for line in source]

    for groups in ([], ["老年游客"]):
        half_day = greedy_itinerary(
            attractions,
            duration_minutes=240,
            groups=groups,
            difficulty="轻松",
            today=date(2026, 10, 6),
        )
        assert half_day["total_minutes"] == 240
        assert not ({item["attraction_id"] for item in half_day["attractions"]} & REMOTE_UPPER_RIZE_IDS)

    full_day_older = greedy_itinerary(
        attractions,
        duration_minutes=480,
        groups=["老年游客"],
        difficulty="轻松",
        today=date(2026, 10, 6),
    )
    assert full_day_older["total_minutes"] >= 420
    assert not ({item["attraction_id"] for item in full_day_older["attractions"]} & REMOTE_UPPER_RIZE_IDS)


def test_older_visitor_recommendation_explains_why_primitive_forest_is_skipped():
    dataset = Path(__file__).resolve().parents[2] / "data" / "jiuzhaigou" / "attractions.jsonl"
    with dataset.open(encoding="utf-8") as source:
        attractions = [json.loads(line) for line in source]

    result = main.build_recommendation(
        duration_minutes=480,
        groups=["老年游客"],
        attractions=attractions,
    )
    names = {item["name"] for item in result["attractions"]}
    assert result["total_minutes"] >= 420
    assert "原始森林" not in names
    assert "原始森林到下方景点" in result["rest_notes"]
    assert "通常不建议老人专程前往" in result["rest_notes"]
