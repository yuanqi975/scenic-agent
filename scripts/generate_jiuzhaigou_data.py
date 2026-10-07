"""Generate the Jiuzhaigou dataset from curated public facts.

Real facts (park, attractions, facilities, shuttle/walk segments, official FAQs,
park-level knowledge documents) live in ``data/jiuzhaigou/facts/*.json`` and are
traceable to public sources listed in ``facts/sources.json``.

Everything the project needs for demos and load tests but that no public source
provides - per-attraction FAQ pairs, evaluation questions, visitor feedback - is
generated here and explicitly labelled ``derived`` or ``derived_simulation`` so
it can never be mistaken for official Jiuzhaigou data.

Usage::

    python scripts/generate_jiuzhaigou_data.py
    python scripts/generate_jiuzhaigou_data.py --output data/jiuzhaigou --feedback-rows 3000
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

PARK_ID = "jiuzhaigou_scenic_area"
SEED = 20260917
NOW = "2026-09-30T00:00:00+00:00"
KNOWLEDGE_VERSION = "v2.0"
ROOT = Path(__file__).resolve().parents[1]

OFFICIAL = "official"
THIRD_PARTY = "third_party"
DERIVED = "derived"
SIMULATION = "derived_simulation"

DURATION_SOURCE_LABEL = {
    "official": "官方口径",
    "third_party": "第三方公开资料",
    "estimated": "按同沟谷同类景点估算",
}

PARK_QUESTIONS = {
    "park_001": "九寨沟是什么级别的景区？有哪些称号？",
    "park_002": "九寨沟在哪里？属于哪个省哪个县？",
    "park_003": "九寨沟面积多大？海拔有多高？",
    "park_004": "九寨沟为什么叫九寨沟？",
    "park_005": "九寨沟有哪几条沟？各有什么景点？",
    "park_006": "九寨沟的「六绝」和 108 个海子指什么？",
    "park_007": "九寨沟门票多少钱？旺季和淡季价格一样吗？",
    "park_008": "九寨沟有哪些门票优惠和免票政策？",
    "park_009": "九寨沟开放时间是几点？几点停止入园？",
    "park_010": "九寨沟一天限流多少人？约不上票怎么办？",
    "park_011": "九寨沟怎么买票？现场能买票吗？",
    "park_012": "九寨沟观光车怎么换乘？车票能用几次？",
    "park_013": "九寨沟一日游路线怎么安排？",
    "park_014": "九寨沟景区有哪些禁止事项？",
    "park_015": "九寨沟为什么不能住在沟内？",
    "park_016": "九寨沟冬季关闭哪些景点？会封山吗？",
    "park_017": "九寨沟高原反应该怎么预防？",
    "park_018": "九寨沟什么时候去最好？",
    "park_019": "从成都怎么去九寨沟？有哪些交通方式？",
    "park_020": "九寨沟有哪些服务设施？咨询电话是多少？",
}

FEEDBACK_TEMPLATES = {
    "评价": "{name}景色非常漂亮，湖水颜色层次丰富，拍照很出片，值得再来。",
    "建议": "建议在{name}附近增加休息座椅和导览标识，方便老人休息。",
    "设施问题": "{name}附近的{facility_type}使用不便，建议检查维护。",
    "路线问题": "前往{name}的栈道雨后比较湿滑，建议增加防滑提示和扶手。",
    "投诉": "节假日前往{name}时排队换乘时间较长，建议加强客流引导。",
}

FEEDBACK_TYPES = ["评价", "建议", "设施问题", "路线问题", "投诉"]

FEEDBACK_RATINGS = {"评价": 5, "建议": 4, "设施问题": 2, "路线问题": 3, "投诉": 1}


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _record(data_source: str, **values):
    return {"park_id": PARK_ID, "data_source": data_source, "created_at": NOW, **values}


def haversine_meters(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> int:
    """Straight-line distance in meters between two estimated coordinates."""
    radius = 6371000.0
    phi_a, phi_b = math.radians(lat_a), math.radians(lat_b)
    delta_phi = math.radians(lat_b - lat_a)
    delta_lambda = math.radians(lon_b - lon_a)
    h = math.sin(delta_phi / 2) ** 2 + math.cos(phi_a) * math.cos(phi_b) * math.sin(delta_lambda / 2) ** 2
    return int(round(2 * radius * math.asin(math.sqrt(h))))


def generate_dataset(output_dir: Path, facts_dir: Path | None = None, feedback_rows: int = 3000) -> dict:
    facts_dir = facts_dir or output_dir / "facts"
    park_facts = _read_json(facts_dir / "park.json")
    attraction_facts = _read_json(facts_dir / "attractions.json")["items"]
    facility_facts = _read_json(facts_dir / "facilities.json")["items"]
    transport = _read_json(facts_dir / "transport.json")
    faq_facts = _read_json(facts_dir / "faqs.json")["items"]
    knowledge_facts = _read_json(facts_dir / "knowledge.json")["items"]
    policies = _read_json(facts_dir / "policies.json")
    notices = _read_json(facts_dir / "notices.json")["items"]
    sources = _read_json(facts_dir / "sources.json")

    output_dir.mkdir(parents=True, exist_ok=True)

    park = {
        "park_id": PARK_ID,
        "data_source": OFFICIAL,
        "created_at": NOW,
        "updated_at": NOW,
        "knowledge_version": KNOWLEDGE_VERSION,
        **park_facts,
    }

    opening_hours = park["opening_hours"]
    ticket_note = f"九寨沟景点不单独售票，含于景区门票（旺季 {park['ticket_price']} 元 / 淡季 {park['ticket_price_off_season']} 元）。"
    notice_pool = policies["notice_pool"]

    # ---------------------------------------------------------------- attractions
    attractions = []
    for index, item in enumerate(attraction_facts, start=1):
        seasonal = item.get("seasonal_closure")
        elevation = item.get("elevation_meters")
        if (elevation or 0) >= 2800:
            notice = notice_pool["high_altitude"]
        elif item.get("notice_extra") and "栈道" in item["notice_extra"]:
            notice = notice_pool["boardwalk"]
        else:
            notice = notice_pool["default"]
        if seasonal:
            notice = f"{notice}{notice_pool['seasonal_closure']}"
        source_ids = list(dict.fromkeys(["src_02", "src_05", *item.get("source_ids", [])]))
        if item["visit_duration_source"] == "third_party":
            source_ids = list(dict.fromkeys([*source_ids, "src_24"]))
        if seasonal:
            source_ids = list(dict.fromkeys([*source_ids, "src_22", "src_23"]))
        attractions.append(_record(
            OFFICIAL,
            attraction_id=item["attraction_id"],
            name=item["name"],
            valley=item["valley"],
            category=item["category"],
            description=item["description"],
            highlights=item["highlights"],
            elevation_meters=elevation,
            elevation_note=item.get("elevation_note"),
            elevation_source="official" if elevation else None,
            opening_hours=opening_hours,
            opening_hours_source="park_level",
            ticket_price=0,
            ticket_note=ticket_note,
            visit_duration_minutes=item["visit_duration_minutes"],
            visit_duration_source=item["visit_duration_source"],
            difficulty=item["difficulty"],
            difficulty_source=DERIVED,
            suitable_for=item["suitable_for"],
            suitable_for_source=DERIVED,
            latitude=item["latitude"],
            longitude=item["longitude"],
            coordinate_source="estimated",
            in_standard_tour=item["in_standard_tour"],
            seasonal_closure=seasonal,
            notice=notice,
            notice_extra=item.get("notice_extra"),
            status="open",
            updated_at=NOW,
            source_ids=source_ids,
        ))

    # ---------------------------------------------------------------- facilities
    facilities = []
    for item in facility_facts:
        facilities.append(_record(
            OFFICIAL,
            facility_id=item["facility_id"],
            name=item["name"],
            facility_type=item["facility_type"],
            description=item["description"],
            latitude=item["latitude"],
            longitude=item["longitude"],
            coordinate_source="estimated" if item["latitude"] is not None else "not_published",
            accessibility=item["accessibility"],
            status=item["status"],
            inside_park=item["inside_park"],
            distance_to_gate_meters=item.get("distance_to_gate_meters"),
            phone=item.get("phone"),
            nearby_attraction_id=item.get("nearby_attraction_id"),
            notice=item.get("notice"),
            updated_at=NOW,
            source_ids=item.get("source_ids", []),
        ))

    locations = {item["attraction_id"]: item for item in attractions}
    locations.update({item["facility_id"]: item for item in facilities})

    # ---------------------------------------------------------------- routes
    routes = []
    for segment in transport["shuttle_segments"]:
        start, end = locations[segment["start_id"]], locations[segment["end_id"]]
        routes.append(_record(
            THIRD_PARTY,
            route_id=f"route_{len(routes) + 1:03d}",
            name=f"{start['name']}—{end['name']}（观光车）",
            start_id=segment["start_id"],
            end_id=segment["end_id"],
            start_name=start["name"],
            end_name=end["name"],
            distance_meters=segment["distance_meters"],
            distance_source=THIRD_PARTY,
            estimated_minutes=segment["estimated_minutes"],
            difficulty="轻松",
            difficulty_source=DERIVED,
            accessible=True,
            route_type="观光车",
            notice=transport["shuttle_notice"],
            updated_at=NOW,
            source_ids=segment["source_ids"],
        ))
    walk_speed = transport["walk_parameters"]["speed_meters_per_minute"]
    for segment in transport["walk_segments"]:
        start, end = locations[segment["start_id"]], locations[segment["end_id"]]
        distance = haversine_meters(start["latitude"], start["longitude"], end["latitude"], end["longitude"])
        official_minutes = segment.get("official_minutes") or 0
        minutes = max(transport["walk_parameters"]["min_minutes"], official_minutes, int(round(distance / walk_speed)))
        start_elevation, end_elevation = start.get("elevation_meters"), end.get("elevation_meters")
        # An unknown elevation must not look like a 2000 m climb.
        elevation_delta = abs(start_elevation - end_elevation) if start_elevation and end_elevation else 0
        if distance >= 3000 or elevation_delta >= 200:
            difficulty = "挑战"
        elif distance >= 800 or elevation_delta >= 60:
            difficulty = "中等"
        else:
            difficulty = "轻松"
        high_altitude = max(start_elevation or 0, end_elevation or 0) >= 2800
        routes.append(_record(
            DERIVED,
            route_id=f"route_{len(routes) + 1:03d}",
            name=f"{start['name']}—{end['name']}（步行栈道）",
            start_id=segment["start_id"],
            end_id=segment["end_id"],
            start_name=start["name"],
            end_name=end["name"],
            distance_meters=distance,
            distance_source=DERIVED,
            estimated_minutes=minutes,
            estimated_minutes_source=OFFICIAL if official_minutes else DERIVED,
            difficulty=difficulty,
            difficulty_source=DERIVED,
            accessible=False,
            route_type="步行栈道",
            notice=notice_pool["high_altitude"] if high_altitude else transport["walk_notice"],
            accessibility_notice=transport["walk_accessible_notice"],
            updated_at=NOW,
            source_ids=segment["source_ids"],
        ))

    # ---------------------------------------------------------------- documents
    documents = []
    attraction_docs: dict[str, str] = {}
    facility_docs: dict[str, str] = {}
    route_docs: dict[str, str] = {}
    faq_docs: dict[str, str] = {}
    park_docs: dict[str, str] = {}
    notice_docs: dict[str, str] = {}

    def add_document(source_type: str, source_id: str, content: str, data_source: str, source_ids: list[str], topic: str | None = None, metadata_extra: dict | None = None) -> str:
        document_id = f"doc_{len(documents) + 1:05d}"
        documents.append(_record(
            data_source,
            document_id=document_id,
            source_type=source_type,
            source_id=source_id,
            content=content,
            metadata={
                "source_type": source_type,
                "source_id": source_id,
                "topic": topic,
                "version": KNOWLEDGE_VERSION,
                "updated_at": NOW,
                "park_id": PARK_ID,
                "data_source": data_source,
                "source_ids": source_ids,
                **(metadata_extra or {}),
            },
            knowledge_version=KNOWLEDGE_VERSION,
            updated_at=NOW,
        ))
        return document_id

    for item in knowledge_facts:
        park_docs[item["knowledge_id"]] = add_document(
            "park", PARK_ID, f"【{item['topic']}】{item['content']}", OFFICIAL, item["source_ids"], item["topic"]
        )
    for item in notices:
        notice_docs[item["notice_id"]] = add_document(
            "park", item["notice_id"],
            f"【{item['title']}】{item['content']} 生效时间：{item['effective_from']}。失效时间：{item['effective_to'] or '未标注'}。状态：{item['status']}。",
            OFFICIAL, item["source_ids"], item["notice_type"],
            {"title": item["title"], "document_kind": "notice", "notice_type": item["notice_type"], "status": item["status"], "risk_level": item["risk_level"], "effective_from": item["effective_from"], "effective_to": item["effective_to"], "published_at": item["published_at"], "authority": item["authority"]}
        )
    for item in attractions:
        if item["elevation_meters"]:
            note = f"（{item['elevation_note']}）" if item.get("elevation_note") else ""
            elevation_text = f"海拔约 {item['elevation_meters']} 米{note}。"
        elif "未公布" in item["description"]:
            elevation_text = ""
        else:
            elevation_text = "官方未公布海拔。"
        overview = (
            f"景点：{item['name']}。所属沟谷：{item['valley']}。类型：{item['category']}。{elevation_text}"
            f"{item['description']} 主要看点：{item['highlights']}。"
            f"适合人群：{'、'.join(item['suitable_for'])}。建议游览时长约 {item['visit_duration_minutes']} 分钟"
            f"（{DURATION_SOURCE_LABEL[item['visit_duration_source']]}）。开放时间：{item['opening_hours']}。{item['ticket_note']}"
        )
        attraction_docs[item["attraction_id"]] = add_document(
            "attraction", item["attraction_id"], overview, item["data_source"], item["source_ids"], item["name"]
        )
        practical = f"游览提示：{item['name']}（{item['valley']} · {item['category']}）。{item['notice']}"
        if item.get("notice_extra"):
            practical += item["notice_extra"]
        add_document("attraction", item["attraction_id"], practical, item["data_source"], item["source_ids"], item["name"])
    for item in facilities:
        location_text = "位于景区内。" if item["inside_park"] else "位于景区外。"
        phone_text = f"联系电话：{item['phone']}。" if item["phone"] else ""
        notice_text = f"提示：{item['notice']}" if item["notice"] else ""
        content = (
            f"设施：{item['name']}。类型：{item['facility_type']}。{location_text}{item['description']} "
            f"当前状态：{item['status']}。{phone_text}{notice_text}"
        )
        facility_docs[item["facility_id"]] = add_document(
            "facility", item["facility_id"], content, item["data_source"], item["source_ids"], item["name"]
        )
    for item in routes:
        if item["route_type"] == "观光车":
            content = (
                f"观光车线路：{item['start_name']}至{item['end_name']}。距离约 {item['distance_meters']} 米"
                f"（{item['distance_meters'] / 1000:.0f} 公里），运行时间约 {item['estimated_minutes']} 分钟。{item['notice']}"
            )
        else:
            content = (
                f"步行栈道：{item['start_name']}至{item['end_name']}。距离约 {item['distance_meters']} 米"
                f"（按估算坐标计算），预计步行 {item['estimated_minutes']} 分钟，难度{item['difficulty']}。{item['notice']}"
            )
        route_docs[item["route_id"]] = add_document(
            "route", item["route_id"], content, item["data_source"], item["source_ids"], item["name"]
        )

    # ---------------------------------------------------------------- FAQs
    faqs = []
    for item in faq_facts:
        faqs.append(_record(
            OFFICIAL,
            faq_id=item["faq_id"],
            topic=item["topic"],
            question=item["question"],
            answer=item["answer"],
            keywords=item["keywords"],
            related_attraction_id=item.get("related_attraction_id"),
            official=True,
            updated_at=NOW,
            source_ids=item["source_ids"],
        ))
    for item in attractions:
        question = f"{item['name']}在哪里？有什么看点？"
        answer = (
            f"{item['name']}位于九寨沟{item['valley']}，类型为{item['category']}。{item['description']} "
            f"主要看点：{item['highlights']}。{item['ticket_note']}"
        )
        faqs.append(_record(
            DERIVED,
            faq_id=f"faq_{len(faqs) + 1:03d}",
            topic="景点介绍",
            question=question,
            answer=answer,
            keywords=[item["name"], item["valley"], item["category"]],
            related_attraction_id=item["attraction_id"],
            official=False,
            updated_at=NOW,
            source_ids=item["source_ids"],
        ))
        question = f"游览{item['name']}需要注意什么？"
        faqs.append(_record(
            DERIVED,
            faq_id=f"faq_{len(faqs) + 1:03d}",
            topic="注意事项",
            question=question,
            answer=f"{item['notice']}建议预留约 {item['visit_duration_minutes']} 分钟游览时间。",
            keywords=[item["name"], "注意事项", "游览提示"],
            related_attraction_id=item["attraction_id"],
            official=False,
            updated_at=NOW,
            source_ids=item["source_ids"],
        ))
    for item in faqs:
        faq_docs[item["faq_id"]] = add_document(
            "faq", item["faq_id"], f"问：{item['question']} 答：{item['answer']}", item["data_source"], item["source_ids"], item["topic"]
        )

    # ---------------------------------------------------------------- simulated feedback
    feedback_facilities = [item for item in facilities if item["inside_park"]]
    feedbacks, candidates = [], []
    for index in range(feedback_rows):
        attraction = attractions[index % len(attractions)]
        facility = feedback_facilities[(index * 3) % len(feedback_facilities)]
        feedback_type = FEEDBACK_TYPES[index % len(FEEDBACK_TYPES)]
        content = FEEDBACK_TEMPLATES[feedback_type].format(name=attraction["name"], facility_type=facility["facility_type"])
        status = ["pending_review", "pending_review", "approved", "rejected"][index % 4]
        rating = max(1, min(5, FEEDBACK_RATINGS[feedback_type] + (index // len(FEEDBACK_TYPES)) % 3 - 1))
        feedback_id = f"feedback_{index + 1:04d}"
        feedbacks.append(_record(
            SIMULATION,
            feedback_id=feedback_id,
            feedback_type=feedback_type,
            content=content,
            rating=rating,
            status=status,
            is_simulated=True,
            disclaimer="模拟游客反馈，用于演示与压测，不代表真实游客意见。",
            related_attraction_id=attraction["attraction_id"],
            related_facility_id=facility["facility_id"],
            submitted_at=NOW,
        ))
        candidates.append(_record(
            SIMULATION,
            candidate_id=f"candidate_{index + 1:04d}",
            feedback_id=feedback_id,
            candidate_fact=content,
            related_attraction_id=attraction["attraction_id"],
            related_facility_id=facility["facility_id"],
            status="pending_review" if status != "approved" else "approved",
            is_simulated=True,
            disclaimer="模拟候选知识，用于演示人工审核流程，不代表真实游客意见。",
            source_feedback_count=1,
            generated_at=NOW,
        ))

    # ---------------------------------------------------------------- time-sensitive notices
    notice_rows = []
    for item in notices:
        notice_rows.append(_record(
            OFFICIAL, notice_id=item["notice_id"], title=item["title"],
            notice_type=item["notice_type"], content=item["content"],
            published_at=item["published_at"], effective_from=item["effective_from"],
            effective_to=item["effective_to"], status=item["status"],
            risk_level=item["risk_level"], authority=item["authority"],
            source_ids=item["source_ids"], tags=item.get("tags", []), updated_at=NOW,
        ))

    # ---------------------------------------------------------------- evaluation set
    evaluations = []
    for item in attractions:
        evaluations.append(_record(
            DERIVED,
            question_id=f"eval_{len(evaluations) + 1:03d}",
            question=f"{item['name']}开放时间和游玩建议是什么？",
            question_type="attraction_query",
            expected_document_id=attraction_docs[item["attraction_id"]],
            answer_keywords=[item["name"], item["valley"]],
        ))
    for item in facilities:
        evaluations.append(_record(
            DERIVED,
            question_id=f"eval_{len(evaluations) + 1:03d}",
            question=f"{item['name']}在哪里？提供什么服务？",
            question_type="facility_query",
            expected_document_id=facility_docs[item["facility_id"]],
            answer_keywords=[item["name"], item["facility_type"]],
        ))
    for item in faq_facts:
        evaluations.append(_record(
            DERIVED,
            question_id=f"eval_{len(evaluations) + 1:03d}",
            question=item["question"],
            question_type="faq_query",
            expected_document_id=faq_docs[item["faq_id"]],
            answer_keywords=item["keywords"][:3],
        ))
    for item in knowledge_facts:
        evaluations.append(_record(
            DERIVED,
            question_id=f"eval_{len(evaluations) + 1:03d}",
            question=PARK_QUESTIONS[item["knowledge_id"]],
            question_type="park_query",
            expected_document_id=park_docs[item["knowledge_id"]],
            answer_keywords=item["tags"][:3],
        ))
    for item in evaluations:
        item.setdefault("evaluation_split", "baseline")
        item.setdefault("relevant_document_ids", [item["expected_document_id"]])
        item.setdefault("graded_relevance", {item["expected_document_id"]: 3})
        item.setdefault("risk_level", "low")
        item.setdefault("should_abstain", False)
    composite_evals = [
        _record(DERIVED, question_id="eval_complex_001", question="2026年7月泥石流后九寨沟现在是否全域开放？旺季门票和每日承载量是多少？", question_type="complex_current_status", intent="realtime", expected_document_id=notice_docs["notice_2026_full_reopen"], relevant_document_ids=[notice_docs["notice_2026_full_reopen"], notice_docs["notice_2026_capacity_and_entry"]], graded_relevance={notice_docs["notice_2026_full_reopen"]: 3, notice_docs["notice_2026_capacity_and_entry"]: 2}, required_facts=["全域恢复开放", "190元", "41000人次/天"], answer_keywords=["全域恢复开放", "190", "41000"], expected_channels=["sparse", "structured"], risk_level="high", should_abstain=False),
        _record(DERIVED, question_id="eval_complex_002", question="8月12日没有预约到九寨沟门票还能直接去吗？", question_type="capacity_and_reservation", intent="realtime", expected_document_id=notice_docs["notice_2026_aug12_sold_out"], relevant_document_ids=[notice_docs["notice_2026_aug12_sold_out"], notice_docs["notice_2026_capacity_and_entry"]], graded_relevance={notice_docs["notice_2026_aug12_sold_out"]: 3, notice_docs["notice_2026_capacity_and_entry"]: 2}, required_facts=["不要贸然前往", "候补", "改订其他日期"], answer_keywords=["不要贸然前往", "候补"], expected_channels=["sparse", "structured"], risk_level="high", should_abstain=False),
        _record(DERIVED, question_id="eval_complex_003", question="教师节当天教师带一位朋友去九寨沟，门票和观光车怎么收费？", question_type="temporary_discount", intent="policy", expected_document_id=notice_docs["notice_2026_teachers_day"], relevant_document_ids=[notice_docs["notice_2026_teachers_day"]], graded_relevance={notice_docs["notice_2026_teachers_day"]: 3}, required_facts=["2026年9月10日", "教师资格证", "观光车需单独购买"], answer_keywords=["教师资格证", "观光车", "免票"], expected_channels=["sparse", "structured"], risk_level="medium", should_abstain=False),
        _record(DERIVED, question_id="eval_complex_004", question="带老人和孩子游玩半天，偏好水景且尽量少走路，怎么安排？", question_type="route_planning", intent="recommendation", expected_document_id=park_docs["park_013"], relevant_document_ids=[park_docs["park_013"]], graded_relevance={park_docs["park_013"]: 2}, required_facts=["半天", "老人", "少走路"], answer_keywords=["路线", "观光车"], expected_channels=["entity", "structured"], risk_level="medium", should_abstain=False),
        _record(DERIVED, question_id="eval_complex_005", question="如果临时遇到暴雨或泥石流预警，景区问答系统应该直接给出确定的开放结论吗？", question_type="safety_abstention", intent="realtime", expected_document_id=notice_docs["notice_2026_full_reopen"], relevant_document_ids=[notice_docs["notice_2026_full_reopen"]], graded_relevance={notice_docs["notice_2026_full_reopen"]: 1}, required_facts=["以最新官方公告为准"], answer_keywords=["最新官方公告", "不能确认"], expected_channels=["structured"], risk_level="high", should_abstain=True),
    ]
    for item in composite_evals:
        item["evaluation_split"] = "complex"

    # ---------------------------------------------------------------- write everything
    _write_json(output_dir / "parks.json", park)
    _write_jsonl(output_dir / "attractions.jsonl", attractions)
    _write_jsonl(output_dir / "facilities.jsonl", facilities)
    _write_jsonl(output_dir / "routes.jsonl", routes)
    _write_jsonl(output_dir / "faqs.jsonl", faqs)
    _write_jsonl(output_dir / "notices.jsonl", notice_rows)
    _write_jsonl(output_dir / "feedbacks.jsonl", feedbacks)
    _write_jsonl(output_dir / "feedback_candidates.jsonl", candidates)
    _write_jsonl(output_dir / "rag_documents.jsonl", documents)
    _write_jsonl(output_dir / "evaluation_questions.jsonl", evaluations)
    _write_jsonl(output_dir / "evaluation_complex_questions.jsonl", composite_evals)
    _write_jsonl(output_dir / "evaluation_questions_v2.jsonl", evaluations + composite_evals)
    _write_json(output_dir / "sources.json", sources)

    manifest = {
        "park_id": PARK_ID,
        "park_name": park["name"],
        "generated_at": NOW,
        "knowledge_version": KNOWLEDGE_VERSION,
        "random_seed": SEED,
        "facts_dir": "facts/",
        "data_source_legend": {
            OFFICIAL: "来自官方来源（景区官网、政府网站或权威媒体转述的官方口径）",
            THIRD_PARTY: "来自可查证的第三方公开资料，非官方发布",
            DERIVED: "依据真实数据派生或推算，不是官方发布内容",
            SIMULATION: "程序模拟数据，仅用于演示与压测，不代表真实情况",
        },
        "counts": {
            "attractions": len(attractions),
            "facilities": len(facilities),
            "routes": len(routes),
            "routes_shuttle": sum(1 for item in routes if item["route_type"] == "观光车"),
            "routes_walk": sum(1 for item in routes if item["route_type"] == "步行栈道"),
            "faqs": len(faqs),
            "faqs_official": sum(1 for item in faqs if item["official"]),
            "feedbacks": len(feedbacks),
            "feedback_candidates": len(candidates),
            "documents": len(documents),
            "documents_by_source_type": {
                source_type: sum(1 for item in documents if item["source_type"] == source_type)
                for source_type in ("park", "attraction", "facility", "route", "faq")
            },
            "notices": len(notice_rows),
            "evaluation_base_questions": 121,
            "evaluation_complex_questions": len(composite_evals),
            "evaluation_questions": len(evaluations),
            "sources": len(sources["items"]),
        },
        "coordinate_policy": (
            "官方从未公布景点、设施与景区四至的经纬度。本数据集所有 latitude/longitude 均为按沟谷走向与官方海拔"
            "推算的估算值（精度约 ±0.01°，约 ±1 km），coordinate_source 标记为 estimated，不可用于测绘或导航。"
        ),
        "content_policy": (
            "景点、设施、票价、开放时间、优惠政策、游览规则、FAQ、公告与景区级知识文档均来自公开可查证来源；"
            "景点级派生问答、步行栈道距离与时长、难度与适宜人群、评测题、游客反馈均为程序生成并已标记。"
        ),
        "known_gaps": [
            "官方未公布任何景点与设施的经纬度，坐标为估算值。",
            "官方未公布景区几何中心坐标与精确四至边界。",
            "官方未公布卫生间点位清单、行李寄存点位与收费标准、轮椅与婴儿车租用费用。",
            "官方未公布观光车完整站点名单，本数据集仅收录公开可检索到的站点。",
            "官方未公布秋季彩林精确起止日期，口径为「十月中下旬」。",
            "景区救护电话存在 0837-7738818 与 0837-7739309 两个公开口径。",
            "官网 FAQ 页仍保留旺季门票 220 元的旧口径，本数据集采用现行 190 元。",
            "临时公告按 effective_from/effective_to 管理，过期公告保留用于审计但不得直接作为当前状态。",
        ],
    }
    _write_json(output_dir / "dataset_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the Jiuzhaigou dataset from curated public facts.")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "jiuzhaigou")
    parser.add_argument("--facts", type=Path, default=None, help="curated facts directory (default: <output>/facts)")
    parser.add_argument("--feedback-rows", type=int, default=3000)
    args = parser.parse_args()
    result = generate_dataset(args.output, args.facts, args.feedback_rows)
    print(f"Jiuzhaigou data generated in {args.output}")
    for key, value in result["counts"].items():
        print(f"  {key}: {value}")
