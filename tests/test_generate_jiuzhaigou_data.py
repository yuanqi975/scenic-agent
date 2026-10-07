"""Contract tests for the Jiuzhaigou dataset generator."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from generate_jiuzhaigou_data import PARK_ID, generate_dataset

FACTS_DIR = Path(__file__).resolve().parents[1] / "data" / "jiuzhaigou" / "facts"
ALLOWED_DATA_SOURCES = {"official", "third_party", "derived", "derived_simulation"}


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_dataset(tmp_path: Path, feedback_rows: int = 50) -> dict:
    generate_dataset(tmp_path, FACTS_DIR, feedback_rows=feedback_rows)
    return {
        "park": json.loads((tmp_path / "parks.json").read_text(encoding="utf-8")),
        "attractions": read_jsonl(tmp_path / "attractions.jsonl"),
        "facilities": read_jsonl(tmp_path / "facilities.jsonl"),
        "routes": read_jsonl(tmp_path / "routes.jsonl"),
        "faqs": read_jsonl(tmp_path / "faqs.jsonl"),
        "feedbacks": read_jsonl(tmp_path / "feedbacks.jsonl"),
        "documents": read_jsonl(tmp_path / "rag_documents.jsonl"),
        "candidates": read_jsonl(tmp_path / "feedback_candidates.jsonl"),
        "evaluations": read_jsonl(tmp_path / "evaluation_questions.jsonl"),
        "manifest": json.loads((tmp_path / "dataset_manifest.json").read_text(encoding="utf-8")),
        "sources": json.loads((tmp_path / "sources.json").read_text(encoding="utf-8")),
    }


def test_generator_creates_consistent_jiuzhaigou_dataset(tmp_path):
    data = read_dataset(tmp_path)

    assert data["park"]["park_id"] == PARK_ID == "jiuzhaigou_scenic_area"
    assert data["park"]["name"] == "九寨沟风景名胜区"
    assert data["park"]["ticket_price"] == 190
    assert data["park"]["ticket_price_off_season"] == 80
    assert data["park"]["shuttle_bus_price"] == 90
    assert data["park"]["reservation_required"] is True

    assert len(data["attractions"]) == 40
    assert len(data["facilities"]) == 37
    assert len(data["routes"]) == 47
    assert sum(1 for route in data["routes"] if route["route_type"] == "观光车") == 9
    assert sum(1 for route in data["routes"] if route["route_type"] == "步行栈道") == 38
    assert len(data["faqs"]) == 104
    assert sum(1 for faq in data["faqs"] if faq["official"]) == 24
    assert len(data["feedbacks"]) == 50
    assert len(data["candidates"]) == 50
    assert len(data["evaluations"]) == 121
    assert len(data["sources"]["items"]) == 30

    counts = data["manifest"]["counts"]
    assert counts["attractions"] == len(data["attractions"])
    assert counts["documents"] == len(data["documents"])
    assert sum(counts["documents_by_source_type"].values()) == counts["documents"]


def test_every_real_record_is_traceable_and_every_derived_record_is_labelled(tmp_path):
    data = read_dataset(tmp_path)
    records = data["attractions"] + data["facilities"] + data["routes"] + data["faqs"] + data["documents"]
    assert {record["data_source"] for record in records} <= ALLOWED_DATA_SOURCES
    assert all(record["park_id"] == PARK_ID for record in records)

    # Real, publicly sourced content must stay traceable to a source id.
    for record in data["attractions"] + data["facilities"]:
        assert record["data_source"] == "official"
        assert record["source_ids"], record["name"]
    for faq in data["faqs"]:
        assert faq["source_ids"], faq["question"]

    # Simulated visitor feedback must never masquerade as real feedback.
    for feedback in data["feedbacks"]:
        assert feedback["data_source"] == "derived_simulation"
        assert feedback["is_simulated"] is True
        assert feedback["disclaimer"]
    for candidate in data["candidates"]:
        assert candidate["data_source"] == "derived_simulation"
        assert candidate["is_simulated"] is True


def test_coordinates_stay_inside_the_estimated_park_boundary(tmp_path):
    data = read_dataset(tmp_path)
    boundary = data["park"]["boundary"]
    latitude_range, longitude_range = boundary["latitude"], boundary["longitude"]
    located = [item for item in data["attractions"] + data["facilities"] if item["latitude"] is not None]
    # 40 attractions plus the 33 facilities whose location is publicly known.
    assert len(located) == 73
    for item in located:
        assert latitude_range[0] <= item["latitude"] <= latitude_range[1], item["name"]
        assert longitude_range[0] <= item["longitude"] <= longitude_range[1], item["name"]
        assert item["coordinate_source"] == "estimated", item["name"]

    # Facilities without a public location must not get an invented coordinate.
    for item in data["facilities"]:
        if item["latitude"] is None:
            assert item["coordinate_source"] == "not_published", item["name"]


def test_routes_and_evaluations_reference_existing_records(tmp_path):
    data = read_dataset(tmp_path)
    known_ids = {item["attraction_id"] for item in data["attractions"]} | {
        item["facility_id"] for item in data["facilities"]
    }
    for route in data["routes"]:
        assert route["start_id"] in known_ids, route["route_id"]
        assert route["end_id"] in known_ids, route["route_id"]
        assert route["distance_meters"] > 0
        assert route["estimated_minutes"] > 0

    document_ids = {item["document_id"] for item in data["documents"]}
    for question in data["evaluations"]:
        assert question["expected_document_id"] in document_ids, question["question"]
        assert question["answer_keywords"]


def test_shuttle_segments_keep_the_published_distances(tmp_path):
    data = read_dataset(tmp_path)
    shuttle = [route for route in data["routes"] if route["route_type"] == "观光车"]
    names = {(route["start_name"], route["end_name"]): route for route in shuttle}
    assert names[("沟口站", "长海站")]["distance_meters"] == 32000
    assert names[("沟口站", "长海站")]["estimated_minutes"] == 38
    assert names[("诺日朗中心站", "原始森林站")]["distance_meters"] == 17000
    assert all(route["data_source"] == "third_party" for route in shuttle)


def test_missing_public_values_never_leak_placeholders_or_fake_severity(tmp_path):
    """Unknown elevations must not print 'None' nor make a short walk look like a climb."""
    data = read_dataset(tmp_path)
    for record in data["attractions"] + data["documents"]:
        for value in record.values():
            if isinstance(value, str):
                assert "None" not in value, record.get("name") or record.get("document_id")

    by_id = {item["attraction_id"]: item for item in data["attractions"]}
    for route in data["routes"]:
        if route["route_type"] != "步行栈道":
            continue
        start, end = by_id.get(route["start_id"]), by_id.get(route["end_id"])
        if start is None or end is None:
            continue
        if start["elevation_meters"] is None or end["elevation_meters"] is None:
            # A missing elevation may only relax the rating, never harden it.
            assert route["difficulty"] != "挑战" or route["distance_meters"] >= 3000, route["name"]
        assert route["distance_meters"] > 0 and route["estimated_minutes"] >= 5


def test_generated_rows_satisfy_the_postgres_importer_contract(tmp_path):
    """scripts/import_to_postgres.py reads park_id plus one business key per table."""
    import_keys = {
        "attractions.jsonl": "attraction_id",
        "facilities.jsonl": "facility_id",
        "routes.jsonl": "route_id",
        "faqs.jsonl": "faq_id",
        "feedbacks.jsonl": "feedback_id",
        "feedback_candidates.jsonl": "candidate_id",
        "evaluation_questions.jsonl": "question_id",
    }
    generate_dataset(tmp_path, FACTS_DIR, feedback_rows=20)
    park = json.loads((tmp_path / "parks.json").read_text(encoding="utf-8"))
    assert park["park_id"] == PARK_ID
    assert park["created_at"]
    for filename, key in import_keys.items():
        for record in read_jsonl(tmp_path / filename):
            assert record["park_id"] == PARK_ID
            assert record[key], f"{filename} row without {key}"

    document_columns = {
        "park_id", "document_id", "source_type", "source_id",
        "content", "metadata", "knowledge_version", "updated_at",
    }
    for record in read_jsonl(tmp_path / "rag_documents.jsonl"):
        assert document_columns <= set(record)
        assert record["source_type"] in {"park", "attraction", "facility", "route", "faq"}
        assert record["content"].strip()
        assert record["metadata"]["data_source"] == record["data_source"]


def test_dataset_contains_no_leftover_fictional_park_branding(tmp_path):
    generate_dataset(tmp_path, FACTS_DIR, feedback_rows=10)
    for path in sorted(tmp_path.glob("*")):
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            for forbidden in ("云岭", "yunling", "YUNLING", "云州", "江南省"):
                assert forbidden not in text, f"{forbidden} found in {path.name}"


def test_deployment_identity_switched_to_jiuzhaigou():
    """The rebrand must not leave the old park id in any identity-defining file."""
    root = Path(__file__).resolve().parents[1]
    files = [
        root / "backend" / "app" / "main.py",
        root / "backend" / "app" / "__init__.py",
        root / "frontend" / "src" / "App.vue",
        root / "frontend" / "package.json",
        root / ".env.example",
        root / "docker-compose.yml",
        root / "README.md",
        root / "README_DATA.md",
        root / "loadtests" / "locustfile.py",
        root / "scripts" / "import_to_postgres.py",
    ]
    for path in files:
        text = path.read_text(encoding="utf-8")
        for forbidden in ("yunling", "YUNLING", "云岭", "data/synthetic", "generate_synthetic_data"):
            assert forbidden not in text, f"{forbidden} found in {path.relative_to(root)}"
    assert 'PARK_ID = os.getenv("PARK_ID", "jiuzhaigou_scenic_area")' in (
        root / "backend" / "app" / "main.py"
    ).read_text(encoding="utf-8")
