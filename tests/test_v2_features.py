from __future__ import annotations

import json
from pathlib import Path


def test_query_planner_preserves_original_and_expands_compound_questions():
    from app.services.query_planner import plan_queries

    planned = plan_queries("五花海开放时间和老人游玩建议")
    assert planned[0]["text"] == "五花海开放时间和老人游玩建议"
    assert len(planned) >= 2
    assert all(item["text"] for item in planned)


def test_v2_dataset_has_separate_complex_split_and_notice_metadata():
    root = Path(__file__).resolve().parents[1]
    rows = [json.loads(line) for line in (root / "data/jiuzhaigou/evaluation_questions_v2.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 126
    assert {row["evaluation_split"] for row in rows} == {"baseline", "complex"}
    assert any(row.get("should_abstain") for row in rows)
    notices = [json.loads(line) for line in (root / "data/jiuzhaigou/notices.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(notices) == 4
    assert all("effective_from" in row and row["source_ids"] for row in notices)
