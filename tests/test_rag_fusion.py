"""RAG fusion regressions: freshness weighting and conflict detection.

Both behaviours exist because the knowledge base repeats the same fact across many
document ids, which is exactly the condition under which plain "smallest vector
distance wins" picks an arbitrary copy - or picks a stale one.

The conflict detector has a sharper failure mode than "misses a conflict": a heuristic
that shouts about every chunk containing two prices would flag almost every official
page and destroy trust in the signal. The false-positive cases below are therefore as
important as the true-positive ones.
"""

from __future__ import annotations

from datetime import date

from app.services.rag import (
    AUTHORITY_WEIGHT,
    RagHit,
    detect_conflicts,
    freshness_weight,
    fuse_hits,
)

TODAY = date(2026, 9, 19)


def _hit(content: str, **kwargs) -> RagHit:
    kwargs.setdefault("chunk_id", content[:12])
    kwargs.setdefault("score", 1.0)
    kwargs.setdefault("channel", "dense")
    return RagHit(content=content, **kwargs)


# --------------------------------------------------------------------------- freshness
def test_freshness_weight_is_bounded_and_decays():
    """Bounded so it reorders near-ties without overturning a better textual match."""
    fresh = _hit("x", updated_at="2026-09-01")
    stale = _hit("x", updated_at="2018-01-01")
    undated = _hit("x")

    assert freshness_weight(fresh, today=TODAY) > freshness_weight(stale, today=TODAY)
    assert freshness_weight(undated, today=TODAY) == 1.0
    for weight in (freshness_weight(fresh, today=TODAY), freshness_weight(stale, today=TODAY)):
        assert 0.9 <= weight <= 1.1


def test_a_current_official_page_beats_a_stale_one_at_equal_relevance():
    """Same authority, same rank: freshness has to be the tie-breaker."""
    channels = {
        "dense": [
            _hit("旺季门票 190 元", source_id="park", document_id="old", authority="official", updated_at="2020-01-01"),
            _hit("旺季门票 190 元", source_id="park", document_id="new", authority="official", updated_at="2026-03-24"),
        ]
    }
    fused = fuse_hits(channels, top_k=2, today=TODAY)
    assert [item.document_id for item in fused] == ["new", "old"]


def test_authority_still_outranks_freshness_inside_the_bound():
    """A fresh community post must not outrank an official page."""
    channels = {
        "dense": [
            _hit("门票信息", source_id="park", document_id="official", authority="official", updated_at="2025-01-01"),
            _hit("门票信息", source_id="park", document_id="community", authority="community", updated_at="2026-09-01"),
        ]
    }
    fused = fuse_hits(channels, top_k=2, today=TODAY)
    assert [item.document_id for item in fused] == ["official", "community"]
    assert AUTHORITY_WEIGHT["official"] > AUTHORITY_WEIGHT["community"]


def test_fusion_merges_the_same_document_seen_by_two_channels():
    channels = {
        "dense": [_hit("五花海海拔 2472 米", source_id="attr_001", document_id="d1")],
        "sparse": [_hit("五花海海拔 2472 米", source_id="attr_001", document_id="d1")],
    }
    fused = fuse_hits(channels, top_k=5, today=TODAY)
    assert len(fused) == 1
    assert fused[0].channels == ["dense", "sparse"]


# --------------------------------------------------------------------------- conflicts
def test_two_different_facts_in_one_sentence_are_not_a_conflict():
    """「门票 190 元、观光车票 90 元」 is one sentence and two unrelated facts."""
    hits = [
        _hit(
            "旺季门票 190 元、淡季 80 元，观光车票 90 元。",
            source_id="jiuzhaigou_scenic_area",
            document_id="d1",
        )
    ]
    assert detect_conflicts(hits) == []


def test_the_same_label_with_different_values_across_documents_is_a_conflict():
    hits = [
        _hit("旺季门票 190 元。", source_id="park", document_id="d1", updated_at="2024-01-01"),
        _hit("旺季门票 220 元。", source_id="park", document_id="d2", updated_at="2026-03-24"),
    ]
    conflicts = detect_conflicts(hits)
    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert conflict["severity"] == "conflict"
    assert conflict["label"] == "门票"
    assert conflict["values"] == ["190", "220"]
    assert {source["value"] for source in conflict["sources"]} == {"190", "220"}


def test_a_conditional_price_in_one_document_is_flagged_as_ambiguous_not_conflicting():
    """A peak/off-peak pair is conditional, so it must not read as contradictory."""
    hits = [
        _hit("旺季门票 190 元。", source_id="park", document_id="d1"),
        _hit("淡季门票 80 元。", source_id="park", document_id="d1"),
    ]
    conflicts = detect_conflicts(hits)
    assert [item["severity"] for item in conflicts] == ["ambiguous"]
    assert "适用条件" in conflicts[0]["advice"]


def test_different_subjects_do_not_conflict():
    """Two attractions legitimately have different elevations."""
    hits = [
        _hit("五花海海拔 2472 米。", source_id="attr_001", document_id="d1"),
        _hit("原始森林海拔 3060 米。", source_id="attr_028", document_id="d2"),
    ]
    assert detect_conflicts(hits) == []


def test_elevation_disagreement_on_one_subject_is_a_conflict():
    hits = [
        _hit("五花海海拔 2472 米。", source_id="attr_001", document_id="d1"),
        _hit("五花海海拔 2500 米。", source_id="attr_001", document_id="d2"),
    ]
    conflicts = detect_conflicts(hits)
    assert [item["fact"] for item in conflicts] == ["elevation"]
    assert conflicts[0]["values"] == ["2472", "2500"]


def test_conflicts_lower_reported_confidence_but_ambiguity_barely_does():
    """Confidence feeds the answer's hedging, so it must react to real disagreement."""
    from app.services.rag import HybridRagService

    clean = HybridRagService._result(
        {"dense": [_hit("旺季门票 190 元。", source_id="park", document_id="d1")]},
        backend="milvus",
    )
    conflicting = HybridRagService._result(
        {
            "dense": [
                _hit("旺季门票 190 元。", source_id="park", document_id="d1"),
                _hit("旺季门票 220 元。", source_id="park", document_id="d2"),
            ]
        },
        backend="milvus",
    )
    assert conflicting["confidence"] < clean["confidence"]
    assert conflicting["conflicts"]
    assert clean["conflicts"] == []
