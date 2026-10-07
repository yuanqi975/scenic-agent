import asyncio


def test_evaluation_accepts_legacy_single_document_field(monkeypatch):
    from app.services import evaluation

    class FakeService:
        async def search(self, request):
            return {"backend": "test", "items": [{"document_id": "doc-1", "retrieval": "entity"}]}

    monkeypatch.setattr(evaluation, "get_rag_service", lambda: FakeService())
    result = asyncio.run(evaluation.evaluate_questions([{"question_id": "q1", "question": "x", "expected_document_id": "doc-1", "answer_keywords": ["x"]}]))
    assert result["questions"] == 1
    assert result["invalid_questions"] == 0
    assert result["recall_at_k"] == 1
    assert result["mrr_at_k"] == 1


def test_production_config_rejects_development_defaults(monkeypatch):
    from app.core import config

    monkeypatch.setattr(config, "ENVIRONMENT", "production")
    monkeypatch.setattr(config, "JWT_SECRET", "change-this-secret-in-production")
    monkeypatch.setattr(config, "ADMIN_PASSWORD", "admin123456")
    monkeypatch.setattr(config, "LLM_MODE", "fallback")
    errors = config.production_config_errors()
    assert any("JWT_SECRET" in error for error in errors)
    assert any("ADMIN_PASSWORD" in error for error in errors)
    assert any("LLM_MODE" in error for error in errors)


def test_login_failure_limiter_blocks_repeated_attempts(monkeypatch):
    from app.core import security
    from fastapi import HTTPException

    security._failed_logins.clear()
    monkeypatch.setattr(security, "ADMIN_LOGIN_RATE_LIMIT", 2)
    identity = "127.0.0.1:user@example.com"
    security.record_login_failure(identity)
    security.record_login_failure(identity)
    try:
        security.check_login_rate_limit(identity)
    except HTTPException as exc:
        assert exc.status_code == 429
    else:
        raise AssertionError("expected login rate limit")


def test_rag_result_exposes_grounding_state():
    from app.services.rag import HybridRagService

    result = HybridRagService._result([], backend="test")
    assert result["grounding_status"] == "insufficient"
    assert result["abstention_required"] is True
