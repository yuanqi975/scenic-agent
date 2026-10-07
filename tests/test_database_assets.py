from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_schema_declares_pgvector_and_idempotent_business_keys():
    sql = (ROOT / "sql" / "init.sql").read_text(encoding="utf-8").lower()
    assert "create extension if not exists vector" in sql
    assert "embedding halfvec(2560)" in sql
    assert "embedding halfvec_cosine_ops" in sql
    assert "unique (park_id, attraction_id)" in sql
    assert "unique (park_id, document_id)" in sql


def test_import_and_embedding_scripts_expose_required_cli_options():
    importer = (ROOT / "scripts" / "import_to_postgres.py").read_text(encoding="utf-8")
    embedder = (ROOT / "scripts" / "build_embeddings.py").read_text(encoding="utf-8")
    assert '"--dsn"' in importer
    assert "on_conflict_do_nothing" in importer
    assert '"--input"' in embedder
    assert "EMBEDDING_API_KEY" in embedder
    assert "max_retries" in embedder
