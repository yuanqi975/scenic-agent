from pathlib import Path


def test_milvus_configuration_has_safe_local_defaults():
    from app.core import config

    assert config.RAG_BACKEND in {"postgres", "shadow", "milvus"}
    assert config.MILVUS_URI.startswith("http")
    assert config.MILVUS_COLLECTION


def test_compose_declares_milvus_stack():
    compose = Path("docker-compose.yml").read_text(encoding="utf-8")
    for service in ("etcd:", "minio:", "milvus:"):
        assert service in compose
