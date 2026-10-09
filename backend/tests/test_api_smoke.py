import sys

sys.path.insert(0, "backend")
from fastapi.testclient import TestClient
from app.main import app


def test_public_endpoints_and_admin_authentication():
    with TestClient(app) as client:
        assert client.get("/api/v1/health").status_code == 200
        attractions = client.get("/api/v1/attractions?limit=3")
        assert attractions.status_code == 200
        assert len(attractions.json()["items"]) == 3
        assert client.get("/api/v1/admin/dashboard").status_code == 401
        login = client.post("/api/v1/admin/auth/login", json={"email": "admin@jiuzhaigou.local", "password": "admin123456"})
        assert login.status_code == 200
        response = client.post("/api/v1/recommendations", json={"duration_minutes": 240, "groups": ["老年游客"]})
        assert response.status_code == 200
        assert response.json()["attractions"]


def test_admin_logout_revokes_the_session_token():
    with TestClient(app) as client:
        token = client.post(
            "/api/v1/admin/auth/login",
            json={"email": "admin@jiuzhaigou.local", "password": "admin123456"},
        ).json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        assert client.get("/api/v1/admin/dashboard", headers=headers).status_code == 200
        assert client.post("/api/v1/admin/auth/logout", headers=headers).status_code == 200
        assert client.get("/api/v1/admin/dashboard", headers=headers).status_code == 401


def test_streaming_chat_uses_json_encoded_sse_payloads():
    with TestClient(app) as client:
        response = client.post("/api/v1/chat/stream", json={"message": "五花海开放时间和游玩建议"})
        assert response.status_code == 200
        assert 'event: status\ndata: "正在识别问题"' in response.text
        assert "event: result" in response.text


def test_feedback_creates_isolated_candidate_and_approved_candidate_publishes_document():
    """Removing candidate creation or any publish side effect must break this flow."""
    with TestClient(app) as client:
        submitted = client.post(
            "/api/v1/feedbacks",
            json={"content": "五花海观景台附近建议增加休息座椅和防滑提示。"},
        )
        assert submitted.status_code == 200
        candidate_id = submitted.json()["candidate_id"]
        assert candidate_id

        token = client.post(
            "/api/v1/admin/auth/login",
            json={"email": "admin@jiuzhaigou.local", "password": "admin123456"},
        ).json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        candidates = client.get("/api/v1/admin/feedback-candidates", headers=headers).json()["items"]
        candidate = next(item for item in candidates if item["candidate_id"] == candidate_id)
        assert candidate["status"] == "pending_review"

        reviewed = client.post(f"/api/v1/admin/feedback-candidates/{candidate_id}/approve", headers=headers)
        assert reviewed.status_code == 200
        body = reviewed.json()
        assert body["status"] == "approved"
        assert body["document_id"]
        assert body["task_id"]

        documents = client.get("/api/v1/admin/documents", headers=headers).json()["items"]
        published = next(item for item in documents if item["document_id"] == body["document_id"])
        assert published["source_type"] == "feedback_review"
        assert "休息座椅" in published["content"]

        task = client.get(f"/api/v1/admin/tasks/{body['task_id']}", headers=headers)
        assert task.status_code == 200
        assert task.json()["task_type"] == "embed_document"
        assert task.json()["status"] == "pending"

        repeated = client.post(f"/api/v1/admin/feedback-candidates/{candidate_id}/approve", headers=headers)
        assert repeated.status_code == 200
        assert repeated.json()["document_id"] == body["document_id"]
        assert repeated.json()["task_id"] == body["task_id"]


def test_conversation_history_returns_messages_for_its_own_conversation():
    """Persist both server- and client-created conversation ids for history reads."""
    with TestClient(app) as client:
        supplied_id = "client-created-conversation"
        answer = client.post(
            "/api/v1/chat/messages",
            json={"conversation_id": supplied_id, "message": "not a known attraction"},
        )
        assert answer.status_code == 200
        conversation_id = answer.json()["conversation_id"]
        assert conversation_id == supplied_id
        history = client.get(f"/api/v1/conversations/{conversation_id}")
        assert history.status_code == 200
        messages = history.json()["messages"]
        assert [message["role"] for message in messages[-2:]] == ["user", "assistant"]
        assert messages[-2]["content"] == "not a known attraction"


def test_embedding_worker_never_invents_a_vector_without_credentials():
    """A worker regression that writes a fake vector must fail this safety contract."""
    from app.worker import process_embedding_task

    with TestClient(app) as client:
        submitted = client.post("/api/v1/feedbacks", json={"content": "诺日朗中心站需要新增一个饮水点，需人工核实后入库。"})
        token = client.post(
            "/api/v1/admin/auth/login",
            json={"email": "admin@jiuzhaigou.local", "password": "admin123456"},
        ).json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        approved = client.post(
            f"/api/v1/admin/feedback-candidates/{submitted.json()['candidate_id']}/approve",
            headers=headers,
        ).json()
        outcome = process_embedding_task(approved["task_id"])
        assert outcome == "waiting_for_configuration"
        task = client.get(f"/api/v1/admin/tasks/{approved['task_id']}", headers=headers).json()
        assert task["status"] == "waiting_for_configuration"
