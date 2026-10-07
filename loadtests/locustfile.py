"""Focused load scenarios for browse, recommendation, cache, model and SSE."""
from __future__ import annotations

from locust import HttpUser, between, tag, task


class ScenicVisitor(HttpUser):
    wait_time = between(0.5, 1.5)

    @tag("browse")
    @task(4)
    def browse_attractions(self):
        self.client.get("/api/v1/attractions?limit=20", name="GET attractions")

    @tag("recommend")
    @task(2)
    def route_recommendation(self):
        self.client.post("/api/v1/recommendations", json={"duration_minutes": 240, "groups": ["老年游客"], "preferences": ["湖泊/海子"], "weather": "晴"}, name="POST recommendations")

    @tag("cache")
    @task(3)
    def cached_question(self):
        self.client.post("/api/v1/chat/messages", json={"message": "九寨沟门票多少钱"}, name="POST chat cached")

    @tag("model")
    @task(1)
    def model_question(self):
        self.client.post("/api/v1/chat/messages", json={"message": "五花海开放时间和游玩建议"}, name="POST chat model")

    @tag("sse")
    @task(1)
    def streamed_question(self):
        with self.client.post("/api/v1/chat/stream", json={"message": "诺日朗瀑布怎么去"}, name="POST chat stream", stream=True, catch_response=True) as response:
            body = response.content.decode("utf-8", errors="replace")
            if "event: result" not in body or "event: citations" not in body:
                response.failure("stream did not finish with citations and result events")
