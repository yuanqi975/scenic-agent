from __future__ import annotations

import json
import sys
import requests

wanted = set(sys.argv[1:-1])
out = sys.argv[-1]
rows = [json.loads(line) for line in open("data/jiuzhaigou/evaluation_complex_questions_v3.jsonl", encoding="utf-8")]
result = {}
for item in rows:
    if item["question_id"] not in wanted:
        continue
    response = requests.post("http://127.0.0.1:8000/api/v1/chat/messages", json={"message": item["question"], "mode": "agent"}, timeout=180)
    result[item["question_id"]] = response.json()
with open(out, "w", encoding="utf-8") as handle:
    json.dump(result, handle, ensure_ascii=False, indent=2)
