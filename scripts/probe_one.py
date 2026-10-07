from __future__ import annotations

import json
import sys
import requests

if len(sys.argv) > 1 and sys.argv[1].startswith("cx_"):
    rows = [json.loads(line) for line in open("data/jiuzhaigou/evaluation_complex_questions_v3.jsonl", encoding="utf-8")]
    q = next(item["question"] for item in rows if item["question_id"] == sys.argv[1])
else:
    q = sys.argv[1]
r = requests.post("http://127.0.0.1:8000/api/v1/chat/messages", json={"message": q, "mode": "agent"}, timeout=180)
payload = r.json()
if len(sys.argv) > 2:
    with open(sys.argv[2], "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
else:
    print(r.status_code)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
