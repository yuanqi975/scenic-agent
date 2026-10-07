"""Run the retrieval evaluator inside the configured backend environment."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from app.services.evaluation import evaluate_questions


def main() -> None:
    path = Path(sys.argv[1])
    questions = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = asyncio.run(evaluate_questions(questions, top_k=5))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
