"""Run the offline retrieval evaluator against a JSONL dataset.

Usage: ``python scripts/evaluate_rag.py --input data/jiuzhaigou/evaluation_questions.jsonl``
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services.evaluation import evaluate_questions  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", choices=("all", "baseline", "complex"), default="all")
    args = parser.parse_args()
    questions = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.split != "all":
        questions = [q for q in questions if q.get("evaluation_split", "baseline") == args.split]
    report = asyncio.run(evaluate_questions(questions, top_k=args.top_k))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
