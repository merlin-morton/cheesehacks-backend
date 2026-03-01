"""
Populate the questions cache table from a JSON file or built-in defaults.
Run from project root: python populate_questions.py [path/to/questions.json]
If no path given, uses built-in default questions.
"""
import json
import sys
from pathlib import Path

import db

# Default questions (same shape as routes.py STUB_QUESTIONS) if no file provided
DEFAULT_QUESTIONS = [
    {
        "id": 294029492059020,
        "question_type": 0,
        "question": {"number": 1, "text": "What is the middle of the world?"},
        "answers": [
            {"id": 0, "text": "foo"},
            {"id": 1, "text": "bar"},
            {"id": 2, "text": "baz"},
        ],
    },
    {
        "id": 294029492059021,
        "question_type": 1,
        "question": {"number": 2, "text": "Pick all that apply."},
        "answers": [
            {"id": 0, "text": "Option A"},
            {"id": 1, "text": "Option B"},
            {"id": 2, "text": "Option C"},
        ],
    },
]


def load_questions(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return DEFAULT_QUESTIONS
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "questions" in data:
        return data["questions"]
    return [data]


def main() -> None:
    path = None
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    questions = load_questions(path)
    for q in questions:
        qid = q.get("id")
        if qid is None:
            continue
        db.cache_question(qid, q)
        print(f"Cached question id={qid}")
    print(f"Done. Cached {len(questions)} question(s).")


if __name__ == "__main__":
    main()
