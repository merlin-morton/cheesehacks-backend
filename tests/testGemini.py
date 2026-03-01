"""
Run locally to verify generate_question works (with or without GEMINI_API_KEY).

From repo root:
  python tests/testGemini.py
  python -m tests.testGemini
"""
import json
import os
import sys
from pathlib import Path

# Ensure repo root is on path when run as tests/testGemini.py
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Load .env if GEMINI/GOOGLE API key not in environment
if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
    from dotenv import load_dotenv
    load_dotenv(_repo_root / ".env")

import interfaceMLP


def main() -> None:
    print("Calling generate_question() with no args (random mood/topic)...")
    q = interfaceMLP.generate_question()
    print(json.dumps(q, indent=2))
    print()

    print("Calling generate_question(mood='playful', topic='honesty and lying')...")
    q2 = interfaceMLP.generate_question(mood="playful", topic="honesty and lying")
    print(json.dumps(q2, indent=2))


if __name__ == "__main__":
    main()
