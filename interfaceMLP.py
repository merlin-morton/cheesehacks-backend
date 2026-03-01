"""
Interface to a separate MLP (personality model). Stubs for now.
- Format question/response into text for the model.
- Update personality vector from one or many question/response strings.
- Call MLP with personality vector to get inferred characteristics (key-value); persist to characteristics table.
"""
import json
import os
import random
import re
import struct
from typing import Any

import db

import torch
from model import SharedEncoderBinaryHeads
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download

def load_mlp_model(checkpoint_path: str = None) -> SharedEncoderBinaryHeads:
    """
    Load the MLP model from a checkpoint path.
    """
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model = SharedEncoderBinaryHeads(input_dim=checkpoint['input_dim'], latent_dim=checkpoint['latent_dim'], tasks=checkpoint['tasks'])
        model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model

model_path = hf_hub_download(
    repo_id="Praneet-P/ethics-multihead-model",
    filename="checkpoints/shared_encoder_heads.pt"
)
inference_model = load_mlp_model(model_path)
embedder = SentenceTransformer("BAAI/bge-base-en-v1.5", device=str(torch.device("cuda" if torch.cuda.is_available() else "cpu")))

# --- Question types (match routes.py) ---
# 0 = single select, 1 = multi-select, 2 = scale, 3 = free text, 4 = ranking, 5 = yes/no, 6 = other


def _answers_to_texts(answers: list[Any]) -> list[str]:
    """Normalize answers to list of strings."""
    out = []
    for a in answers:
        if isinstance(a, dict) and "text" in a:
            out.append(str(a["text"]))
        else:
            out.append(str(a))
    return out


def format_question_response(
    question_text: str,
    question_type: int,
    answers: list[Any],
    response: Any,
) -> str:
    """
    Format a question and the user's response into a single string for the MLP.
    Only the user's response is included; unselected answer choices are not.
    Pattern: "I respond '...' when asked '...'."
    question_text: the question text
    question_type: 0-6 (single select, multi-select, scale, free text, ranking, yes/no, other)
    answers: list of answer texts or list of {"id", "text"} (used to resolve ids to text; not included in output)
    response: for 0/5 one chosen text or "yes"/"no"; for 1/4 list of texts; for 2 a number; for 3 a string; for 6 any
    """
    answer_texts = _answers_to_texts(answers)

    if question_type == 0:
        # Single select: "I respond 'X' when asked '...'."
        chosen = response if isinstance(response, str) else answer_texts[response] if isinstance(response, int) and response < len(answer_texts) else str(response)
        return f"I respond '{chosen}' when asked '{question_text}'."

    if question_type == 1:
        # Multi-select: "I respond 'A', 'B' when asked '...'."
        if isinstance(response, list):
            chosen = [r if isinstance(r, str) else answer_texts[r] if isinstance(r, int) and r < len(answer_texts) else str(r) for r in response]
        else:
            chosen = [str(response)]
        chosen_phrase = ", ".join(f"'{c}'" for c in chosen)
        return f"I respond {chosen_phrase} when asked '{question_text}'."

    if question_type == 2:
        # Scale: "I respond 'N' when asked '...'."
        return f"I respond '{response}' when asked '{question_text}'."

    if question_type == 3:
        # Free text: "I respond '...' when asked '...'."
        return f"I respond '{response}' when asked '{question_text}'."

    if question_type == 4:
        # Ranking: "I respond 'X', 'Y', 'Z' when asked '...'." (order = rank)
        if isinstance(response, list):
            ranked = [r if isinstance(r, str) else answer_texts[r] if isinstance(r, int) and r < len(answer_texts) else str(r) for r in response]
        else:
            ranked = [str(response)]
        rank_phrase = ", ".join(f"'{r}'" for r in ranked)
        return f"I respond {rank_phrase} when asked '{question_text}'."

    if question_type == 5:
        # Yes/no: "I respond 'yes' when asked '...'."
        yes_no = "yes" if response in (True, "yes", "y", 1) else "no"
        return f"I respond '{yes_no}' when asked '{question_text}'."

    if question_type == 6:
        return f"I respond '{response}' when asked '{question_text}'."

    return f"I respond '{response}' when asked '{question_text}'."


def _call_mlp(personality_vector: list[float], response_strings: list[str]) -> list[float]:
    """
    Stub: send current personality vector and formatted response strings to the MLP;
    return the updated personality vector.
    Replace with real HTTP/gRPC call to your ML service.
    """
    inference_model.eval()

    with torch.no_grad():
        combined = " ".join(response_strings)
        # run the sentence encoder
        emb = embedder.encode(
            combined,
            batch_size=1,
            normalize_embeddings=True,
            convert_to_tensor=True,
            show_progress_bar=False,
        )

        # run the MLP model
        personality_vector = inference_model.embedding_forward(emb)

    return list[float](personality_vector)


# Predefined lists for mood and topic when not provided (~20 each)
MOODS = [
    "reflective", "playful", "serious", "curious", "contemplative",
    "lighthearted", "philosophical", "challenging", "thoughtful", "witty",
    "provocative", "empathetic", "skeptical", "optimistic", "pragmatic",
    "idealistic", "neutral", "introspective", "bold", "gentle",
]

TOPICS = [
    "honesty and lying", "fairness and justice", "family and loyalty",
    "technology and privacy", "environment and future generations",
    "wealth and inequality", "freedom vs security", "punishment and mercy",
    "truth and consequences", "individual vs collective good",
    "consent and boundaries", "work and life balance", "sacrifice and duty",
    "forgiveness and revenge", "tradition vs progress", "rights and responsibilities",
    "life and death choices", "loyalty and betrayal", "creativity and ownership",
    "authority and disobedience",
]


def generate_question(mood: str | None = None, topic: str | None = None) -> dict[str, Any]:
    """
    Generate an interesting and engaging moral/ethical question using Google Gemini.
    mood: optional mood (e.g. reflective, playful). If blank, chosen randomly from MOODS.
    topic: optional general topic (e.g. honesty, fairness). If blank, chosen randomly from TOPICS.
    Returns question in standard format:
    { "id": int, "question_type": int, "question": { "number": int, "text": str }, "answers": [ { "id": int, "text": str } ] }
    """
    m = (mood or "").strip() or random.choice(MOODS)
    t = (topic or "").strip() or random.choice(TOPICS)

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        # Try loading .env from repo root (same dir as this file)
        try:
            from pathlib import Path
            from dotenv import load_dotenv
            _env_path = Path(__file__).resolve().parent / ".env"
            load_dotenv(_env_path)
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        except Exception:
            pass
    if not api_key:
        import sys
        print("No GEMINI_API_KEY or GOOGLE_API_KEY in env or .env (using fallback question).", file=sys.stderr)
        question_id = random.randint(1, 2**31 - 1)
        return {
            "id": question_id,
            "question_type": 5,
            "question": {"number": 1, "text": f"In a {m} mood, consider: What matters more—{t.replace(' and ', ' or ')}?"},
            "answers": [{"id": 0, "text": "Yes"}, {"id": 1, "text": "No"}],
        }

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    prompt = f"""Generate exactly one moral or ethical question that is interesting and engaging.
Mood: {m}
General topic: {t}

Reply with only a single JSON object, no other text, in this exact format:
{{"question": "<the question text>", "answers": ["<option A>", "<option B>", ...]}}
Provide 2 to 5 short answer options. The question should make people think about values and trade-offs."""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=512),
        )
        text = response.text if response.text else ""
    except Exception as e:
        import sys
        print(f"Gemini API error (using fallback): {e}", file=sys.stderr)
        question_id = random.randint(1, 2**31 - 1)
        return {
            "id": question_id,
            "question_type": 5,
            "question": {"number": 1, "text": f"In a {m} mood, consider: What matters more: {t.replace(' and ', ' or ')}?"},
            "answers": [{"id": 0, "text": "Yes"}, {"id": 1, "text": "No"}],
        }

    # Parse JSON from response (allow markdown code block wrapper)
    json_match = re.search(r"\{[\s\S]*\}", text)
    if not json_match:
        question_id = random.randint(1, 2**31 - 1)
        return {
            "id": question_id,
            "question_type": 5,
            "question": {"number": 1, "text": f"In a {m} mood, consider: What matters more—{t.replace(' and ', ' or ')}?"},
            "answers": [{"id": 0, "text": "Yes"}, {"id": 1, "text": "No"}],
        }
    try:
        data = json.loads(json_match.group())
        q_text = data.get("question") or data.get("question_text") or "What do you think?"
        raw_answers = data.get("answers") or data.get("options") or ["Yes", "No"]
        if not isinstance(raw_answers, list):
            raw_answers = ["Yes", "No"]
        answers = [{"id": i, "text": str(a)[:200]} for i, a in enumerate(raw_answers[:10])]
    except (json.JSONDecodeError, TypeError):
        question_id = random.randint(1, 2**31 - 1)
        return {
            "id": question_id,
            "question_type": 5,
            "question": {"number": 1, "text": f"In a {m} mood, consider: What matters more—{t.replace(' and ', ' or ')}?"},
            "answers": [{"id": 0, "text": "Yes"}, {"id": 1, "text": "No"}],
        }

    question_id = random.randint(1, 2**31 - 1)
    question_type = 0 if len(answers) > 2 else 5  # single select or yes/no
    return {
        "id": question_id,
        "question_type": question_type,
        "question": {"number": 1, "text": q_text[:500]},
        "answers": answers,
    }


def _call_mlp_characteristics_callback(personality_vector: list[float]) -> dict[str, Any]:
    """
    Stub: different callback — send personality vector to the MLP and get back
    inferred characteristics as key-value pairs (e.g. star_sign, myers_briggs).
    Replace with real HTTP/gRPC call to your ML characteristics endpoint.
    Returns dict of trait_key -> value (values should be string or JSON-serializable).
    """
    # Stub: return example characteristics; replace with actual MLP call
    return {}


def _vector_to_bytes(vec: list[float]) -> bytes:
    """Serialize list of floats to BLOB bytes (double precision)."""
    if not vec:
        return b""
    return struct.pack(f"{len(vec)}d", *vec)


def _bytes_to_vector(data: bytes | None) -> list[float]:
    """Deserialize BLOB bytes to list of floats."""
    if not data:
        return []
    n = len(data) // 8
    return list(struct.unpack(f"{n}d", data[: n * 8]))


def update_personality_from_batch(
    user_id: str,
    personality_vector: list[float],
    formatted_response_strings: list[str],
) -> list[float]:
    """
    Send personality vector and a batch of formatted question/response strings to the MLP;
    receive the updated vector and persist it for the user.
    Returns the new personality vector (list of floats).
    """
    new_vector = _call_mlp(personality_vector, formatted_response_strings)
    db.update_personality_vector(user_id, _vector_to_bytes(new_vector))
    return new_vector


def update_characteristics_from_mlp(
    user_id: str,
    personality_vector: list[float],
    is_public: bool = False,
) -> dict[str, Any]:
    """
    Call the MLP characteristics callback with the personality vector; MLP returns
    key-value characteristics. For each (key, value), update the characteristics
    table only if that trait was not manually overridden via POST /profile/updateCharacteristics.
    Skips the key 'personality_vector' so the vector is not overwritten.
    Returns the dict of traits that were applied.
    """
    characteristics = _call_mlp_characteristics_callback(personality_vector)
    overridden = db.get_manually_overridden_trait_keys(user_id)
    applied = {}
    for trait_key, value in characteristics.items():
        if trait_key == "personality_vector":
            continue
        if value is None:
            continue
        if trait_key in overridden:
            continue
        text_val = value if isinstance(value, str) else json.dumps(value)
        db.set_characteristic(user_id, trait_key, text_val, is_public, value_is_blob=False, manually_overridden=False)
        applied[trait_key] = value
    return applied


def fetch_and_save_characteristics_from_mlp(
    user_id: str,
    is_public: bool = False,
    *,
    personality_vector: list[float] | None = None,
) -> dict[str, Any]:
    """
    Load user's personality vector (or use provided one), call MLP characteristics
    callback, then persist every returned key-value into the characteristics table.
    Returns the dict of traits that were applied.
    """
    vec = personality_vector if personality_vector is not None else get_user_personality_vector(user_id)
    return update_characteristics_from_mlp(user_id, vec, is_public=is_public)


def update_personality_from_response(
    user_id: str,
    personality_vector: list[float],
    question_response_pair: dict[str, Any],
) -> list[float]:
    """
    Format a single question/response pair, then update the user's personality vector
    via the MLP (delegates to update_personality_from_batch with one string).
    question_response_pair: dict with question_text, question_type, answers, response
    Returns the new personality vector.
    """
    text = format_question_response(
        question_text=question_response_pair["question_text"],
        question_type=question_response_pair["question_type"],
        answers=question_response_pair["answers"],
        response=question_response_pair["response"],
    )
    return update_personality_from_batch(user_id, personality_vector, [text])


# --- Helpers for routes: get current vector from user, then update ---


def get_user_personality_vector(user_id: str) -> list[float]:
    """Load user's personality vector from characteristics table as list of floats. Returns [] if none."""
    raw = db.get_personality_vector(user_id)
    return _bytes_to_vector(raw)


def update_personality_after_response(user_id: str, question_response_pair: dict[str, Any]) -> list[float]:
    """
    Load user's current personality vector, run format + MLP update for one Q/R pair,
    save and return the new vector.
    """
    vec = get_user_personality_vector(user_id)
    return update_personality_from_response(user_id, vec, question_response_pair)


def update_personality_after_batch(user_id: str, question_response_pairs: list[dict[str, Any]]) -> list[float]:
    """
    Load user's current personality vector, format all pairs, run MLP batch update,
    save and return the new vector.
    """
    vec = get_user_personality_vector(user_id)
    strings = [
        format_question_response(
            qr["question_text"],
            qr["question_type"],
            qr["answers"],
            qr["response"],
        )
        for qr in question_response_pairs
    ]
    return update_personality_from_batch(user_id, vec, strings)
