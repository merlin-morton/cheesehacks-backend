"""
Interface to a separate MLP (personality model). Stubs for now.
- Format question/response into text for the model.
- Update personality vector from one or many question/response strings.
- Call MLP with personality vector to get inferred characteristics (key-value); persist to characteristics table.
"""
import json
import struct
from typing import Any

import db


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
    question_text: the question text
    question_type: 0-6 (single select, multi-select, scale, free text, ranking, yes/no, other)
    answers: list of answer texts or list of {"id", "text"}
    response: for 0/5 one chosen text or "yes"/"no"; for 1/4 list of texts; for 2 a number; for 3 a string; for 6 any
    """
    answer_texts = _answers_to_texts(answers)
    answers_phrase = ", ".join(f"'{t}'" for t in answer_texts) if answer_texts else ""

    if question_type == 0:
        # Single select: "When asked, '...', I would choose 'X' of 'A', 'B', 'C'."
        chosen = response if isinstance(response, str) else answer_texts[response] if isinstance(response, int) else str(response)
        return f"When asked, '{question_text}', I would choose '{chosen}' of {answers_phrase}."

    if question_type == 1:
        # Multi-select: "When asked, '...', I would choose 'X', 'Y' of 'A', 'B', 'C'."
        if isinstance(response, list):
            chosen = [r if isinstance(r, str) else answer_texts[r] if isinstance(r, int) else str(r) for r in response]
        else:
            chosen = [str(response)]
        chosen_phrase = ", ".join(f"'{c}'" for c in chosen)
        return f"When asked, '{question_text}', I would choose {chosen_phrase} of {answers_phrase}."

    if question_type == 2:
        # Scale: "When asked, '...', I answered with scale value N."
        return f"When asked, '{question_text}', I answered with scale value {response}."

    if question_type == 3:
        # Free text: "When asked, '...', I said '...'."
        return f"When asked, '{question_text}', I said '{response}'."

    if question_type == 4:
        # Ranking: "When asked, '...', I ranked: first 'X', then 'Y', then 'Z'."
        if isinstance(response, list):
            ranked = [r if isinstance(r, str) else answer_texts[r] if isinstance(r, int) else str(r) for r in response]
        else:
            ranked = [str(response)]
        rank_phrase = ", then ".join(f"'{r}'" for r in ranked)
        return f"When asked, '{question_text}', I ranked: {rank_phrase}."

    if question_type == 5:
        # Yes/no: "When asked, '...', I said yes/no."
        yes_no = "yes" if response in (True, "yes", "y", 1) else "no"
        return f"When asked, '{question_text}', I said {yes_no}."

    if question_type == 6:
        # Other: generic
        return f"When asked, '{question_text}', I responded: {response}."

    return f"When asked, '{question_text}', I responded: {response}."


def _call_mlp(personality_vector: list[float], response_strings: list[str]) -> list[float]:
    """
    Stub: send current personality vector and formatted response strings to the MLP;
    return the updated personality vector.
    Replace with real HTTP/gRPC call to your ML service.
    """
    # Stub: return same vector (or a copy so caller can mutate)
    return list(personality_vector)


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
    table. Skips the key 'personality_vector' so the vector is not overwritten.
    Returns the dict of traits that were applied.
    """
    characteristics = _call_mlp_characteristics_callback(personality_vector)
    applied = {}
    for trait_key, value in characteristics.items():
        if trait_key == "personality_vector":
            continue
        if value is None:
            continue
        text_val = value if isinstance(value, str) else json.dumps(value)
        db.set_characteristic(user_id, trait_key, text_val, is_public, value_is_blob=False)
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
