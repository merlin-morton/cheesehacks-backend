"""
MySQL database connection and query helpers for Align backend.
Uses environment variables: MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE, MYSQL_PORT

Cloud SQL (Cloud Run): set MYSQL_HOST=/cloudsql/PROJECT:REGION:INSTANCE to use Unix socket.
Local: MYSQL_HOST=localhost (default), optional MYSQL_PORT.
"""
import os
import json
from contextlib import contextmanager
from typing import Any, Optional

try:
    import mysql.connector
    from mysql.connector import Error as MySQLError
except ImportError:
    mysql = None
    MySQLError = Exception


def get_connection_config() -> dict:
    host = os.getenv("MYSQL_HOST", "localhost")
    cfg = {
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", ""),
        "database": os.getenv("MYSQL_DATABASE", "align"),
    }
    # Cloud SQL via Unix socket (Cloud Run)
    if host.startswith("/cloudsql/"):
        cfg["unix_socket"] = host
    else:
        cfg["host"] = host
        port = os.getenv("MYSQL_PORT")
        if port is not None:
            cfg["port"] = int(port)
    return cfg


@contextmanager
def get_connection():
    """Context manager for MySQL connections."""
    if mysql is None:
        raise RuntimeError("mysql-connector-python is not installed. Run: pip install mysql-connector-python")
    conn = mysql.connector.connect(**get_connection_config())
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _ensure_user_id(provider_sub: str, provider: str = "google") -> str:
    """Build user id as providerSub + provider."""
    return f"{provider_sub}{provider}"


# --- User operations ---

def get_user(user_id: str) -> Optional[dict]:
    """Fetch user by id. Returns None if not found. (Characteristics are in get_characteristics.)"""
    with get_connection() as conn:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT id, provider, provider_sub, email, birthday, age, "
            "user_settings, is_hidden, privacy_settings, created_at, updated_at FROM users WHERE id = %s",
            (user_id,),
        )
        row = cur.fetchone()
        cur.close()
    if not row:
        return None
    return _row_to_user(row)


def get_user_public(user_id: str) -> Optional[dict]:
    """
    Fetch user profile for public lookup. Respects privacy_settings and is_hidden.
    Returns None if user not found or is hidden. Public characteristics included when showPersonality.
    """
    user = get_user(user_id)
    if not user or user.get("is_hidden"):
        return None
    priv = user.get("privacy_settings") or {}
    if isinstance(priv, str):
        priv = json.loads(priv) if priv else {}
    out = {"id": user["id"], "email": user["email"] if priv.get("showEmail") else None}
    if priv.get("showAge"):
        out["age"] = user.get("age")
    if priv.get("showBirthday"):
        out["birthday"] = str(user["birthday"]) if user.get("birthday") else None
    if priv.get("showPersonality"):
        out["characteristics"] = get_characteristics(user_id, public_only=True)
    return out


def upsert_user(provider_sub: str, provider: str, email: str, **kwargs) -> dict:
    """Create or update user. id = providerSub + provider."""
    user_id = _ensure_user_id(provider_sub, provider)
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO users (id, provider, provider_sub, email)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                email = VALUES(email),
                updated_at = CURRENT_TIMESTAMP
            """,
            (user_id, provider, provider_sub, email),
        )
        cur.close()
    return get_user(user_id) or {}


def update_user_settings(user_id: str, settings: dict) -> bool:
    """Merge new settings into user_settings JSON."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT user_settings FROM users WHERE id = %s", (user_id,))
        row = cur.fetchone()
        if not row:
            cur.close()
            return False
        existing = json.loads(row[0] or "{}")
        existing.update(settings)
        cur.execute("UPDATE users SET user_settings = %s WHERE id = %s", (json.dumps(existing), user_id))
        cur.close()
    return True


def update_user_privacy(user_id: str, is_hidden: Optional[bool] = None, privacy_settings: Optional[dict] = None) -> bool:
    """Update privacy fields."""
    updates = []
    params = []
    if is_hidden is not None:
        updates.append("is_hidden = %s")
        params.append(is_hidden)
    if privacy_settings is not None:
        updates.append("privacy_settings = %s")
        params.append(json.dumps(privacy_settings))
    if not updates:
        return True
    params.append(user_id)
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(f"UPDATE users SET {', '.join(updates)} WHERE id = %s", params)
        affected = cur.rowcount
        cur.close()
    return affected > 0


def update_user_profile(user_id: str, birthday: Optional[str] = None, age: Optional[int] = None) -> bool:
    """Update optional profile fields."""
    updates = []
    params = []
    if birthday is not None:
        updates.append("birthday = %s")
        params.append(birthday)
    if age is not None:
        updates.append("age = %s")
        params.append(age)
    if not updates:
        return True
    params.append(user_id)
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(f"UPDATE users SET {', '.join(updates)} WHERE id = %s", params)
        affected = cur.rowcount
        cur.close()
    return affected > 0


# --- Characteristics (personality vector + other traits, each with is_public) ---

# Trait keys for personality/philosophical characteristics (use these or any string)
CHARACTERISTIC_KEYS = (
    "personality_vector", "star_sign", "myers_briggs", "attachment_style", "enneagram_type",
    "love_language", "moral_foundation", "political_leaning", "humor_style", "conflict_style",
    "learning_style", "big_five", "optimism_level", "introvert_extrovert", "chronotype",
    "decision_style", "creativity_style", "spirituality", "life_philosophy", "mindset",
    "core_values", "top_strengths", "communication_style", "stress_response", "motivation_style",
    "risk_tolerance", "perfectionism_level", "empathy_style", "leadership_style", "learning_orientation",
    "time_orientation", "self_monitoring", "need_for_closure", "cognitive_style", "emotional_expressiveness",
    "assertiveness", "emotional_intelligence", "curiosity_level",
)


def get_characteristics(user_id: str, public_only: bool = False) -> dict:
    """
    Get all characteristics for a user. Returns dict: trait_key -> value.
    For personality_vector value is bytes (vector of floats); for others, str.
    If public_only=True, only returns traits where is_public=TRUE.
    """
    with get_connection() as conn:
        cur = conn.cursor(dictionary=True)
        if public_only:
            cur.execute(
                "SELECT trait_key, value_text, value_blob FROM characteristics WHERE user_id = %s AND is_public = TRUE",
                (user_id,),
            )
        else:
            cur.execute(
                "SELECT trait_key, value_text, value_blob FROM characteristics WHERE user_id = %s",
                (user_id,),
            )
        rows = cur.fetchall()
        cur.close()
    out = {}
    for r in rows:
        k = r["trait_key"]
        if r["value_blob"] is not None:
            out[k] = r["value_blob"]
        elif r["value_text"] is not None:
            out[k] = r["value_text"]
    return out


def get_characteristics_with_visibility(user_id: str) -> list[dict]:
    """
    Get all characteristics with is_public and manually_overridden for API.
    Each item: { trait_key, value, is_public, manually_overridden }.
    personality_vector value is returned as bytes; caller can convert to list[float] for JSON.
    """
    with get_connection() as conn:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT trait_key, value_text, value_blob, is_public, manually_overridden FROM characteristics WHERE user_id = %s",
            (user_id,),
        )
        rows = cur.fetchall()
        cur.close()
    out = []
    for r in rows:
        if r["value_blob"] is not None:
            val = r["value_blob"]
        elif r["value_text"] is not None:
            val = r["value_text"]
        else:
            continue
        out.append({
            "trait_key": r["trait_key"],
            "value": val,
            "is_public": bool(r["is_public"]),
            "manually_overridden": bool(r.get("manually_overridden", False)),
        })
    return out


def set_characteristic(
    user_id: str,
    trait_key: str,
    value: Any,
    is_public: bool = False,
    *,
    value_is_blob: bool = False,
    manually_overridden: bool = False,
) -> bool:
    """
    Set one characteristic. value_is_blob=True for personality_vector (bytes); else value stored as text.
    manually_overridden=True when set via POST /profile/updateCharacteristics; MLP callback will not overwrite those.
    """
    with get_connection() as conn:
        cur = conn.cursor()
        if value_is_blob:
            cur.execute(
                """
                INSERT INTO characteristics (user_id, trait_key, value_blob, is_public, manually_overridden)
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE value_blob = VALUES(value_blob), value_text = NULL, is_public = VALUES(is_public), manually_overridden = VALUES(manually_overridden)
                """,
                (user_id, trait_key, value, is_public, manually_overridden),
            )
        else:
            text_val = value if isinstance(value, str) else json.dumps(value) if value is not None else None
            cur.execute(
                """
                INSERT INTO characteristics (user_id, trait_key, value_text, is_public, manually_overridden)
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE value_text = VALUES(value_text), value_blob = NULL, is_public = VALUES(is_public), manually_overridden = VALUES(manually_overridden)
                """,
                (user_id, trait_key, text_val, is_public, manually_overridden),
            )
        cur.close()
    return True


def get_manually_overridden_trait_keys(user_id: str) -> set:
    """Return set of trait_keys that were manually set via POST /profile/updateCharacteristics; MLP should not overwrite these."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT trait_key FROM characteristics WHERE user_id = %s AND manually_overridden = TRUE",
            (user_id,),
        )
        rows = cur.fetchall()
        cur.close()
    return {r[0] for r in rows}


def set_characteristic_visibility(user_id: str, trait_key: str, is_public: bool) -> bool:
    """Update only the is_public flag for a trait."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE characteristics SET is_public = %s WHERE user_id = %s AND trait_key = %s",
            (is_public, user_id, trait_key),
        )
        affected = cur.rowcount
        cur.close()
    return affected > 0


def get_personality_vector(user_id: str) -> Optional[bytes]:
    """Get personality vector BLOB for user from characteristics table. None if not set."""
    chars = get_characteristics(user_id, public_only=False)
    return chars.get("personality_vector")


def update_personality_vector(user_id: str, vector_bytes: bytes, is_public: bool = False) -> bool:
    """Store personality vector in characteristics table (trait_key=personality_vector)."""
    return set_characteristic(user_id, "personality_vector", vector_bytes, is_public, value_is_blob=True)


def _row_to_user(row: dict) -> dict:
    """Convert DB row to API-friendly dict."""
    out = dict(row)
    if "user_settings" in out and out["user_settings"]:
        out["user_settings"] = json.loads(out["user_settings"]) if isinstance(out["user_settings"], str) else out["user_settings"]
    if "privacy_settings" in out and out["privacy_settings"]:
        out["privacy_settings"] = json.loads(out["privacy_settings"]) if isinstance(out["privacy_settings"], str) else out["privacy_settings"]
    if "birthday" in out and out["birthday"]:
        out["birthday"] = str(out["birthday"])
    return out


# --- Friends ---

def add_friend(user_id: str, friend_id: str) -> tuple:
    """Add friend. Returns (success, message)."""
    if user_id == friend_id:
        return False, "Cannot add yourself as friend"
    if not get_user(friend_id):
        return False, "Friend user not found"
    if not get_user(user_id):
        return False, "User not found"
    try:
        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT IGNORE INTO friends (user_id, friend_id) VALUES (%s, %s), (%s, %s)",
                (user_id, friend_id, friend_id, user_id),
            )
            if cur.rowcount == 0:
                cur.close()
                return False, "Already friends"
            cur.close()
        return True, "Friend added"
    except MySQLError:
        raise


# --- Quiz ---

def get_quiz_response(user_id: str, question_id: str) -> Optional[dict]:
    """Get stored response for a question."""
    with get_connection() as conn:
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT response_data FROM quiz_responses WHERE user_id = %s AND question_id = %s", (user_id, question_id))
        row = cur.fetchone()
        cur.close()
    if not row:
        return None
    data = row["response_data"]
    return json.loads(data) if isinstance(data, str) else data


def save_quiz_response(user_id: str, question_id: str, response_data: dict) -> None:
    """Upsert quiz response."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO quiz_responses (user_id, question_id, response_data)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE response_data = VALUES(response_data)
            """,
            (user_id, question_id, json.dumps(response_data)),
        )
        cur.close()


# --- Diagnostics ---

def get_diagnostics(user_id: str) -> list[dict]:
    """Get diagnostics for user."""
    with get_connection() as conn:
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT id, diagnostic_data, created_at FROM diagnostics WHERE user_id = %s ORDER BY created_at DESC", (user_id,))
        rows = cur.fetchall()
        cur.close()
    for r in rows:
        d = r.get("diagnostic_data")
        if isinstance(d, str):
            r["diagnostic_data"] = json.loads(d) if d else {}
    return rows


def save_diagnostic(user_id: str, diagnostic_data: dict) -> int:
    """Insert diagnostic record. Returns id."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("INSERT INTO diagnostics (user_id, diagnostic_data) VALUES (%s, %s)", (user_id, json.dumps(diagnostic_data)))
        cur.execute("SELECT LAST_INSERT_ID()")
        rid = cur.fetchone()[0]
        cur.close()
    return rid
