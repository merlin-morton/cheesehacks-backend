"""
MySQL database connection and query helpers for Cheesehacks backend.
Uses environment variables: MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE
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
    cfg = {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", ""),
        "database": os.getenv("MYSQL_DATABASE", "cheesehacks"),
    }
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
    """Fetch user by id. Returns None if not found."""
    with get_connection() as conn:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT id, provider, provider_sub, email, personality_vector, birthday, age, "
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
    Returns None if user not found or is hidden.
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
        out["personality_vector"] = user.get("personality_vector")
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


def update_personality_vector(user_id: str, vector_bytes: bytes) -> bool:
    """Store personality vector (BLOB)."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE users SET personality_vector = %s WHERE id = %s", (vector_bytes, user_id))
        affected = cur.rowcount
        cur.close()
    return affected > 0


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
