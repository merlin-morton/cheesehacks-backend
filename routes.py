from fastapi import APIRouter, FastAPI, Header, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Any

import db

# --- Dependencies ---

def get_current_user_id(x_user_id: Optional[str] = Header(None, alias="X-User-Id")) -> str:
    """Get current user from X-User-Id header (set by auth middleware after Google login)."""
    if not x_user_id:
        raise HTTPException(status_code=401, detail="X-User-Id header required")
    return x_user_id


# --- Quiz question JSON format (response from getQuestion) ---

class QuestionAnswer(BaseModel):
    id: int
    text: str


class QuestionContent(BaseModel):
    number: int
    text: str


class QuizQuestion(BaseModel):
    """Question payload returned by GET /quiz/getQuestion."""
    id: int
    question_type: int  # 0–6: e.g. 0=single select, 1=multi-select, etc.
    question: QuestionContent
    answers: list[QuestionAnswer]


class QuizQuestionResponse(BaseModel):
    """Full getQuestion response: question + optional prior_response."""
    id: int
    question_type: int
    question: QuestionContent
    answers: list[QuestionAnswer]
    prior_response: Optional[dict[str, Any]] = None


# --- Pydantic models ---

class SendResponseBody(BaseModel):
    question_id: int  # Matches QuizQuestion.id
    response_data: dict[str, Any]  # e.g. {"selected_ids": [0, 2]} for multi-select


class SubmitQuizBody(BaseModel):
    personality_vector: list[float]  # Will be serialized to bytes for BLOB storage


class UpdateSettingsBody(BaseModel):
    user_settings: dict[str, Any]


class UpdatePrivacyBody(BaseModel):
    is_hidden: Optional[bool] = None
    privacy_settings: Optional[dict[str, Any]] = None


class AddFriendBody(BaseModel):
    friend_id: str


class RegisterBody(BaseModel):
    """Called after Google login to create/update user."""
    provider_sub: str
    provider: str = "google"
    email: str


class TraitUpdate(BaseModel):
    """One characteristic to set. value: string for text traits, list[float] only for personality_vector."""
    trait_key: str
    value: Any  # str for most traits; list[float] for personality_vector
    is_public: bool = False


class UpdateCharacteristicsBody(BaseModel):
    traits: list[TraitUpdate]


# --- Stub quiz questions (replace with DB later) ---
# question_type: 0=single select, 1=multi-select, 2=scale, 3=free text, 4=ranking, 5=yes/no, 6=other
STUB_QUESTIONS: list[dict] = [
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


# Quiz routes
quiz_router = APIRouter(prefix="/quiz", tags=["quiz"])


@quiz_router.get("/")
async def quiz_root():
    return {"message": "Quiz API"}


@quiz_router.get("/getQuestion", response_model=QuizQuestionResponse)
async def get_question(
    question_id: Optional[int] = Query(None, description="Specific question id, or omit for next question"),
    index: Optional[int] = Query(None, description="0-based index into question list (used when question_id omitted)"),
    user_id: str = Depends(get_current_user_id),
):
    """Get a quiz question in the standard JSON format. Includes prior_response if user already answered this question."""
    prior = None
    if question_id is not None:
        prior = db.get_quiz_response(user_id, str(question_id))
    # Resolve which question to return
    if question_id is not None:
        q = next((q for q in STUB_QUESTIONS if q["id"] == question_id), None)
    else:
        idx = index if index is not None else 0
        q = STUB_QUESTIONS[idx % len(STUB_QUESTIONS)] if STUB_QUESTIONS else None
    if not q:
        raise HTTPException(status_code=404, detail="Question not found")
    return QuizQuestionResponse(prior_response=prior, **q)


@quiz_router.post("/sendResponse")
async def send_response(
    body: SendResponseBody,
    user_id: str = Depends(get_current_user_id),
):
    db.save_quiz_response(user_id, str(body.question_id), body.response_data)
    return {"message": "Response received", "question_id": body.question_id}


@quiz_router.post("/submit")
async def submit_quiz(
    body: SubmitQuizBody,
    user_id: str = Depends(get_current_user_id),
):
    """Submit quiz and store personality vector in user profile."""
    import struct
    vector_bytes = struct.pack(f"{len(body.personality_vector)}d", *body.personality_vector)
    ok = db.update_personality_vector(user_id, vector_bytes)
    if not ok:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "Quiz submitted", "user_id": user_id}


# Diagnostics routes
diagnostics_router = APIRouter(prefix="/diagnostics", tags=["diagnostics"])


@diagnostics_router.get("/")
async def diagnostics_root():
    return {"message": "Diagnostics API"}


@diagnostics_router.get("/getDiagnostics")
async def get_diagnostics(user_id: str = Depends(get_current_user_id)):
    rows = db.get_diagnostics(user_id)
    return {"diagnostics": rows}


# Profile routes
profile_router = APIRouter(prefix="/profile", tags=["profile"])


@profile_router.get("/")
async def profile_root():
    return {"message": "Profile API"}


@profile_router.post("/register")
async def register(body: RegisterBody):
    """Register or update user after Google login. id = providerSub + provider."""
    user = db.upsert_user(body.provider_sub, body.provider, body.email)
    return {"message": "Registered", "user_id": user["id"], "user": user}


@profile_router.get("/my")
async def profile_my(user_id: str = Depends(get_current_user_id)):
    user = db.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@profile_router.get("/getSettings")
async def get_settings(user_id: str = Depends(get_current_user_id)):
    user = db.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"user_settings": user.get("user_settings") or {}}


@profile_router.post("/updateSettings")
async def update_settings(
    body: UpdateSettingsBody,
    user_id: str = Depends(get_current_user_id),
):
    ok = db.update_user_settings(user_id, body.user_settings)
    if not ok:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "Settings updated", "user_settings": body.user_settings}


@profile_router.post("/updatePrivacy")
async def update_privacy(
    body: UpdatePrivacyBody,
    user_id: str = Depends(get_current_user_id),
):
    ok = db.update_user_privacy(
        user_id,
        is_hidden=body.is_hidden,
        privacy_settings=body.privacy_settings,
    )
    if not ok:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "Privacy updated"}


@profile_router.get("/getProfile")
async def get_profile(
    user_id: Optional[str] = Query(None, alias="user_id"),
):
    """Get a user's public profile. Respects privacy (is_hidden, privacy_settings)."""
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id query param required")
    profile = db.get_user_public(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found or hidden")
    return profile


@profile_router.post("/addFriend")
async def add_friend(
    body: AddFriendBody,
    user_id: str = Depends(get_current_user_id),
):
    success, msg = db.add_friend(user_id, body.friend_id)
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"message": msg}


@profile_router.get("/getCharacteristics")
async def get_characteristics(user_id: str = Depends(get_current_user_id)):
    """Get current user's characteristics (personality vector as list of floats; other traits as text). Each has is_public."""
    import struct
    rows = db.get_characteristics_with_visibility(user_id)
    out = []
    for r in rows:
        val = r["value"]
        if r["trait_key"] == "personality_vector" and isinstance(val, bytes):
            n = len(val) // 8
            val = list(struct.unpack(f"{n}d", val[: n * 8]))
        elif isinstance(val, bytes):
            val = val.decode("utf-8", errors="replace")
        out.append({
            "trait_key": r["trait_key"],
            "value": val,
            "is_public": r["is_public"],
            "manually_overridden": r.get("manually_overridden", False),
        })
    return {"characteristics": out}


@profile_router.post("/updateCharacteristics")
async def update_characteristics(
    body: UpdateCharacteristicsBody,
    user_id: str = Depends(get_current_user_id),
):
    """Set or update characteristics. Marks traits as manually_overridden so MLP callback will not overwrite them."""
    import struct
    if not db.get_user(user_id):
        raise HTTPException(status_code=404, detail="User not found")
    for t in body.traits:
        if t.trait_key == "personality_vector" and isinstance(t.value, list):
            vector_bytes = struct.pack(f"{len(t.value)}d", *t.value)
            db.set_characteristic(user_id, t.trait_key, vector_bytes, t.is_public, value_is_blob=True, manually_overridden=True)
        else:
            db.set_characteristic(user_id, t.trait_key, t.value, t.is_public, value_is_blob=False, manually_overridden=True)
    return {"message": "Characteristics updated", "count": len(body.traits)}


# Main
app = FastAPI(title="Aligned", description="Quiz, profile, and diagnostics API.")
app.include_router(quiz_router)
app.include_router(diagnostics_router)
app.include_router(profile_router)


def _handle_mysql_unavailable(request, exc):
    return JSONResponse(
        status_code=503,
        content={
            "detail": "Database unavailable. Is MySQL running? Start the MySQL service (e.g. Win+R → services.msc → start MySQL) and check MYSQL_HOST/MYSQL_PORT in .env."
        },
    )


try:
    from mysql.connector.errors import InterfaceError as MySQLInterfaceError
    app.add_exception_handler(MySQLInterfaceError, _handle_mysql_unavailable)
except ImportError:
    pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
