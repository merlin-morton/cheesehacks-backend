from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import db
import interfaceMLP
from schemas import (
    AddFriendBody,
    QuizQuestionResponse,
    QuizResponseItem,
    RegisterBody,
    SendResponseBody,
    SubmitQuizBody,
    TraitUpdate,
    UpdateCharacteristicsBody,
    UpdatePrivacyBody,
    UpdateSettingsBody,
)

# --- Dependencies ---


def get_current_user_id(x_user_id: Optional[str] = Header(None, alias="X-User-Id")) -> str:
    """Get current user from X-User-Id header (set by auth middleware after Google login)."""
    if not x_user_id:
        raise HTTPException(status_code=401, detail="X-User-Id header required")
    return x_user_id


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
    question_id: Optional[int] = Query(None, description="Specific question id, or omit to generate a new question"),
    mood: Optional[str] = Query(None, description="Mood for generated question (e.g. reflective, playful); random if omitted"),
    topic: Optional[str] = Query(None, description="General topic for generated question (e.g. honesty, fairness); random if omitted"),
    user_id: str = Depends(get_current_user_id),
):
    """Get a quiz question in the standard JSON format. Includes prior_response if user already answered this question."""
    # 1) Resolve question content: by id from cache, or generate new question via Gemini
    if question_id is not None:
        q = db.get_question_to_return(question_id)
        if q is None:
            q = next((s for s in STUB_QUESTIONS if s["id"] == question_id), None)
    else:
        q = interfaceMLP.generate_question(mood=mood, topic=topic)
        db.cache_question(q["id"], q)
    if not q:
        raise HTTPException(status_code=404, detail="Question not found")
    # 2) Attach this user's prior answer for this question (if any)
    prior = db.get_quiz_response(user_id, str(q["id"]))
    return QuizQuestionResponse(prior_response=prior, **q)


@quiz_router.post("/sendResponse")
async def send_response(
    body: SendResponseBody,
    user_id: str = Depends(get_current_user_id),
):
    db.save_quiz_response(user_id, str(body.question_id), body.response_data)
    return {"message": "Response received", "question_id": body.question_id}


def _response_data_to_answer(response_data: dict, question_type: int) -> Any:
    """Map client response_data (selected_ids, ranked_ids, text) to format_question_response response."""
    if "text" in response_data:
        return response_data["text"]
    if "ranked_ids" in response_data:
        return response_data["ranked_ids"]  # list of ids (order = rank)
    sel = response_data.get("selected_ids") or []
    if question_type == 0:
        return sel[0] if sel else None  # single id
    if question_type in (1, 2):
        return sel  # list of ids (multi-select or scale as single choice)
    if question_type == 5:
        return sel[0] if sel else None  # yes/no index
    return sel[0] if sel else None


@quiz_router.post("/submit")
async def submit_quiz(
    body: SubmitQuizBody,
    user_id: str = Depends(get_current_user_id),
):
    """Submit batch of quiz responses: format each, send to MLP, update personality vector."""
    pairs = []
    for item in body.responses:
        q = db.get_question_to_return(item.questionId) or db.get_cached_question(item.questionId)
        if not q:
            raise HTTPException(status_code=404, detail=f"Question {item.questionId} not found")
        response = _response_data_to_answer(item.response_data, q["question_type"])
        if response is None and "text" not in item.response_data:
            continue
        pairs.append({
            "question_text": q["question"]["text"],
            "question_type": q["question_type"],
            "answers": q.get("answers") or [],
            "response": response,
        })
        db.save_quiz_response(user_id, str(item.questionId), item.response_data)
    if not pairs:
        raise HTTPException(status_code=400, detail="No valid responses to submit")
    new_vector = interfaceMLP.update_personality_after_batch(user_id, pairs)
    return {"message": "Quiz submitted", "user_id": user_id, "vector_length": len(new_vector)}


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
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: ensure DB and tables exist. Shutdown: (none)."""
    try:
        db.ensure_schema()
    except Exception:
        pass
    yield


app = FastAPI(title="Align", description="Quiz, profile, and diagnostics API.", lifespan=lifespan)

# Allow browser requests from any origin (fixes CORS when calling from other sites/devices)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
