"""
Pydantic models for API request/response JSON.
"""
from typing import Any, Optional

from pydantic import BaseModel, Field


# --- Quiz question (response from getQuestion) ---


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


# --- Quiz request bodies ---


class SendResponseBody(BaseModel):
    question_id: int  # Matches QuizQuestion.id
    response_data: dict[str, Any]  # e.g. {"selected_ids": [0, 2]} for multi-select


class QuizResponseItem(BaseModel):
    questionId: int
    response_data: dict[str, Any]  # { selected_ids: [int] } | { ranked_ids: [int] } | { text: str }


class SubmitQuizBody(BaseModel):
    """Batch of quiz responses: format each, send to MLP, update personality vector."""

    responses: list[QuizResponseItem] = Field(
        ...,
        alias="quizResponses",
        description="List of questionId + response_data; client may send as quizResponses (camelCase).",
    )

    model_config = {"populate_by_name": True}  # accept both "responses" and "quizResponses"


# --- Profile / settings ---


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


# --- Characteristics ---


class TraitUpdate(BaseModel):
    """One characteristic to set. value: string for text traits, list[float] only for personality_vector."""

    trait_key: str
    value: Any  # str for most traits; list[float] for personality_vector
    is_public: bool = False


class UpdateCharacteristicsBody(BaseModel):
    traits: list[TraitUpdate]
