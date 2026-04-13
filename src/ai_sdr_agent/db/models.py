from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    Uuid,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> uuid.UUID:
    return uuid.uuid4()


class Base(DeclarativeBase):
    pass


class BotConfigRow(Base):
    __tablename__ = "bot_configs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_new_uuid)
    user_id: Mapped[uuid.UUID] = mapped_column(Uuid, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False, default="My Bot")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # LLM
    llm_provider: Mapped[str] = mapped_column(String(20), default="openai")
    llm_model_name: Mapped[str] = mapped_column(String(100), default="gpt-4o-mini")
    llm_temperature: Mapped[float] = mapped_column(Float, default=0.4)
    llm_max_tokens: Mapped[int] = mapped_column(Integer, default=300)
    openai_api_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    anthropic_api_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    groq_api_key: Mapped[str | None] = mapped_column(Text, nullable=True)

    # TTS – ElevenLabs
    elevenlabs_api_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    elevenlabs_voice_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    elevenlabs_model_id: Mapped[str] = mapped_column(String(100), default="eleven_turbo_v2")

    # STT – Deepgram
    deepgram_api_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    deepgram_model: Mapped[str] = mapped_column(String(50), default="nova-2")
    deepgram_language: Mapped[str] = mapped_column(String(10), default="en-US")

    # Telephony – Twilio
    twilio_account_sid: Mapped[str | None] = mapped_column(Text, nullable=True)
    twilio_auth_token: Mapped[str | None] = mapped_column(Text, nullable=True)
    twilio_phone_number: Mapped[str | None] = mapped_column(String(30), nullable=True)

    # Conversation behaviour
    initial_greeting: Mapped[str] = mapped_column(
        Text,
        default=(
            "Hi, this is John — I know I'm calling out of the blue. "
            "Do you have 30 seconds so I can tell you why I'm reaching out?"
        ),
    )
    max_call_turns: Mapped[int] = mapped_column(Integer, default=12)
    max_objection_attempts: Mapped[int] = mapped_column(Integer, default=2)
    max_qualify_attempts: Mapped[int] = mapped_column(Integer, default=3)
    max_booking_attempts: Mapped[int] = mapped_column(Integer, default=3)
    sales_rep_name: Mapped[str] = mapped_column(String(200), default="Sales Team")

    # Custom prompts (nullable → use defaults)
    prompt_greeting: Mapped[str | None] = mapped_column(Text, nullable=True)
    prompt_qualify: Mapped[str | None] = mapped_column(Text, nullable=True)
    prompt_pitch: Mapped[str | None] = mapped_column(Text, nullable=True)
    prompt_objection: Mapped[str | None] = mapped_column(Text, nullable=True)
    prompt_booking: Mapped[str | None] = mapped_column(Text, nullable=True)
    prompt_wrapup: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    leads: Mapped[list[LeadRow]] = relationship(back_populates="bot", cascade="all, delete-orphan")
    call_logs: Mapped[list[CallLogRow]] = relationship(back_populates="bot", cascade="all, delete-orphan")

    def to_config_dict(self) -> dict:
        """Return a plain dict suitable for embedding in ConversationState."""
        return {
            "bot_id": str(self.id),
            "llm_provider": self.llm_provider,
            "llm_model_name": self.llm_model_name,
            "llm_temperature": self.llm_temperature,
            "llm_max_tokens": self.llm_max_tokens,
            "openai_api_key": self.openai_api_key,
            "anthropic_api_key": self.anthropic_api_key,
            "groq_api_key": self.groq_api_key,
            "elevenlabs_api_key": self.elevenlabs_api_key,
            "elevenlabs_voice_id": self.elevenlabs_voice_id,
            "elevenlabs_model_id": self.elevenlabs_model_id,
            "deepgram_api_key": self.deepgram_api_key,
            "deepgram_model": self.deepgram_model,
            "deepgram_language": self.deepgram_language,
            "twilio_account_sid": self.twilio_account_sid,
            "twilio_auth_token": self.twilio_auth_token,
            "twilio_phone_number": self.twilio_phone_number,
            "initial_greeting": self.initial_greeting,
            "max_call_turns": self.max_call_turns,
            "max_objection_attempts": self.max_objection_attempts,
            "max_qualify_attempts": self.max_qualify_attempts,
            "max_booking_attempts": self.max_booking_attempts,
            "sales_rep_name": self.sales_rep_name,
            "prompt_greeting": self.prompt_greeting,
            "prompt_qualify": self.prompt_qualify,
            "prompt_pitch": self.prompt_pitch,
            "prompt_objection": self.prompt_objection,
            "prompt_booking": self.prompt_booking,
            "prompt_wrapup": self.prompt_wrapup,
        }


class LeadRow(Base):
    __tablename__ = "leads"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_new_uuid)
    bot_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("bot_configs.id", ondelete="CASCADE"), nullable=False, index=True)
    lead_name: Mapped[str] = mapped_column(String(200), nullable=False)
    company: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    phone_number: Mapped[str] = mapped_column(String(30), nullable=False)
    lead_email: Mapped[str] = mapped_column(String(320), nullable=False, default="")
    lead_context: Mapped[str] = mapped_column(Text, default="")
    lifecycle_stage: Mapped[str] = mapped_column(String(50), default="follow_up")
    timezone: Mapped[str] = mapped_column(String(50), default="UTC")
    owner_name: Mapped[str] = mapped_column(String(200), default="Sales Team")
    calendar_id: Mapped[str] = mapped_column(String(100), default="sales-team")
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    bot: Mapped[BotConfigRow] = relationship(back_populates="leads")

    __table_args__ = (
        UniqueConstraint("bot_id", "phone_number", name="uq_lead_bot_phone"),
    )


class CallLogRow(Base):
    __tablename__ = "call_logs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=_new_uuid)
    bot_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("bot_configs.id", ondelete="CASCADE"), nullable=False, index=True)
    conversation_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    lead_id: Mapped[str] = mapped_column(String(100), nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    call_outcome: Mapped[str] = mapped_column(String(30), default="follow_up_needed")
    transcript: Mapped[list] = mapped_column(JSON, default=list)
    qualification_notes: Mapped[dict] = mapped_column(JSON, default=dict)
    meeting_booked: Mapped[bool] = mapped_column(Boolean, default=False)
    proposed_slot: Mapped[str | None] = mapped_column(String(100), nullable=True)
    follow_up_action: Mapped[str | None] = mapped_column(String(100), nullable=True)

    bot: Mapped[BotConfigRow] = relationship(back_populates="call_logs")


class SessionRow(Base):
    __tablename__ = "sessions"

    conversation_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    bot_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("bot_configs.id", ondelete="CASCADE"), nullable=False, index=True)
    state_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)
