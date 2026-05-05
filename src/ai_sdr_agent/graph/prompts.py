from __future__ import annotations

import re

from ai_sdr_agent.graph.state import ConversationState

# Appended to every spoken-response system prompt (defaults and custom).
_VOICE_OUTPUT_RULES = """
---
Voice output (mandatory — this is read aloud by TTS in one pass):
- Use one or two short sentences in most turns. Use at most three very short sentences only when listing meeting time options.
- No paragraphs, bullet lists, markdown, or numbered lists.
- Do not use line breaks; write a single continuous line of speech (commas and periods are fine).
- Do not end the call yourself: avoid saying goodbye, hang up, "I'll let you go", "I need to go", or pretending the call ended unless the user clearly wants to stop. Stay on topic and follow your node's goal until routing moves on.
"""


def format_reply_for_tts(text: str) -> str:
    """Collapse newlines and extra whitespace so Vocode sends one synthesis string."""
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"\n+", " ", t)
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


def _template_vars(state: ConversationState) -> dict[str, str]:
    """Common template variables available in custom prompts."""
    return {
        "lead_name": state["lead_name"],
        "company": state["company"],
        "lead_context": state["lead_context"],
        "calendar_id": state["calendar_id"],
        "pain_points": ", ".join(state["pain_points"]) or "none yet",
        "call_outcome": state["call_outcome"],
        "meeting_booked": str(state["meeting_booked"]),
        "sales_rep_name": state.get("bot_config", {}).get("sales_rep_name", "Sales Team"),
    }


def _apply_custom(state: ConversationState, key: str, default_fn) -> str:
    """Return the custom prompt (with variable interpolation) if set, else the default."""
    custom = state.get("bot_config", {}).get(key)
    if custom:
        try:
            return custom.format(**_template_vars(state))
        except (KeyError, IndexError):
            return custom
    return default_fn(state)


def _with_voice_rules(body: str) -> str:
    return body.rstrip() + _VOICE_OUTPUT_RULES


# ── Default prompt builders ─────────────────────────────────────────

def _default_greeting_prompt(state: ConversationState) -> str:
    return (
        f"You are an AI SDR making an outbound cold call.\n"
        f"Lead: {state['lead_name']} at {state['company']}.\n"
        f"CRM context: {state['lead_context']}.\n\n"
        f"Goal for this turn: confirm you reached the right person and ask permission "
        f"to explain why you're calling. Be transparent that this is a cold call. "
        f"Do not pitch yet."
    )


def _default_qualify_prompt(state: ConversationState) -> str:
    known_dm = state.get("is_decision_maker")
    known_budget = state.get("budget_confirmed")
    known_timeline = state.get("timeline")
    known_pain = ", ".join(state["pain_points"]) if state["pain_points"] else "none yet"
    attempt = state.get("qualify_attempts", 0) + 1
    max_attempts = state.get("bot_config", {}).get("max_qualify_attempts", 3)

    return (
        f"You are qualifying the prospect on a live outbound sales call.\n"
        f"Lead: {state['lead_name']} at {state['company']}.\n"
        f"Known context: {state['lead_context']}.\n"
        f"Qualification attempt: {attempt} of {max_attempts}.\n\n"
        f"What we know so far:\n"
        f"- Decision maker: {known_dm}\n"
        f"- Budget confirmed: {known_budget}\n"
        f"- Timeline: {known_timeline}\n"
        f"- Pain points: {known_pain}\n\n"
        f"Goal for this turn: ask the NEXT unanswered qualification question.\n"
        f"Prioritize: role/authority, then pain points, then budget, then timeline.\n"
        f"Only ask ONE question at a time. If they just answered, acknowledge in a few "
        f"words, then ask the next question in the same spoken reply."
    )


def _default_pitch_prompt(state: ConversationState) -> str:
    pain_points = ", ".join(state["pain_points"]) or "slow lead follow-up"
    return (
        f"You are the SDR for an AI outbound calling and follow-up platform.\n"
        f"Prospect: {state['lead_name']} at {state['company']}.\n"
        f"Pain points heard so far: {pain_points}.\n\n"
        f"Goal for this turn: one tight pitch focused on business outcomes, then one "
        f"question about taking a next step. No feature dump."
    )


def _default_objection_prompt(state: ConversationState) -> str:
    return (
        f"You are handling an objection on a live SDR follow-up call.\n"
        f"Prospect: {state['lead_name']} at {state['company']}.\n"
        f"Context: {state['lead_context']}.\n\n"
        f"Goal for this turn: acknowledge, one helpful reframe, one low-commitment ask — "
        f"still in one or two sentences total."
    )


def _default_booking_prompt(state: ConversationState) -> str:
    slot_lines = "\n".join(f"- {slot['label']}" for slot in state["available_slots"][:3])
    return (
        f"You are booking a meeting for the sales rep.\n"
        f"Prospect: {state['lead_name']} at {state['company']}.\n"
        f"Available slots:\n{slot_lines}\n\n"
        f"Goal for this turn: book a meeting. Offer up to three time options in plain "
        f"spoken language (no lists); when confirming, restate the chosen time in one short line."
    )


def _default_wrap_up_prompt(state: ConversationState) -> str:
    return (
        f"You need to wrap up this SDR call with {state['lead_name']} from {state['company']}.\n"
        f"Current outcome: {state['call_outcome']}.\n"
        f"Meeting booked: {state['meeting_booked']}.\n\n"
        f"Goal for this turn: polite close, one sentence on next steps, done."
    )


# ── Public API (used by nodes.py) ───────────────────────────────────

def greeting_prompt(state: ConversationState) -> str:
    return _with_voice_rules(_apply_custom(state, "prompt_greeting", _default_greeting_prompt))


def qualify_prompt(state: ConversationState) -> str:
    return _with_voice_rules(_apply_custom(state, "prompt_qualify", _default_qualify_prompt))


def pitch_prompt(state: ConversationState) -> str:
    return _with_voice_rules(_apply_custom(state, "prompt_pitch", _default_pitch_prompt))


def objection_prompt(state: ConversationState) -> str:
    return _with_voice_rules(_apply_custom(state, "prompt_objection", _default_objection_prompt))


def booking_prompt(state: ConversationState) -> str:
    return _with_voice_rules(_apply_custom(state, "prompt_booking", _default_booking_prompt))


def wrap_up_prompt(state: ConversationState) -> str:
    return _with_voice_rules(_apply_custom(state, "prompt_wrapup", _default_wrap_up_prompt))


# ── Router prompts (unchanged, not per-bot customizable) ───────────

QUALIFY_ROUTER_PROMPT = """
You are deciding the next step after the prospect answered a qualification question.

- Return 'continue_qualifying' if there are still important unanswered questions
  (role/authority, pain points, budget, timeline) AND the prospect is still engaged.
  This is the default when the prospect gives a substantive answer but qualification
  is not yet complete.
  IMPORTANT: If the prospect says they are not the right person BUT offers to
  connect you with someone else or provides a referral, return 'continue_qualifying'
  so the agent can gather the referral details (name, role, contact info).
- Return 'pitch' if enough qualification info has been gathered (at least role and
  one pain point are known) AND the prospect sounds open or curious.
- Return 'not_interested' ONLY if the prospect clearly refuses or asks you to stop,
  using explicit phrases (e.g. not interested, no thanks, stop calling, don't call,
  remove me, we're not buying, wrong person). Do NOT use 'not_interested' for
  short affirmatives, garbled speech-to-text, or fragments that contain "yeah",
  "sure", "go ahead", "ok" without a clear refusal. Truncated audio often looks
  like random words; when in doubt, prefer 'continue_qualifying'.

Respond with only the label.
""".strip()


PITCH_ROUTER_PROMPT = """
Classify the prospect's latest reply after the pitch.
- Return 'book_meeting' if they want to schedule time.
- Return 'handle_objection' if they raise a concern but remain engaged.
- Return 'wrap_up' if they clearly decline.
Respond with only the label.
""".strip()


OBJECTION_ROUTER_PROMPT = """
Classify the prospect's reply after objection handling.
- Return 'pitch' if they are open to hearing more.
- Return 'wrap_up' if they are still declining or want to end the call.
Respond with only the label.
""".strip()


BOOKING_ROUTER_PROMPT = """
The prospect was asked to choose a meeting slot. Classify their reply.
- Return 'continue_booking' if they are trying to pick a slot or still engaged.
- Return 'wrap_up' if they want to end the call, are frustrated, or refuse to book.
Respond with only the label.
""".strip()


def qualification_extraction_prompt(existing_pain_points: list[str]) -> str:
    known = ", ".join(existing_pain_points) if existing_pain_points else "none yet"
    return (
        "You are analyzing a live sales qualification conversation. Based on EVERYTHING "
        "the prospect has said so far in the transcript, extract the following fields.\n\n"
        "Fields to extract:\n\n"
        "1. is_decision_maker (true / false / null)\n"
        "   - true: the prospect confirmed they make or directly influence purchasing\n"
        '     decisions (e.g. "I own that", "I\'m the VP of sales", "yes, it\'s my call").\n'
        "   - false: the prospect explicitly said they are NOT a decision-maker.\n"
        "   - null: not enough information to determine.\n\n"
        "2. budget_confirmed (true / false / null)\n"
        "   - true: the prospect indicated they have budget, spending authority, or\n"
        '     funding approved (e.g. "we have budget for this quarter", "already approved").\n'
        "   - false: the prospect said they have no budget or it was denied.\n"
        "   - null: not discussed or unclear.\n\n"
        "3. timeline (string / null)\n"
        "   - A short phrase capturing when the prospect needs a solution or is\n"
        '     evaluating (e.g. "this quarter", "next month", "before Q3").\n'
        "   - null: no timeline mentioned.\n\n"
        "4. pain_points (list of strings)\n"
        "   - Specific business problems or frustrations the prospect has expressed.\n"
        "   - Only include NEW pain points not already in the known list below.\n"
        '   - Use short, descriptive phrases (e.g. "slow lead follow-up",\n'
        '     "manual data entry", "reps missing hot leads").\n\n'
        f"Already known pain points: {known}\n\n"
        "Return ONLY valid JSON matching this exact schema, with no extra text:\n"
        '{"is_decision_maker": ..., "budget_confirmed": ..., "timeline": ..., "pain_points": [...]}'
    )
