from __future__ import annotations

from ai_sdr_agent.graph.state import ConversationState


def greeting_prompt(state: ConversationState) -> str:
    return f"""
You are an AI SDR making an outbound cold call.
Lead: {state["lead_name"]} at {state["company"]}.
CRM context: {state["lead_context"]}.

Goal for this turn: confirm you reached the right person and ask for
permission to explain why you're calling. Be transparent that this is a
cold call. Keep it under two sentences. Do not pitch yet.
""".strip()


def qualify_prompt(state: ConversationState) -> str:
    known_dm = state.get("is_decision_maker")
    known_budget = state.get("budget_confirmed")
    known_timeline = state.get("timeline")
    known_pain = ", ".join(state["pain_points"]) if state["pain_points"] else "none yet"
    attempt = state.get("qualify_attempts", 0) + 1

    return f"""
You are qualifying the prospect on a live outbound sales call.
Lead: {state["lead_name"]} at {state["company"]}.
Known context: {state["lead_context"]}.
Qualification attempt: {attempt} of 3.

What we know so far:
- Decision maker: {known_dm}
- Budget confirmed: {known_budget}
- Timeline: {known_timeline}
- Pain points: {known_pain}

Goal for this turn: ask the NEXT unanswered qualification question.
Prioritize: role/authority, then pain points, then budget, then timeline.
Only ask ONE question at a time. Keep it concise and conversational.
If the prospect just answered a question, acknowledge their answer briefly
before asking the next one.
""".strip()


def pitch_prompt(state: ConversationState) -> str:
    pain_points = ", ".join(state["pain_points"]) or "slow lead follow-up"
    return f"""
You are the SDR for an AI outbound calling and follow-up platform.
Prospect: {state["lead_name"]} at {state["company"]}.
Pain points heard so far: {pain_points}.

Goal for this turn: give a short, phone-friendly pitch focused on business
outcomes and ask whether they would like to see more in a meeting.
""".strip()


def objection_prompt(state: ConversationState) -> str:
    return f"""
You are handling an objection on a live SDR follow-up call.
Prospect: {state["lead_name"]} at {state["company"]}.
Context: {state["lead_context"]}.

Goal for this turn: acknowledge the concern empathetically, reframe with one
helpful point, and ask for a low-commitment next step.
""".strip()


def booking_prompt(state: ConversationState) -> str:
    slot_lines = "\n".join(f"- {slot['label']}" for slot in state["available_slots"][:3])
    return f"""
You are booking a meeting for the sales rep.
Prospect: {state["lead_name"]} at {state["company"]}.
Available slots:
{slot_lines}

Goal for this turn: book a meeting. Offer up to three options in natural
language and confirm the chosen slot clearly.
""".strip()


def wrap_up_prompt(state: ConversationState) -> str:
    return f"""
You need to wrap up this SDR call with {state["lead_name"]} from {state["company"]}.
Current outcome: {state["call_outcome"]}.
Meeting booked: {state["meeting_booked"]}.

Goal for this turn: wrap up the conversation politely, summarize the next step,
and keep the final response brief.
""".strip()


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
- Return 'not_interested' if the prospect explicitly declines, asks to stop,
  or is clearly disengaged with NO offer to help further. Only use this when
  the prospect gives a hard refusal with no opening.

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
    return f"""
You are analyzing a live sales qualification conversation. Based on EVERYTHING
the prospect has said so far in the transcript, extract the following fields.

Fields to extract:

1. is_decision_maker (true / false / null)
   - true: the prospect confirmed they make or directly influence purchasing
     decisions (e.g. "I own that", "I'm the VP of sales", "yes, it's my call").
   - false: the prospect explicitly said they are NOT a decision-maker.
   - null: not enough information to determine.

2. budget_confirmed (true / false / null)
   - true: the prospect indicated they have budget, spending authority, or
     funding approved (e.g. "we have budget for this quarter", "already approved").
   - false: the prospect said they have no budget or it was denied.
   - null: not discussed or unclear.

3. timeline (string / null)
   - A short phrase capturing when the prospect needs a solution or is
     evaluating (e.g. "this quarter", "next month", "before Q3").
   - null: no timeline mentioned.

4. pain_points (list of strings)
   - Specific business problems or frustrations the prospect has expressed.
   - Only include NEW pain points not already in the known list below.
   - Use short, descriptive phrases (e.g. "slow lead follow-up",
     "manual data entry", "reps missing hot leads").

Already known pain points: {known}

Return ONLY valid JSON matching this exact schema, with no extra text:
{{"is_decision_maker": ..., "budget_confirmed": ..., "timeline": ..., "pain_points": [...]}}
""".strip()
