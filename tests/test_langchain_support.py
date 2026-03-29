from vocode.streaming.models.events import Sender
from vocode.streaming.models.transcript import Message, Transcript

from vocode_contact_center.langchain_support import build_prompt_inputs


def test_build_prompt_inputs_compacts_older_history_into_summary():
    transcript = Transcript(
        event_logs=[
            Message(sender=Sender.HUMAN, text="I need help with my booking reference ABC123."),
            Message(sender=Sender.BOT, text="Sure, can you confirm the travel date?"),
            Message(sender=Sender.HUMAN, text="It is next Friday at 10 AM."),
            Message(sender=Sender.BOT, text="Thanks, I found the booking and can help update it."),
            Message(sender=Sender.HUMAN, text="Please move it to the afternoon if possible."),
        ]
    )

    prompt_inputs = build_prompt_inputs(
        transcript,
        call_context="Live call metadata:\n- Caller number: +1234567890",
        recent_message_limit=2,
        summary_max_messages=3,
        summary_max_chars=220,
    )

    assert len(prompt_inputs["chat_history"]) == 2
    assert prompt_inputs["chat_history"][0][0] == "ai"
    assert prompt_inputs["chat_history"][1][0] == "human"
    assert "Conversation summary:" in prompt_inputs["conversation_summary"]
    assert "booking reference ABC123" in prompt_inputs["conversation_summary"]
