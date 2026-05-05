from langchain_core.messages import AIMessage, HumanMessage

from ai_sdr_agent.services.brain import (
    _groq_messages_from_transcript,
    _messages_from_transcript,
)


def test_transcript_agent_turns_are_assistant_messages():
    transcript = [
        {"role": "agent", "content": "Hi, this is John."},
        {"role": "human", "content": "Give me a summary."},
    ]

    messages = _messages_from_transcript(transcript)

    assert isinstance(messages[0], AIMessage)
    assert messages[0].content == "Hi, this is John."
    assert isinstance(messages[1], HumanMessage)
    assert messages[1].content == "Give me a summary."


def test_groq_transcript_agent_turns_are_assistant_role():
    transcript = [
        {"role": "agent", "content": "Hi, this is John."},
        {"role": "human", "content": "Give me a summary."},
    ]

    messages = _groq_messages_from_transcript(system_prompt="System", transcript=transcript)

    assert messages == [
        {"role": "system", "content": "System"},
        {"role": "assistant", "content": "Hi, this is John."},
        {"role": "user", "content": "Give me a summary."},
    ]
