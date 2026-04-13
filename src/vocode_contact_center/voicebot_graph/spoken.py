from __future__ import annotations

from vocode_contact_center.voicebot_graph.state import VoicebotGraphState


def spoken_preview_from_state(state: VoicebotGraphState) -> str:
    """Best-effort user-facing text available at the current graph step (for streaming TTS)."""
    if state.get("response_text"):
        return str(state["response_text"])
    if state.get("pending_prompt"):
        return str(state["pending_prompt"])
    if state.get("response_prefix"):
        return str(state["response_prefix"])
    return ""
