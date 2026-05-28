from __future__ import annotations

import re

from ai_sdr_agent.graph.state import ConversationState

# Appended to every spoken-response system prompt (custom agents).
_VOICE_OUTPUT_RULES = """
---
Φωνητική έξοδος (υποχρεωτική — διαβάζεται από TTS σε μία ροή):
- Απάντα μόνο σε φυσικά, σύγχρονα ελληνικά· μόνο ελληνικό αλφάβητο και στίξη (και αριθμοί όπου χρειάζεται). Χωρίς ξένα αλφάβητα· λατινικά μόνο για ονόματα brand, email ή URL όπου είναι απαραίτητο.
- Στις περισσότερες στροφές χρησιμοποίησε μία ή δύο σύντομες προτάσεις. Το πολύ τρεις πολύ σύντομες προτάσεις μόνο όταν δίνεις επιλογές ωραρίου ραντεβού.
- Χωρίς παραγράφους, κουκίδες, markdown ή αριθμημένες λίστες.
- Χωρίς αλλαγές γραμμής· μία συνεχής γραμμή ομιλίας (κόμματα και τελείες επιτρέπονται).
- Μην «κλείνεις» εσύ την κλήση: απέφυγε αποχαιρετισμούς, «κλείνω», «πρέπει να φύγω» ή προσποίηση τέλους κλήσης, εκτός αν ο χρήστης ξεκάθαρα θέλει να σταματήσει. Μείνε στο θέμα του κόμβου μέχρι να αλλάξει το routing.
"""


def format_reply_for_tts(text: str) -> str:
    """Collapse newlines and extra whitespace so TTS sends one synthesis string."""
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
        "pain_points": ", ".join(state["pain_points"]) or "κανένα ακόμα",
        "call_outcome": state["call_outcome"],
        "meeting_booked": str(state["meeting_booked"]),
        "sales_rep_name": state.get("bot_config", {}).get("sales_rep_name", "Sales Team"),
    }
