from __future__ import annotations

import re

from ai_sdr_agent.graph.state import ConversationState

# Appended to every spoken-response system prompt (defaults and custom).
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
        "pain_points": ", ".join(state["pain_points"]) or "κανένα ακόμα",
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
        f"Είσαι AI SDR που κάνει εξερχόμενη «κρύα» κλήση.\n"
        f"Πελάτης: {state['lead_name']} στην εταιρεία {state['company']}.\n"
        f"Πλαίσιο CRM: {state['lead_context']}.\n\n"
        f"Στόχος αυτής της στροφής: επιβεβαίωσε ότι μίλησες με το σωστό πρόσωπο και ζήτησε "
        f"άδεια να εξηγήσεις γιατί καλείς. Να είσαι ειλικρινής ότι είναι κρύα κλήση. "
        f"Μην κάνεις pitch ακόμα."
    )


def _default_qualify_prompt(state: ConversationState) -> str:
    known_dm = state.get("is_decision_maker")
    known_budget = state.get("budget_confirmed")
    known_timeline = state.get("timeline")
    known_pain = ", ".join(state["pain_points"]) if state["pain_points"] else "κανένα ακόμα"
    attempt = state.get("qualify_attempts", 0) + 1
    max_attempts = state.get("bot_config", {}).get("max_qualify_attempts", 3)

    return (
        f"Κάνεις πιστοποίηση (qualify) τον πελάτη σε ζωντανή εξερχόμενη κλήση πωλήσεων.\n"
        f"Πελάτης: {state['lead_name']} στην εταιρεία {state['company']}.\n"
        f"Γνωστό πλαίσιο: {state['lead_context']}.\n"
        f"Προσπάθεια πιστοποίησης: {attempt} από {max_attempts}.\n\n"
        f"Τι ξέρουμε μέχρι τώρα:\n"
        f"- Αποφασίζων / ρόλος: {known_dm}\n"
        f"- Επιβεβαιωμένο budget: {known_budget}\n"
        f"- Χρονοδιάγραμμα: {known_timeline}\n"
        f"- Σημεία πόνου: {known_pain}\n\n"
        f"Στόχος αυτής της στροφής: κάνε την ΕΠΟΜΕΝΗ ερώτηση πιστοποίησης που δεν έχει απαντηθεί.\n"
        f"Προτεραιότητα: ρόλος/εξουσία, μετά σημεία πόνου, μετά budget, μετά χρονοδιάγραμμα.\n"
        f"Μία ερώτηση τη φορά. Αν μόλις απάντησε, αναγνώρισε σε λίγες λέξεις και ρώτα την επόμενη "
        f"ερώτηση στην ίδια προφορική απάντηση."
    )


def _default_pitch_prompt(state: ConversationState) -> str:
    pain_points = ", ".join(state["pain_points"]) or "αργή παρακολούθηση leads"
    return (
        f"Είσαι ο SDR για πλατφόρμα AI εξερχόμενων κλήσεων και follow-up.\n"
        f"Πελάτης: {state['lead_name']} στην εταιρεία {state['company']}.\n"
        f"Σημεία πόνου που έχουν ακουστεί: {pain_points}.\n\n"
        f"Στόχος αυτής της στροφής: ένα σφιχτό pitch με επίκεντρο επιχειρηματικά αποτελέσματα, "
        f"μετά μία ερώτηση για επόμενο βήμα. Χωρίς λίστα χαρακτηριστικών."
    )


def _default_objection_prompt(state: ConversationState) -> str:
    return (
        f"Χειρίζεσαι ένσταση σε ζωντανή κλήση SDR follow-up.\n"
        f"Πελάτης: {state['lead_name']} στην εταιρεία {state['company']}.\n"
        f"Πλαίσιο: {state['lead_context']}.\n\n"
        f"Στόχος αυτής της στροφής: αναγνώριση, ένα χρήσιμο reframe, ένα αίτημα χαμηλής δέσμευσης — "
        f"συνολικά σε μία ή δύο προτάσεις."
    )


def _default_booking_prompt(state: ConversationState) -> str:
    slot_lines = "\n".join(f"- {slot['label']}" for slot in state["available_slots"][:3])
    return (
        f"Κλείνεις ραντεβού για τον πωλητή.\n"
        f"Πελάτης: {state['lead_name']} στην εταιρεία {state['company']}.\n"
        f"Διαθέσιμα slots:\n{slot_lines}\n\n"
        f"Στόχος αυτής της στροφής: κλείσε ραντεβού. Δώσε έως τρεις επιλογές ώρας σε απλή "
        f"προφορική γλώσσα (όχι λίστες)· όταν επιβεβαιώνεις, επανάλαβε την επιλεγμένη ώρα σε μία σύντομη γραμμή."
    )


def _default_wrap_up_prompt(state: ConversationState) -> str:
    return (
        f"Κλείνεις αυτή την κλήση SDR με τον/την {state['lead_name']} από την {state['company']}.\n"
        f"Τρέχον αποτέλεσμα: {state['call_outcome']}.\n"
        f"Ραντεβού κλεισμένο: {state['meeting_booked']}.\n\n"
        f"Στόχος αυτής της στροφής: ευγενικό κλείσιμο, μία πρόταση για τα επόμενα βήματα, τέλος."
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


# ── Router prompts (labels stay English for the parser; body in Greek) ─

QUALIFY_ROUTER_PROMPT = """
Αποφασίζεις το επόμενο βήμα αφού ο πελάτης απάντησε σε ερώτηση πιστοποίησης.

- Επίστρεψε continue_qualifying αν υπάρχουν ακόμα σημαντικές αναπάντητες ερωτήσεις
  (ρόλος/εξουσία, σημεία πόνου, budget, χρονοδιάγραμμα) ΚΑΙ ο πελάτης παραμένει ενεργός.
  Αυτό είναι το προεπιλεγμένο όταν η απάντηση είναι ουσιαστική αλλά η πιστοποίηση δεν έχει ολοκληρωθεί.
  ΣΗΜΑΝΤΙΚΟ: Αν ο πελάτης λέει ότι δεν είναι το σωστό πρόσωπο ΑΛΛΑ προσφέρεται να σε συνδέσει
  με άλλον ή δίνει παραπομπή, επίστρεψε continue_qualifying ώστε να συλλεχθούν στοιχεία παραπομπής.
- Επίστρεψε pitch αν έχουν συλλεχθεί αρκετά στοιχεία (τουλάχιστον ρόλος και ένα σημείο πόνου)
  ΚΑΙ ο πελάτης φαίνεται ανοιχτός ή περίεργος.
- Επίστρεψε not_interested ΜΟΝΟ αν ο πελάτης αρνείται ξεκάθαρα ή ζητά να σταματήσει η κλήση,
  με ρητές φράσεις (π.χ. δεν ενδιαφέρομαι, όχι ευχαριστώ, μην ξανακαλέσετε, σταματήστε,
  διαγραφή, δεν αγοράζουμε, λάθος άτομο). ΜΗΝ χρησιμοποιείς not_interested για σύντομα καταφατικά,
  θόρυβο ομιλίας-σε-κείμενο ή θραύσματα με «ναι», «οκ», «πάμε» χωρίς ξεκάθαρη άρνηση.
  Όταν αμφιβάλλεις, προτίμησε continue_qualifying.

Απάντησε ΜΟΝΟ με μία από τις ακριβείς ετικέτες (λατινικοί χαρακτήρες, χωρίς εισαγωγικά): continue_qualifying ή pitch ή not_interested
""".strip()


PITCH_ROUTER_PROMPT = """
Ταξινόμησε την τελευταία απάντηση του πελάτη μετά το pitch.
- Επίστρεψε book_meeting αν θέλει να οριστεί χρόνος.
- Επίστρεψε handle_objection αν εγείρει ανησυχία αλλά παραμένει ενεργός.
- Επίστρεψε wrap_up αν αρνείται ξεκάθαρα.

Απάντησε ΜΟΝΟ με μία ετικέτα: book_meeting ή handle_objection ή wrap_up
""".strip()


OBJECTION_ROUTER_PROMPT = """
Ταξινόμησε την απάντηση του πελάτη μετά τη διαχείριση ένστασης.
- Επίστρεψε pitch αν είναι ανοιχτός να ακούσει περισσότερα.
- Επίστρεψε wrap_up αν εξακολουθεί να αρνείται ή θέλει να τελειώσει η κλήση.

Απάντησε ΜΟΝΟ με μία ετικέτα: pitch ή wrap_up
""".strip()


BOOKING_ROUTER_PROMPT = """
Ο πελάτης κλήθηκε να διαλέξει slot ραντεβού. Ταξινόμησε την απάντησή του.
- Επίστρεψε continue_booking αν προσπαθεί να διαλέξει slot ή παραμένει ενεργός.
- Επίστρεψε wrap_up αν θέλει να τελειώσει την κλήση, είναι εκνευρισμένος ή αρνείται το κλείσιμο.

Απάντησε ΜΟΝΟ με μία ετικέτα: continue_booking ή wrap_up
""".strip()


def qualification_extraction_prompt(existing_pain_points: list[str]) -> str:
    known = ", ".join(existing_pain_points) if existing_pain_points else "κανένα ακόμα"
    return (
        "Αναλύεις ζωντανή συνομιλία πιστοποίησης πωλήσεων. Με βάση ΟΛΑ όσα έχει πει ο πελάτης "
        "στο transcript, εξήγαγε τα παρακάτω πεδία.\n\n"
        "Πεδία προς εξαγωγή:\n\n"
        "1. is_decision_maker (true / false / null)\n"
        "   - true: ο πελάτης επιβεβαίωσε ότι αποφασίζει ή επηρεάζει άμεσα αγορές\n"
        '     (π.χ. «εγώ αποφασίζω», «είμαι υπεύθυνος πωλήσεων», «ναι, δική μου απόφαση»).\n'
        "   - false: ο πελάτης είπε ρητά ότι ΔΕΝ είναι αποφασίζων.\n"
        "   - null: ανεπαρκή στοιχεία.\n\n"
        "2. budget_confirmed (true / false / null)\n"
        "   - true: έδειξε budget, εξουσία δαπανών ή έγκριση χρηματοδότησης\n"
        '     (π.χ. «έχουμε budget αυτό το τρίμηνο», «έχει εγκριθεί»).\n'
        "   - false: είπε ότι δεν υπάρχει budget ή απορρίφθηκε.\n"
        "   - null: δεν συζητήθηκε ή είναι ασαφές.\n\n"
        "3. timeline (string / null)\n"
        "   - Σύντομη φράση για πότε χρειάζεται λύση ή αξιολόγηση\n"
        '     (π.χ. «αυτό το τρίμηνο», «τον επόμενο μήνα», «πριν το Q3»).\n'
        "   - null: δεν αναφέρθηκε χρονοδιάγραμμα.\n\n"
        "4. pain_points (λίστα συμβολοσειρών)\n"
        "   - Συγκεκριμένα επιχειρηματικά προβλήματα ή εκνευρισμοί που εξέφρασε ο πελάτης.\n"
        "   - Μόνο ΝΕΑ σημεία πόνου που δεν υπάρχουν ήδη στη γνωστή λίστα παρακάτω.\n"
        '   - Σύντομες περιγραφικές φράσεις (π.χ. «αργό follow-up leads»,\n'
        '     «χειροκίνητη εισαγωγή δεδομένων», «χάνονται ζεστά leads»).\n\n'
        f"Ήδη γνωστά σημεία πόνου: {known}\n\n"
        "Επίστρεψε ΜΟΝΟ έγκυρο JSON που ταιριάζει ακριβώς στο σχήμα, χωρίς επιπλέον κείμενο:\n"
        '{"is_decision_maker": ..., "budget_confirmed": ..., "timeline": ..., "pain_points": [...]}'
    )
