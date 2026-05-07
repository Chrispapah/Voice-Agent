-- Polished copy for the legacy Banking Agent graph (original opaque node ids and edges).
-- Prompts, labels, and classify_hint tuned for voice; typo fixes; closure node wired to complete.

UPDATE public.bot_configs
SET
  conversation_spec = $spec$
{
  "conversation_spec_version": 1,
  "mode": "graph",
  "template": "custom",
  "entry_node_id": "welcome",
  "nodes": [
    {
      "id": "welcome",
      "label": "Welcome — intent routing",
      "tool_ids": [],
      "system_prompt": "You are a professional banking voice assistant on a live call. Sound warm and clear. In one or two short sentences, steer the caller toward the right next step. Do not collect sensitive details here; route verification and card cases to the appropriate step. If they only greeted or were vague, ask one brief clarifying question.",
      "static_message": "Hello, I'm your banking voice assistant. How can I help you today?",
      "reply_turn_modes": ["static", "llm"],
      "loop_min_turns": 1,
      "loop_max_turns": 4,
      "classify_hint": "Pick the destination id exactly.\nwelcome — Stay only if intent is still unclear or you need one short clarifying question.\nfadfaedef — Stolen or lost card, fraud, unauthorized charges, balances, transfers, loans, or any request that needs to verify who they are before helping.\nDWaqasd — Branch or ATM locations, hours, or directions at a general level.\nfdqewfEW — General products, services, or rates overview without accessing their specific account."
    },
    {
      "id": "fadfaedef",
      "label": "Verify identity",
      "tool_ids": [],
      "system_prompt": "Verify the caller before sensitive help. Ask for the phone number on file or a callback number, one question at a time. Speak in short sentences suitable for voice. Do not claim you looked up their account in real time. Never ask for full card numbers. If they refuse verification, stay calm: offer generic safety guidance or a specialist handoff in one or two sentences.",
      "static_message": "To help you securely, please share the best phone number to reach you on this profile.",
      "reply_turn_modes": ["static", "llm"],
      "loop_min_turns": 1,
      "loop_max_turns": 6,
      "classify_hint": "Pick the destination id exactly.\nfadfaedef — Stay only if they still have not given a usable phone or callback number.\nFSDAfsd — They want to schedule a branch or phone appointment.\nqualify — Credit or debit card theft, fraud, dispute, or replacing a card for security.\ndwsWDAWSD — Balances, transfers, payments, or everyday money movement after verification or while staying in this lane."
    },
    {
      "id": "qualify",
      "label": "Card theft or dispute",
      "tool_ids": [],
      "system_prompt": "You assist with suspected card theft or disputes. Stay calm and practical. Ask one clear question at a time. Give short guidance: securing the card, dispute paths, and timelines in plain language. Do not play investigator; avoid long interrogations. Keep each reply under three short sentences.",
      "loop_min_turns": 1,
      "loop_max_turns": 4,
      "classify_hint": "qualify — Continue card theft or dispute assistance.\ndqwdqwdas — They need something else or are done with this topic."
    },
    {
      "id": "FSDAfsd",
      "label": "Book an appointment",
      "tool_ids": [],
      "system_prompt": "Help schedule an in-branch or phone appointment. Clarify what they need in one question if helpful. Describe typical next steps without inventing specific calendar slots; offer confirmation through the app or a banker when appropriate.",
      "loop_min_turns": 1,
      "loop_max_turns": 4,
      "classify_hint": "FSDAfsd — Continue scheduling or answering appointment questions.\ndqwdqwdas — They want general help or are finished with scheduling."
    },
    {
      "id": "dwsWDAWSD",
      "label": "Transactions and payments",
      "tool_ids": [],
      "system_prompt": "Discuss balances, transfers, and payments at a high level. Point to mobile or online banking for exact figures and execution. Do not state live balances or move money from this assistant. Keep replies brief and voice-friendly.",
      "loop_min_turns": 1,
      "loop_max_turns": 4,
      "classify_hint": "dwsWDAWSD — Stay on transfers, payments, or balance questions at a general level.\ndqwdqwdas — Topic changes or they need something else."
    },
    {
      "id": "DWaqasd",
      "label": "Branch and ATM locations",
      "tool_ids": [],
      "system_prompt": "Help find branches or ATMs. If you need a location, ask for city or ZIP in one short question. Do not invent addresses; offer app search or secure follow-up for precise listings.",
      "loop_min_turns": 1,
      "loop_max_turns": 3,
      "classify_hint": "DWaqasd — Continue narrowing location or hours.\ndqwdqwdas — They move on or need another topic."
    },
    {
      "id": "fdqewfEW",
      "label": "Products and rates (general)",
      "tool_ids": [],
      "system_prompt": "Answer general product and rate questions in plain language. Do not guarantee approval or quote binding rates without proper channels. Offer a specialist for specifics when needed.",
      "loop_min_turns": 1,
      "loop_max_turns": 4,
      "classify_hint": "fdqewfEW — Continue general product or rate discussion.\ndqwdqwdas — Different request or they are done."
    },
    {
      "id": "dqwdqwdas",
      "label": "Anything else",
      "tool_ids": [],
      "system_prompt": "Ask if there is anything else you can help with today in one concise sentence. If they indicate they are finished, acknowledge and move toward closing without adding new tasks.",
      "loop_min_turns": 1,
      "loop_max_turns": 3,
      "classify_hint": "Pick the destination id exactly.\ndqwdqwdas — Stay only if you still need one clarification about what they need next.\ndqwaasdasd — They are done, thank you, or want to end the call.\nDWaqasd — They switch to branch or ATM help.\nfadfaedef — They need verification or raise fraud, stolen card, or account-specific needs.\nfdqewfEW — They want general product or rate information."
    },
    {
      "id": "dqwaasdasd",
      "label": "Goodbye",
      "tool_ids": [],
      "system_prompt": "Thank them for calling, wish them well, and close politely in one or two short sentences. Do not introduce new tasks or questions.",
      "reply_turn_modes": ["llm"]
    }
  ],
  "edges": [
    { "from": "FSDAfsd", "to": "FSDAfsd" },
    { "from": "FSDAfsd", "to": "dqwdqwdas" },
    { "from": "qualify", "to": "qualify" },
    { "from": "qualify", "to": "dqwdqwdas" },
    { "from": "dqwdqwdas", "to": "dqwdqwdas" },
    { "from": "dqwdqwdas", "to": "dqwaasdasd" },
    { "from": "dqwdqwdas", "to": "DWaqasd" },
    { "from": "dqwdqwdas", "to": "fadfaedef" },
    { "from": "dqwdqwdas", "to": "fdqewfEW" },
    { "from": "welcome", "to": "fadfaedef" },
    { "from": "welcome", "to": "DWaqasd" },
    { "from": "welcome", "to": "fdqewfEW" },
    { "from": "welcome", "to": "welcome" },
    { "from": "fadfaedef", "to": "FSDAfsd" },
    { "from": "fadfaedef", "to": "qualify" },
    { "from": "fadfaedef", "to": "dwsWDAWSD" },
    { "from": "fadfaedef", "to": "fadfaedef" },
    { "from": "dwsWDAWSD", "to": "dwsWDAWSD" },
    { "from": "dwsWDAWSD", "to": "dqwdqwdas" },
    { "from": "DWaqasd", "to": "DWaqasd" },
    { "from": "DWaqasd", "to": "dqwdqwdas" },
    { "from": "fdqewfEW", "to": "fdqewfEW" },
    { "from": "fdqewfEW", "to": "dqwdqwdas" },
    { "from": "dqwaasdasd", "to": "complete" }
  ],
  "positions": {
    "welcome": { "x": 80, "y": 200 },
    "fadfaedef": { "x": 320, "y": 120 },
    "qualify": { "x": 520, "y": 0 },
    "FSDAfsd": { "x": 520, "y": 120 },
    "dwsWDAWSD": { "x": 520, "y": 240 },
    "DWaqasd": { "x": 320, "y": 320 },
    "fdqewfEW": { "x": 320, "y": 440 },
    "dqwdqwdas": { "x": 720, "y": 200 },
    "dqwaasdasd": { "x": 920, "y": 200 }
  }
}
$spec$::jsonb,
  updated_at = now()
WHERE id = '067ab5eb-b9b8-4003-9101-ad3de067f95a'::uuid;
