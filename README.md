# Voice Agent

Configurable voice agents with a FastAPI backend, Supabase persistence, and a React flow builder. Conversations run on custom LangGraph flows (single-prompt or multi-node graphs) with optional knowledge-base tool calling during live calls.

## What this includes

- FastAPI app with Supabase-authenticated bot, lead, test-session, and voice WebSocket APIs
- Custom conversation flows stored in `bot_configs.conversation_spec`
- Browser voice via Deepgram STT + ElevenLabs TTS, or OpenAI Realtime
- Knowledge base retrieval exposed to the LLM as `lookup_knowledge` during graph turns
- React UI under `vocal-frontend/` (Flow Builder, agents, call history, tools)

## Architecture

```mermaid
flowchart LR
  ui[vocal_frontend] --> api[FastAPI]
  api --> graph[LangGraph_custom_flow]
  graph --> brain[Groq_LLM]
  graph --> kb[Knowledge_RAG]
  api --> db[Supabase_Postgres]
  ui --> voice[WebSocket_voice]
  voice --> graph
```

## Install

Use Python `3.11`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configure

Copy `.env.example` to `.env` and set at minimum:

- `DATABASE_URL` — Supabase Postgres pooler URL
- `SUPABASE_URL` and `SUPABASE_JWT_SECRET`
- `LLM_PROVIDER=groq` and `GROQ_API_KEY` (or `stub` for offline tests)
- `DEEPGRAM_API_KEY`, `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID` for builtin browser voice

Apply Supabase migrations under `supabase/migrations/` (including `00021_remove_sdr_legacy.sql`).

## Run API

```bash
uvicorn main:app --host 0.0.0.0 --port 3000
```

## Run frontend

```bash
cd vocal-frontend
npm install
npm run dev
```

## Agent setup

Every agent must have a `conversation_spec` with `template: "custom"` and either:

- `mode: "single"` plus `system_prompt`, or
- `mode: "graph"` plus `nodes`, `edges`, and `entry_node_id`

Configure flows in the UI at `/agents/:id`. Legacy Classic SDR template agents are removed by migration `00021`.

## Tests

```bash
pytest tests/
```
