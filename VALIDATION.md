# Validation Checklist

Use this checklist before enabling real outbound SDR calling.

## Local Readiness

1. Start Redis if you want Redis-backed Vocode call config storage.
2. Fill `.env` with either stub mode settings or real provider credentials.
3. Run:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn main:app --host 0.0.0.0 --port 3000
```

4. Confirm:

```powershell
Invoke-WebRequest http://127.0.0.1:3000/healthz
```

Expected:
- `status` is `ok` for a fully configured telephony environment, or `degraded` when only stub mode is active
- `provider_summary` reflects the current runtime mode

## Graph-Only SDR Flow

Validate the non-telephony flow first:

1. Start a session for `lead-001`.
2. Confirm the first response is the greeting.
3. Send a qualification answer such as `Yes, I run sales operations`.
4. Confirm the next turn asks qualification questions or advances to the pitch.
5. Raise an objection such as `We already have a process`.
6. Confirm the graph enters objection handling rather than ending immediately.
7. Confirm a booking phrase such as `Tomorrow at 3 PM works`.
8. Confirm the state marks the meeting as booked and the follow-up action is set.

## Stub Integrations

Check the in-memory adapters:

- A follow-up email is recorded after the wrap-up path.
- CRM updates are recorded with the final call outcome.
- Calendar booking is recorded when a slot is confirmed.
- Session state persists across multiple `/sessions/{conversation_id}/turns` requests.

## Outbound Telephony

When Twilio, Deepgram, and TTS credentials are configured:

1. Ensure `BASE_URL` is reachable over HTTPS.
2. Call:

```powershell
Invoke-WebRequest -Method POST http://127.0.0.1:3000/outbound/calls -ContentType "application/json" -Body '{"lead_id":"lead-001"}'
```

3. Confirm Twilio places the call and Vocode attaches to the conversation.
4. Confirm the prospect hears the SDR greeting and can interrupt naturally.
5. Confirm speech-to-text keeps pace with normal phone audio.
6. Confirm the SDR graph advances one stage per turn rather than dumping multiple stages at once.

## Call Outcome Cases

Validate each of these manually:

- Positive qualification leading to a booked meeting
- Mild objection leading to a handled objection and renewed pitch
- Hard decline leading to `not_interested`
- Ambiguous response leading to `follow_up_needed`

## Production Readiness

Only move to real prospect traffic when all of the following are true:

- graph behavior is stable in local API tests
- outbound telephony works against your own phone number
- booking, CRM, and email adapters are either production-ready or intentionally stubbed off
- transcripts, call outcome logging, and session cleanup behave as expected
- you have approved call scripts and compliance review for outbound AI calling
