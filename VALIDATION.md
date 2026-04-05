# Validation Checklist

Use this checklist before moving production traffic away from Dialogflow.

## Local Readiness

1. Start Redis.
2. Fill `.env` with real provider credentials.
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
- `status` is `ok`
- `public_base_url` is present
- `twilio_webhook_path` is `/inbound_call`

On Railway, use:

```text
https://YOUR-RAILWAY-DOMAIN/healthz
```

and copy `inbound_call_url` from the response.

## Twilio Routing

1. Open your Twilio SIP Domain configuration.
2. Set the incoming request URL to:

```text
https://YOUR-PUBLIC-HOST/inbound_call
```

3. Set the method to `HTTP POST`.
4. Keep Zadarma routing to the Twilio SIP Domain during the test window.
5. Call the Zadarma number and verify the request reaches the Vocode server.

## Call Quality

Test each of these live:

- Caller hears the ElevenLabs voice instead of Dialogflow TTS.
- The agent can be interrupted naturally during longer answers.
- The agent recovers after short pauses and does not talk over the caller excessively.
- A simple question gets a short, phone-friendly response.
- A longer support-style question produces a coherent multi-turn conversation.

## Fallback And Escalation

Test these phrases:

- `I want a human`
- `Can you transfer me to an agent?`
- `Let me speak to a representative`

Expected:
- The custom agent does not hallucinate tool access.
- It returns the fallback handoff message.
- If you later add a live transfer workflow, verify that path separately before enabling it.

## PeopleCert (CertyPal) smoke checks

When this deployment is configured for PeopleCert candidate support, spot-check:

- Opening matches the approved virtual-assistant disclosure (CertyPal / PeopleCert).
- Misroute wording for non-B&IT exam types is available in the system prompt; SELT/LanguageCert mentions route toward human contact.
- Sample scenarios: invalid voucher guidance, OLP stuck screen (refresh / restart), locked profile name (upload + support), results timeline, e-certificate location and hard-copy ordering.
- Document-backed answers only use the configured `INFORMATION_PRODUCTS_*` PDF; no invented policy.

## Production Cutover

Only cut over fully when all of the following are true:

- inbound calls consistently hit `/inbound_call`
- Twilio SIP Domain request URL points at the Vocode HTTPS endpoint
- transcripts and logs are being captured as expected
- ElevenLabs latency is acceptable for phone use
- fallback and escalation prompts behave predictably
- you have a rollback path back to the old Twilio/Dialogflow webhook
