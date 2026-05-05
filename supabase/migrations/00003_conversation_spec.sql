-- conversation_spec: versioned JSON for generic single-agent or multi-node graph (see ai_sdr_agent.graph.spec).
-- NULL = legacy SDR pipeline (same behavior as before migration).

ALTER TABLE public.bot_configs
  ADD COLUMN IF NOT EXISTS conversation_spec jsonb;

-- REPLACE cannot add a column in the middle of the select list; Postgres would
-- treat it as renaming columns. Drop and recreate instead.
DROP VIEW IF EXISTS public.bot_configs_safe;

CREATE VIEW public.bot_configs_safe AS
SELECT
  id, user_id, name, is_active,
  llm_provider, llm_model_name, llm_temperature, llm_max_tokens,
  CASE WHEN openai_api_key IS NOT NULL
       THEN left(openai_api_key, 4) || '****' || right(openai_api_key, 4)
  END AS openai_api_key,
  CASE WHEN anthropic_api_key IS NOT NULL
       THEN left(anthropic_api_key, 4) || '****' || right(anthropic_api_key, 4)
  END AS anthropic_api_key,
  CASE WHEN groq_api_key IS NOT NULL
       THEN left(groq_api_key, 4) || '****' || right(groq_api_key, 4)
  END AS groq_api_key,
  CASE WHEN elevenlabs_api_key IS NOT NULL
       THEN left(elevenlabs_api_key, 4) || '****' || right(elevenlabs_api_key, 4)
  END AS elevenlabs_api_key,
  elevenlabs_voice_id, elevenlabs_model_id,
  CASE WHEN deepgram_api_key IS NOT NULL
       THEN left(deepgram_api_key, 4) || '****' || right(deepgram_api_key, 4)
  END AS deepgram_api_key,
  deepgram_model, deepgram_language,
  CASE WHEN twilio_account_sid IS NOT NULL
       THEN left(twilio_account_sid, 4) || '****' || right(twilio_account_sid, 4)
  END AS twilio_account_sid,
  CASE WHEN twilio_auth_token IS NOT NULL
       THEN left(twilio_auth_token, 4) || '****' || right(twilio_auth_token, 4)
  END AS twilio_auth_token,
  twilio_phone_number,
  initial_greeting, max_call_turns, max_objection_attempts,
  max_qualify_attempts, max_booking_attempts, sales_rep_name,
  prompt_greeting, prompt_qualify, prompt_pitch,
  prompt_objection, prompt_booking, prompt_wrapup,
  conversation_spec,
  created_at, updated_at
FROM public.bot_configs;

GRANT SELECT ON public.bot_configs_safe TO authenticated;
GRANT SELECT ON public.bot_configs_safe TO service_role;
