-- Remove legacy SDR-template bots and unused per-stage prompt columns.

DELETE FROM public.bot_configs
WHERE conversation_spec IS NULL
   OR conversation_spec->>'template' = 'sdr';

-- Drop the view first; Postgres blocks DROP COLUMN while bot_configs_safe depends on it.
DROP VIEW IF EXISTS public.bot_configs_safe;

ALTER TABLE public.bot_configs
  DROP COLUMN IF EXISTS prompt_greeting,
  DROP COLUMN IF EXISTS prompt_qualify,
  DROP COLUMN IF EXISTS prompt_pitch,
  DROP COLUMN IF EXISTS prompt_objection,
  DROP COLUMN IF EXISTS prompt_booking,
  DROP COLUMN IF EXISTS prompt_wrapup,
  DROP COLUMN IF EXISTS max_objection_attempts,
  DROP COLUMN IF EXISTS max_qualify_attempts,
  DROP COLUMN IF EXISTS max_booking_attempts;

CREATE VIEW public.bot_configs_safe AS
SELECT
  id, user_id, folder_id, name, is_active,
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
  voice_provider, openai_realtime_model, openai_realtime_voice,
  openai_realtime_instructions, allow_voice_interruptions,
  CASE WHEN twilio_account_sid IS NOT NULL
       THEN left(twilio_account_sid, 4) || '****' || right(twilio_account_sid, 4)
  END AS twilio_account_sid,
  CASE WHEN twilio_auth_token IS NOT NULL
       THEN left(twilio_auth_token, 4) || '****' || right(twilio_auth_token, 4)
  END AS twilio_auth_token,
  twilio_phone_number,
  max_call_turns, sales_rep_name,
  conversation_spec,
  kb_match_count, kb_min_similarity, kb_embedding_model,
  kb_max_context_chars, kb_max_tool_iterations,
  created_at, updated_at
FROM public.bot_configs;

GRANT SELECT ON public.bot_configs_safe TO authenticated;
GRANT SELECT ON public.bot_configs_safe TO service_role;
