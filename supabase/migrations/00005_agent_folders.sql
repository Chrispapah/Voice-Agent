-- User-created folders for organizing agents in the Agents sidebar.

CREATE TABLE IF NOT EXISTS public.agent_folders (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  name varchar(200) NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE public.bot_configs
  ADD COLUMN IF NOT EXISTS folder_id uuid REFERENCES public.agent_folders(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_agent_folders_user_id ON public.agent_folders(user_id);
CREATE INDEX IF NOT EXISTS idx_bot_configs_folder_id ON public.bot_configs(folder_id);

DROP TRIGGER IF EXISTS agent_folders_updated_at ON public.agent_folders;
CREATE TRIGGER agent_folders_updated_at
  BEFORE UPDATE ON public.agent_folders
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

ALTER TABLE public.agent_folders ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "users crud own folders" ON public.agent_folders;
CREATE POLICY "users crud own folders"
  ON public.agent_folders FOR ALL
  USING (user_id = auth.uid())
  WITH CHECK (user_id = auth.uid());

DROP VIEW IF EXISTS public.bot_configs_safe;

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

GRANT SELECT, INSERT, UPDATE, DELETE ON public.agent_folders TO authenticated;
GRANT ALL ON public.agent_folders TO service_role;
GRANT SELECT ON public.bot_configs_safe TO authenticated;
GRANT SELECT ON public.bot_configs_safe TO service_role;
