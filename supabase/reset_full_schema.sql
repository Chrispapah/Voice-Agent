-- =============================================================================
-- FULL RESET: drops Voice-Agent app tables + view + triggers/functions, then
-- recreates schema from migrations 00001 + 00002 + 00003 + 00004.
--
-- DANGER: Permanently deletes all rows in profiles, bot_configs, leads,
-- agent_tools, call_logs, sessions. Does NOT delete auth.users, but existing users will
-- NOT get new profile rows until you insert them or they sign up again
-- (the signup trigger only runs on INSERT into auth.users).
--
-- Run in Supabase Dashboard → SQL → New query, OR:
--   supabase db execute --file supabase/reset_full_schema.sql --linked
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Tear down (order matters for FKs and auth trigger)
-- ---------------------------------------------------------------------------
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;

DROP VIEW IF EXISTS public.bot_configs_safe CASCADE;

DROP TABLE IF EXISTS public.sessions CASCADE;
DROP TABLE IF EXISTS public.call_logs CASCADE;
DROP TABLE IF EXISTS public.leads CASCADE;
DROP TABLE IF EXISTS public.agent_tools CASCADE;
DROP TABLE IF EXISTS public.bot_configs CASCADE;
DROP TABLE IF EXISTS public.agent_folders CASCADE;
DROP TABLE IF EXISTS public.profiles CASCADE;

DROP FUNCTION IF EXISTS public.handle_new_user() CASCADE;
DROP FUNCTION IF EXISTS public.update_updated_at() CASCADE;

-- ==========================================================================
-- 00001_initial_schema.sql (verbatim)
-- ==========================================================================

-- --------------------------------------------------------------------------
-- Helper: auto-update updated_at on row modification
-- --------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION public.update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- --------------------------------------------------------------------------
-- profiles  (extends auth.users with app-specific fields)
-- --------------------------------------------------------------------------
CREATE TABLE public.profiles (
  id          uuid PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
  display_name text NOT NULL DEFAULT '',
  created_at  timestamptz NOT NULL DEFAULT now(),
  updated_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TRIGGER profiles_updated_at
  BEFORE UPDATE ON public.profiles
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

CREATE POLICY "users read own profile"
  ON public.profiles FOR SELECT
  USING (auth.uid() = id);

CREATE POLICY "users update own profile"
  ON public.profiles FOR UPDATE
  USING (auth.uid() = id);

CREATE POLICY "users insert own profile"
  ON public.profiles FOR INSERT
  WITH CHECK (auth.uid() = id);

-- Auto-create a profile row when a new user signs up
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO public.profiles (id, display_name)
  VALUES (NEW.id, COALESCE(NEW.raw_user_meta_data ->> 'display_name', ''));
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- --------------------------------------------------------------------------
-- agent_folders
-- --------------------------------------------------------------------------
CREATE TABLE public.agent_folders (
  id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id     uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  name        varchar(200) NOT NULL,
  created_at  timestamptz NOT NULL DEFAULT now(),
  updated_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX idx_agent_folders_user_id ON public.agent_folders(user_id);

CREATE TRIGGER agent_folders_updated_at
  BEFORE UPDATE ON public.agent_folders
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

ALTER TABLE public.agent_folders ENABLE ROW LEVEL SECURITY;

CREATE POLICY "users crud own folders"
  ON public.agent_folders FOR ALL
  USING (user_id = auth.uid())
  WITH CHECK (user_id = auth.uid());

-- --------------------------------------------------------------------------
-- bot_configs
-- --------------------------------------------------------------------------
CREATE TABLE public.bot_configs (
  id                    uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id               uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  folder_id             uuid REFERENCES public.agent_folders(id) ON DELETE SET NULL,
  name                  varchar(200) NOT NULL DEFAULT 'My Bot',
  is_active             boolean NOT NULL DEFAULT true,

  -- LLM
  llm_provider          varchar(20)  NOT NULL DEFAULT 'openai',
  llm_model_name        varchar(100) NOT NULL DEFAULT 'gpt-4o-mini',
  llm_temperature       double precision NOT NULL DEFAULT 0.4,
  llm_max_tokens        integer NOT NULL DEFAULT 300,
  openai_api_key        text,
  anthropic_api_key     text,
  groq_api_key          text,

  -- TTS - ElevenLabs
  elevenlabs_api_key    text,
  elevenlabs_voice_id   varchar(100),
  elevenlabs_model_id   varchar(100) NOT NULL DEFAULT 'eleven_turbo_v2',

  -- STT - Deepgram
  deepgram_api_key      text,
  deepgram_model        varchar(50)  NOT NULL DEFAULT 'nova-2',
  deepgram_language     varchar(10)  NOT NULL DEFAULT 'en-US',

  -- Telephony - Twilio
  twilio_account_sid    text,
  twilio_auth_token     text,
  twilio_phone_number   varchar(30),

  -- Conversation behaviour
  initial_greeting      text NOT NULL DEFAULT 'Hi, this is John — I know I''m calling out of the blue. Do you have 30 seconds so I can tell you why I''m reaching out?',
  max_call_turns        integer NOT NULL DEFAULT 12,
  max_objection_attempts integer NOT NULL DEFAULT 2,
  max_qualify_attempts  integer NOT NULL DEFAULT 3,
  max_booking_attempts  integer NOT NULL DEFAULT 3,
  sales_rep_name        varchar(200) NOT NULL DEFAULT 'Sales Team',

  -- Custom prompts (nullable = use defaults)
  prompt_greeting       text,
  prompt_qualify        text,
  prompt_pitch          text,
  prompt_objection      text,
  prompt_booking        text,
  prompt_wrapup         text,

  created_at            timestamptz NOT NULL DEFAULT now(),
  updated_at            timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX idx_bot_configs_user_id ON public.bot_configs(user_id);
CREATE INDEX idx_bot_configs_folder_id ON public.bot_configs(folder_id);

CREATE TRIGGER bot_configs_updated_at
  BEFORE UPDATE ON public.bot_configs
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

ALTER TABLE public.bot_configs ENABLE ROW LEVEL SECURITY;

CREATE POLICY "users crud own bots"
  ON public.bot_configs FOR ALL
  USING (user_id = auth.uid())
  WITH CHECK (user_id = auth.uid());

-- Service role (Railway backend) bypasses RLS automatically.

-- --------------------------------------------------------------------------
-- agent_tools
-- --------------------------------------------------------------------------
CREATE TABLE public.agent_tools (
  id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id     uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  name        varchar(200) NOT NULL,
  description text NOT NULL DEFAULT '',
  kind        varchar(40) NOT NULL DEFAULT 'http',
  config_json jsonb NOT NULL DEFAULT '{}'::jsonb,
  is_active   boolean NOT NULL DEFAULT true,
  created_at  timestamptz NOT NULL DEFAULT now(),
  updated_at  timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX idx_agent_tools_user_id ON public.agent_tools(user_id);

CREATE TRIGGER agent_tools_updated_at
  BEFORE UPDATE ON public.agent_tools
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

ALTER TABLE public.agent_tools ENABLE ROW LEVEL SECURITY;

CREATE POLICY "users crud own tools"
  ON public.agent_tools FOR ALL
  USING (user_id = auth.uid())
  WITH CHECK (user_id = auth.uid());

-- --------------------------------------------------------------------------
-- bot_configs_safe  (view that masks secret columns for frontend queries)
-- --------------------------------------------------------------------------
CREATE OR REPLACE VIEW public.bot_configs_safe AS
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
  created_at, updated_at
FROM public.bot_configs;

-- --------------------------------------------------------------------------
-- leads
-- --------------------------------------------------------------------------
CREATE TABLE public.leads (
  id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  bot_id          uuid NOT NULL REFERENCES public.bot_configs(id) ON DELETE CASCADE,
  lead_name       varchar(200) NOT NULL,
  company         varchar(200) NOT NULL DEFAULT '',
  phone_number    varchar(30)  NOT NULL,
  lead_email      varchar(320) NOT NULL DEFAULT '',
  lead_context    text NOT NULL DEFAULT '',
  lifecycle_stage varchar(50)  NOT NULL DEFAULT 'follow_up',
  timezone        varchar(50)  NOT NULL DEFAULT 'UTC',
  owner_name      varchar(200) NOT NULL DEFAULT 'Sales Team',
  calendar_id     varchar(100) NOT NULL DEFAULT 'sales-team',
  metadata_json   jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at      timestamptz NOT NULL DEFAULT now(),

  UNIQUE (bot_id, phone_number)
);

CREATE INDEX idx_leads_bot_id ON public.leads(bot_id);

ALTER TABLE public.leads ENABLE ROW LEVEL SECURITY;

CREATE POLICY "users crud own leads"
  ON public.leads FOR ALL
  USING  (bot_id IN (SELECT id FROM public.bot_configs WHERE user_id = auth.uid()))
  WITH CHECK (bot_id IN (SELECT id FROM public.bot_configs WHERE user_id = auth.uid()));

-- --------------------------------------------------------------------------
-- call_logs
-- --------------------------------------------------------------------------
CREATE TABLE public.call_logs (
  id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  bot_id              uuid NOT NULL REFERENCES public.bot_configs(id) ON DELETE CASCADE,
  conversation_id     varchar(100) NOT NULL UNIQUE,
  lead_id             varchar(100) NOT NULL,
  started_at          timestamptz NOT NULL DEFAULT now(),
  completed_at        timestamptz,
  call_outcome        varchar(30) NOT NULL DEFAULT 'follow_up_needed',
  transcript          jsonb NOT NULL DEFAULT '[]'::jsonb,
  qualification_notes jsonb NOT NULL DEFAULT '{}'::jsonb,
  meeting_booked      boolean NOT NULL DEFAULT false,
  proposed_slot       varchar(100),
  follow_up_action    varchar(100)
);

CREATE INDEX idx_call_logs_bot_id ON public.call_logs(bot_id);
CREATE INDEX idx_call_logs_conversation_id ON public.call_logs(conversation_id);

ALTER TABLE public.call_logs ENABLE ROW LEVEL SECURITY;

CREATE POLICY "users read own call logs"
  ON public.call_logs FOR SELECT
  USING (bot_id IN (SELECT id FROM public.bot_configs WHERE user_id = auth.uid()));

-- Railway (service role) writes call logs; frontend is read-only.

-- --------------------------------------------------------------------------
-- sessions  (conversation state, managed by Railway AI engine)
-- --------------------------------------------------------------------------
CREATE TABLE public.sessions (
  conversation_id varchar(100) PRIMARY KEY,
  bot_id          uuid NOT NULL REFERENCES public.bot_configs(id) ON DELETE CASCADE,
  state_json      jsonb NOT NULL,
  created_at      timestamptz NOT NULL DEFAULT now(),
  updated_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX idx_sessions_bot_id ON public.sessions(bot_id);

CREATE TRIGGER sessions_updated_at
  BEFORE UPDATE ON public.sessions
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

ALTER TABLE public.sessions ENABLE ROW LEVEL SECURITY;

-- Sessions are managed exclusively by the Railway backend (service role).
-- Frontend can read its own sessions for display purposes.
CREATE POLICY "users read own sessions"
  ON public.sessions FOR SELECT
  USING (bot_id IN (SELECT id FROM public.bot_configs WHERE user_id = auth.uid()));

-- ==========================================================================
-- 00002_harden_handle_new_user.sql (verbatim)
-- ==========================================================================

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
DROP FUNCTION IF EXISTS public.handle_new_user();

CREATE FUNCTION public.handle_new_user()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
BEGIN
  INSERT INTO public.profiles (id, display_name)
  VALUES (NEW.id, COALESCE(NEW.raw_user_meta_data ->> 'display_name', ''))
  ON CONFLICT (id) DO UPDATE
  SET
    display_name = EXCLUDED.display_name,
    updated_at = now();

  RETURN NEW;
EXCEPTION
  WHEN OTHERS THEN
    RAISE WARNING 'handle_new_user failed for user %: %', NEW.id, SQLERRM;
    RETURN NEW;
END;
$$;

CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- ==========================================================================
-- 00003_conversation_spec.sql (verbatim)
-- ==========================================================================

ALTER TABLE public.bot_configs
  ADD COLUMN IF NOT EXISTS conversation_spec jsonb;

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

GRANT SELECT ON public.bot_configs_safe TO authenticated;
GRANT SELECT ON public.bot_configs_safe TO service_role;

-- ==========================================================================
-- 00004_agent_tools.sql (verbatim)
-- ==========================================================================

GRANT SELECT, INSERT, UPDATE, DELETE ON public.agent_tools TO authenticated;
GRANT ALL ON public.agent_tools TO service_role;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.agent_folders TO authenticated;
GRANT ALL ON public.agent_folders TO service_role;
