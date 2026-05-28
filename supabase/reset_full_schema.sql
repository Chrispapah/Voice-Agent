-- WARNING: This schema is for context only and is not meant to be run.
-- Table order and constraints may not be valid for execution.

CREATE TABLE public.agent_folders (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  name character varying NOT NULL,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT agent_folders_pkey PRIMARY KEY (id),
  CONSTRAINT agent_folders_user_id_fkey FOREIGN KEY (user_id) REFERENCES auth.users(id)
);
CREATE TABLE public.agent_node_knowledge_bases (
  bot_id uuid NOT NULL,
  node_id character varying NOT NULL,
  knowledge_base_id uuid NOT NULL,
  user_id uuid NOT NULL,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT agent_node_knowledge_bases_pkey PRIMARY KEY (bot_id, node_id, knowledge_base_id),
  CONSTRAINT agent_node_knowledge_bases_bot_id_fkey FOREIGN KEY (bot_id) REFERENCES public.bot_configs(id),
  CONSTRAINT agent_node_knowledge_bases_knowledge_base_id_fkey FOREIGN KEY (knowledge_base_id) REFERENCES public.knowledge_bases(id),
  CONSTRAINT agent_node_knowledge_bases_user_id_fkey FOREIGN KEY (user_id) REFERENCES auth.users(id)
);
CREATE TABLE public.agent_preview_shares (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  bot_id uuid NOT NULL,
  token_hash text NOT NULL UNIQUE,
  created_by uuid NOT NULL,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  expires_at timestamp with time zone,
  revoked_at timestamp with time zone,
  max_sessions integer NOT NULL DEFAULT 100,
  session_count integer NOT NULL DEFAULT 0,
  title text,
  welcome_message text,
  CONSTRAINT agent_preview_shares_pkey PRIMARY KEY (id),
  CONSTRAINT agent_preview_shares_bot_id_fkey FOREIGN KEY (bot_id) REFERENCES public.bot_configs(id),
  CONSTRAINT agent_preview_shares_created_by_fkey FOREIGN KEY (created_by) REFERENCES auth.users(id)
);
CREATE TABLE public.agent_tools (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  name character varying NOT NULL,
  description text NOT NULL DEFAULT ''::text,
  kind character varying NOT NULL DEFAULT 'http'::character varying,
  config_json jsonb NOT NULL DEFAULT '{}'::jsonb,
  is_active boolean NOT NULL DEFAULT true,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT agent_tools_pkey PRIMARY KEY (id),
  CONSTRAINT agent_tools_user_id_fkey FOREIGN KEY (user_id) REFERENCES auth.users(id)
);
CREATE TABLE public.bot_configs (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  name character varying NOT NULL DEFAULT 'My Bot'::character varying,
  is_active boolean NOT NULL DEFAULT true,
  llm_provider character varying NOT NULL DEFAULT 'openai'::character varying,
  llm_model_name character varying NOT NULL DEFAULT 'gpt-4o-mini'::character varying,
  llm_temperature double precision NOT NULL DEFAULT 0.4,
  llm_max_tokens integer NOT NULL DEFAULT 300,
  openai_api_key text,
  anthropic_api_key text,
  groq_api_key text,
  elevenlabs_api_key text,
  elevenlabs_voice_id character varying,
  elevenlabs_model_id character varying NOT NULL DEFAULT 'eleven_turbo_v2'::character varying,
  deepgram_api_key text,
  deepgram_model character varying NOT NULL DEFAULT 'nova-2'::character varying,
  deepgram_language character varying NOT NULL DEFAULT 'en-US'::character varying,
  twilio_account_sid text,
  twilio_auth_token text,
  twilio_phone_number character varying,
  max_call_turns integer NOT NULL DEFAULT 12,
  sales_rep_name character varying NOT NULL DEFAULT 'Sales Team'::character varying,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  conversation_spec jsonb,
  folder_id uuid,
  voice_provider character varying NOT NULL DEFAULT 'builtin'::character varying,
  openai_realtime_model character varying NOT NULL DEFAULT 'gpt-realtime'::character varying,
  openai_realtime_voice character varying NOT NULL DEFAULT 'alloy'::character varying,
  openai_realtime_instructions text,
  kb_match_count integer NOT NULL DEFAULT 5 CHECK (kb_match_count >= 1 AND kb_match_count <= 20),
  kb_min_similarity double precision NOT NULL DEFAULT 0.2 CHECK (kb_min_similarity >= 0::double precision AND kb_min_similarity <= 1::double precision),
  kb_embedding_model character varying NOT NULL DEFAULT 'text-embedding-3-small'::character varying,
  kb_max_context_chars integer NOT NULL DEFAULT 6000 CHECK (kb_max_context_chars >= 500 AND kb_max_context_chars <= 20000),
  kb_max_tool_iterations integer NOT NULL DEFAULT 2 CHECK (kb_max_tool_iterations >= 1 AND kb_max_tool_iterations <= 4),
  allow_voice_interruptions boolean NOT NULL DEFAULT true,
  CONSTRAINT bot_configs_pkey PRIMARY KEY (id),
  CONSTRAINT bot_configs_user_id_fkey FOREIGN KEY (user_id) REFERENCES auth.users(id),
  CONSTRAINT bot_configs_folder_id_fkey FOREIGN KEY (folder_id) REFERENCES public.agent_folders(id)
);
CREATE TABLE public.bot_knowledge_bases (
  bot_id uuid NOT NULL,
  knowledge_base_id uuid NOT NULL,
  user_id uuid NOT NULL,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT bot_knowledge_bases_pkey PRIMARY KEY (bot_id, knowledge_base_id),
  CONSTRAINT bot_knowledge_bases_bot_id_fkey FOREIGN KEY (bot_id) REFERENCES public.bot_configs(id),
  CONSTRAINT bot_knowledge_bases_knowledge_base_id_fkey FOREIGN KEY (knowledge_base_id) REFERENCES public.knowledge_bases(id),
  CONSTRAINT bot_knowledge_bases_user_id_fkey FOREIGN KEY (user_id) REFERENCES auth.users(id)
);
CREATE TABLE public.call_logs (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  bot_id uuid NOT NULL,
  conversation_id character varying NOT NULL UNIQUE,
  lead_id character varying NOT NULL,
  started_at timestamp with time zone NOT NULL DEFAULT now(),
  completed_at timestamp with time zone,
  call_outcome character varying NOT NULL DEFAULT 'follow_up_needed'::character varying,
  transcript jsonb NOT NULL DEFAULT '[]'::jsonb,
  qualification_notes jsonb NOT NULL DEFAULT '{}'::jsonb,
  meeting_booked boolean NOT NULL DEFAULT false,
  proposed_slot character varying,
  follow_up_action character varying,
  call_quality character varying NOT NULL DEFAULT 'needs_attention'::character varying CHECK (call_quality::text = ANY (ARRAY['satisfactory'::character varying, 'unsatisfactory'::character varying, 'needs_attention'::character varying]::text[])),
  CONSTRAINT call_logs_pkey PRIMARY KEY (id),
  CONSTRAINT call_logs_bot_id_fkey FOREIGN KEY (bot_id) REFERENCES public.bot_configs(id)
);
CREATE TABLE public.conversation_shares (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  call_log_id uuid NOT NULL,
  token_hash text NOT NULL UNIQUE,
  created_by uuid NOT NULL,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  expires_at timestamp with time zone,
  revoked_at timestamp with time zone,
  CONSTRAINT conversation_shares_pkey PRIMARY KEY (id),
  CONSTRAINT conversation_shares_call_log_id_fkey FOREIGN KEY (call_log_id) REFERENCES public.call_logs(id),
  CONSTRAINT conversation_shares_created_by_fkey FOREIGN KEY (created_by) REFERENCES auth.users(id)
);
CREATE TABLE public.knowledge_bases (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL,
  name character varying NOT NULL,
  description text NOT NULL DEFAULT ''::text,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT knowledge_bases_pkey PRIMARY KEY (id),
  CONSTRAINT knowledge_bases_user_id_fkey FOREIGN KEY (user_id) REFERENCES auth.users(id)
);
CREATE TABLE public.knowledge_chunks (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  knowledge_base_id uuid NOT NULL,
  document_id uuid NOT NULL,
  user_id uuid NOT NULL,
  chunk_index integer NOT NULL DEFAULT 0,
  content text NOT NULL,
  metadata_json jsonb NOT NULL DEFAULT '{}'::jsonb,
  embedding USER-DEFINED,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT knowledge_chunks_pkey PRIMARY KEY (id),
  CONSTRAINT knowledge_chunks_knowledge_base_id_fkey FOREIGN KEY (knowledge_base_id) REFERENCES public.knowledge_bases(id),
  CONSTRAINT knowledge_chunks_document_id_fkey FOREIGN KEY (document_id) REFERENCES public.knowledge_documents(id),
  CONSTRAINT knowledge_chunks_user_id_fkey FOREIGN KEY (user_id) REFERENCES auth.users(id)
);
CREATE TABLE public.knowledge_documents (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  knowledge_base_id uuid NOT NULL,
  user_id uuid NOT NULL,
  title character varying NOT NULL,
  source_url text,
  storage_path text,
  mime_type character varying,
  status character varying NOT NULL DEFAULT 'ready'::character varying,
  metadata_json jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT knowledge_documents_pkey PRIMARY KEY (id),
  CONSTRAINT knowledge_documents_knowledge_base_id_fkey FOREIGN KEY (knowledge_base_id) REFERENCES public.knowledge_bases(id),
  CONSTRAINT knowledge_documents_user_id_fkey FOREIGN KEY (user_id) REFERENCES auth.users(id)
);
CREATE TABLE public.leads (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  bot_id uuid NOT NULL,
  lead_name character varying NOT NULL,
  company character varying NOT NULL DEFAULT ''::character varying,
  phone_number character varying NOT NULL,
  lead_email character varying NOT NULL DEFAULT ''::character varying,
  lead_context text NOT NULL DEFAULT ''::text,
  lifecycle_stage character varying NOT NULL DEFAULT 'follow_up'::character varying,
  timezone character varying NOT NULL DEFAULT 'UTC'::character varying,
  owner_name character varying NOT NULL DEFAULT 'Sales Team'::character varying,
  calendar_id character varying NOT NULL DEFAULT 'sales-team'::character varying,
  metadata_json jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT leads_pkey PRIMARY KEY (id),
  CONSTRAINT leads_bot_id_fkey FOREIGN KEY (bot_id) REFERENCES public.bot_configs(id)
);
CREATE TABLE public.profiles (
  id uuid NOT NULL,
  display_name text NOT NULL DEFAULT ''::text,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT profiles_pkey PRIMARY KEY (id),
  CONSTRAINT profiles_id_fkey FOREIGN KEY (id) REFERENCES auth.users(id)
);
CREATE TABLE public.sessions (
  conversation_id character varying NOT NULL,
  bot_id uuid NOT NULL,
  state_json jsonb NOT NULL,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  CONSTRAINT sessions_pkey PRIMARY KEY (conversation_id),
  CONSTRAINT sessions_bot_id_fkey FOREIGN KEY (bot_id) REFERENCES public.bot_configs(id)
);

-- Masked API keys for client reads (must match bot_configs columns after SDR legacy removal).
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