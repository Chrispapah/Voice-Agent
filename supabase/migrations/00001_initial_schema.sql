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
  initial_greeting text NOT NULL DEFAULT 'Hi, this is John — I know I''m calling out of the blue. Do you have 30 seconds so I can tell you why I''m reaching out?'::text,
  max_call_turns integer NOT NULL DEFAULT 12,
  max_objection_attempts integer NOT NULL DEFAULT 2,
  max_qualify_attempts integer NOT NULL DEFAULT 3,
  max_booking_attempts integer NOT NULL DEFAULT 3,
  sales_rep_name character varying NOT NULL DEFAULT 'Sales Team'::character varying,
  prompt_greeting text,
  prompt_qualify text,
  prompt_pitch text,
  prompt_objection text,
  prompt_booking text,
  prompt_wrapup text,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  updated_at timestamp with time zone NOT NULL DEFAULT now(),
  conversation_spec jsonb,
  folder_id uuid,
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
  CONSTRAINT call_logs_pkey PRIMARY KEY (id),
  CONSTRAINT call_logs_bot_id_fkey FOREIGN KEY (bot_id) REFERENCES public.bot_configs(id)
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