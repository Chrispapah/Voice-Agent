CREATE TABLE public.agent_preview_shares (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  bot_id uuid NOT NULL,
  token_hash text NOT NULL,
  created_by uuid NOT NULL,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  expires_at timestamp with time zone,
  revoked_at timestamp with time zone,
  max_sessions integer NOT NULL DEFAULT 100,
  session_count integer NOT NULL DEFAULT 0,
  title text,
  welcome_message text,
  CONSTRAINT agent_preview_shares_pkey PRIMARY KEY (id),
  CONSTRAINT agent_preview_shares_token_hash_key UNIQUE (token_hash),
  CONSTRAINT agent_preview_shares_bot_id_fkey FOREIGN KEY (bot_id) REFERENCES public.bot_configs(id) ON DELETE CASCADE,
  CONSTRAINT agent_preview_shares_created_by_fkey FOREIGN KEY (created_by) REFERENCES auth.users(id)
);

CREATE INDEX agent_preview_shares_bot_id_idx ON public.agent_preview_shares(bot_id);
CREATE INDEX agent_preview_shares_token_hash_idx ON public.agent_preview_shares(token_hash);
CREATE INDEX agent_preview_shares_active_idx
  ON public.agent_preview_shares(token_hash, expires_at)
  WHERE revoked_at IS NULL;
