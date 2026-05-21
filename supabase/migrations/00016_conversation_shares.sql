CREATE TABLE IF NOT EXISTS public.conversation_shares (
  id uuid NOT NULL DEFAULT gen_random_uuid(),
  call_log_id uuid NOT NULL,
  token_hash text NOT NULL,
  created_by uuid NOT NULL,
  created_at timestamp with time zone NOT NULL DEFAULT now(),
  expires_at timestamp with time zone,
  revoked_at timestamp with time zone,
  CONSTRAINT conversation_shares_pkey PRIMARY KEY (id),
  CONSTRAINT conversation_shares_token_hash_key UNIQUE (token_hash),
  CONSTRAINT conversation_shares_call_log_id_fkey
    FOREIGN KEY (call_log_id) REFERENCES public.call_logs(id) ON DELETE CASCADE,
  CONSTRAINT conversation_shares_created_by_fkey
    FOREIGN KEY (created_by) REFERENCES auth.users(id)
);

CREATE INDEX IF NOT EXISTS conversation_shares_call_log_id_idx
  ON public.conversation_shares(call_log_id);

CREATE INDEX IF NOT EXISTS conversation_shares_token_hash_idx
  ON public.conversation_shares(token_hash);

CREATE INDEX IF NOT EXISTS conversation_shares_active_idx
  ON public.conversation_shares(token_hash, expires_at)
  WHERE revoked_at IS NULL;
