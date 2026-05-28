-- Workspace env vars for {{NAME}} substitution in tool URLs/headers.
-- Auth connections for tool HTTP authentication (Phase 2).

CREATE TABLE IF NOT EXISTS public.workspace_env_vars (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  name varchar(120) NOT NULL,
  value text NOT NULL DEFAULT '',
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now(),
  CONSTRAINT workspace_env_vars_user_name_unique UNIQUE (user_id, name)
);

CREATE INDEX IF NOT EXISTS idx_workspace_env_vars_user_id ON public.workspace_env_vars(user_id);

DROP TRIGGER IF EXISTS workspace_env_vars_updated_at ON public.workspace_env_vars;
CREATE TRIGGER workspace_env_vars_updated_at
  BEFORE UPDATE ON public.workspace_env_vars
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

ALTER TABLE public.workspace_env_vars ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "users crud own env vars" ON public.workspace_env_vars;
CREATE POLICY "users crud own env vars"
  ON public.workspace_env_vars FOR ALL
  USING (user_id = auth.uid())
  WITH CHECK (user_id = auth.uid());

GRANT SELECT, INSERT, UPDATE, DELETE ON public.workspace_env_vars TO authenticated;
GRANT ALL ON public.workspace_env_vars TO service_role;

CREATE TABLE IF NOT EXISTS public.auth_connections (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  label varchar(200) NOT NULL,
  type varchar(40) NOT NULL DEFAULT 'api_key',
  config_json jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_auth_connections_user_id ON public.auth_connections(user_id);

DROP TRIGGER IF EXISTS auth_connections_updated_at ON public.auth_connections;
CREATE TRIGGER auth_connections_updated_at
  BEFORE UPDATE ON public.auth_connections
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

ALTER TABLE public.auth_connections ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "users crud own auth connections" ON public.auth_connections;
CREATE POLICY "users crud own auth connections"
  ON public.auth_connections FOR ALL
  USING (user_id = auth.uid())
  WITH CHECK (user_id = auth.uid());

GRANT SELECT, INSERT, UPDATE, DELETE ON public.auth_connections TO authenticated;
GRANT ALL ON public.auth_connections TO service_role;
