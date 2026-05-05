-- Global tool definitions for agents and graph nodes.
-- Agents reference these rows by id inside conversation_spec.tool_ids or nodes[].tool_ids.

CREATE TABLE IF NOT EXISTS public.agent_tools (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  name varchar(200) NOT NULL,
  description text NOT NULL DEFAULT '',
  kind varchar(40) NOT NULL DEFAULT 'http',
  config_json jsonb NOT NULL DEFAULT '{}'::jsonb,
  is_active boolean NOT NULL DEFAULT true,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_agent_tools_user_id ON public.agent_tools(user_id);

DROP TRIGGER IF EXISTS agent_tools_updated_at ON public.agent_tools;
CREATE TRIGGER agent_tools_updated_at
  BEFORE UPDATE ON public.agent_tools
  FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

ALTER TABLE public.agent_tools ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "users crud own tools" ON public.agent_tools;
CREATE POLICY "users crud own tools"
  ON public.agent_tools FOR ALL
  USING (user_id = auth.uid())
  WITH CHECK (user_id = auth.uid());

GRANT SELECT, INSERT, UPDATE, DELETE ON public.agent_tools TO authenticated;
GRANT ALL ON public.agent_tools TO service_role;
