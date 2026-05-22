-- Per-bot knowledge base retrieval settings.
--
-- These columns drive the voice agent's tool-calling RAG path (see
-- src/ai_sdr_agent/services/knowledge.py and graph/dynamic_nodes.py). They are
-- read at turn time from state["bot_config"], so changing a row takes effect
-- on the next call without redeploying the runtime.

ALTER TABLE public.bot_configs
  ADD COLUMN IF NOT EXISTS kb_match_count integer NOT NULL DEFAULT 5,
  ADD COLUMN IF NOT EXISTS kb_min_similarity double precision NOT NULL DEFAULT 0.2,
  ADD COLUMN IF NOT EXISTS kb_embedding_model character varying NOT NULL DEFAULT 'text-embedding-3-small',
  ADD COLUMN IF NOT EXISTS kb_max_context_chars integer NOT NULL DEFAULT 6000,
  ADD COLUMN IF NOT EXISTS kb_max_tool_iterations integer NOT NULL DEFAULT 2;

ALTER TABLE public.bot_configs
  DROP CONSTRAINT IF EXISTS bot_configs_kb_match_count_check,
  DROP CONSTRAINT IF EXISTS bot_configs_kb_min_similarity_check,
  DROP CONSTRAINT IF EXISTS bot_configs_kb_max_context_chars_check,
  DROP CONSTRAINT IF EXISTS bot_configs_kb_max_tool_iterations_check;

ALTER TABLE public.bot_configs
  ADD CONSTRAINT bot_configs_kb_match_count_check
    CHECK (kb_match_count BETWEEN 1 AND 20),
  ADD CONSTRAINT bot_configs_kb_min_similarity_check
    CHECK (kb_min_similarity >= 0 AND kb_min_similarity <= 1),
  ADD CONSTRAINT bot_configs_kb_max_context_chars_check
    CHECK (kb_max_context_chars BETWEEN 500 AND 20000),
  ADD CONSTRAINT bot_configs_kb_max_tool_iterations_check
    CHECK (kb_max_tool_iterations BETWEEN 1 AND 4);

COMMENT ON COLUMN public.bot_configs.kb_match_count IS
  'Top-N chunks the voice agent fetches per knowledge base lookup.';
COMMENT ON COLUMN public.bot_configs.kb_min_similarity IS
  'Cosine-similarity floor (0..1) for chunks returned to the LLM.';
COMMENT ON COLUMN public.bot_configs.kb_embedding_model IS
  'OpenAI embedding model. Must produce 1536-dim vectors and match how documents were ingested.';
COMMENT ON COLUMN public.bot_configs.kb_max_context_chars IS
  'Maximum total characters of KB content injected into a single tool result.';
COMMENT ON COLUMN public.bot_configs.kb_max_tool_iterations IS
  'Maximum number of lookup_knowledge tool calls the model may issue per turn.';
