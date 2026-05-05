DROP FUNCTION IF EXISTS public.match_knowledge_chunks_for_user(vector, uuid, integer, uuid[]);

CREATE OR REPLACE FUNCTION public.match_knowledge_chunks_for_user(
  query_embedding vector(1536),
  match_user_id uuid,
  match_count int DEFAULT 5,
  knowledge_base_ids uuid[] DEFAULT NULL
)
RETURNS TABLE (
  id uuid,
  knowledge_base_id uuid,
  document_id uuid,
  chunk_index integer,
  content text,
  metadata_json jsonb,
  similarity float
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public, extensions
AS $$
  SELECT
    kc.id,
    kc.knowledge_base_id,
    kc.document_id,
    kc.chunk_index,
    kc.content,
    kc.metadata_json,
    1 - (kc.embedding <=> query_embedding) AS similarity
  FROM public.knowledge_chunks kc
  WHERE
    kc.user_id = match_user_id
    AND kc.embedding IS NOT NULL
    AND (
      knowledge_base_ids IS NULL
      OR kc.knowledge_base_id = ANY(knowledge_base_ids)
    )
  ORDER BY kc.embedding <=> query_embedding
  LIMIT match_count;
$$;

GRANT EXECUTE ON FUNCTION public.match_knowledge_chunks_for_user(vector, uuid, int, uuid[]) TO authenticated;
GRANT EXECUTE ON FUNCTION public.match_knowledge_chunks_for_user(vector, uuid, int, uuid[]) TO service_role;
