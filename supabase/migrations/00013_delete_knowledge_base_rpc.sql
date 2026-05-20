-- Cascade-delete a knowledge base and related rows in one trusted call.
-- Avoids relying on broad DELETE grants on knowledge_chunks from the browser.

CREATE OR REPLACE FUNCTION public.delete_knowledge_base_for_user(p_kb_id uuid)
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  v_user uuid := auth.uid();
BEGIN
  IF v_user IS NULL THEN
    RAISE EXCEPTION 'not authenticated';
  END IF;

  IF NOT EXISTS (SELECT 1 FROM public.knowledge_bases WHERE id = p_kb_id AND user_id = v_user) THEN
    RAISE EXCEPTION 'knowledge base not found';
  END IF;

  DELETE FROM public.knowledge_chunks
  WHERE knowledge_base_id = p_kb_id AND user_id = v_user;

  DELETE FROM public.knowledge_documents
  WHERE knowledge_base_id = p_kb_id AND user_id = v_user;

  DELETE FROM public.bot_knowledge_bases
  WHERE knowledge_base_id = p_kb_id AND user_id = v_user;

  DELETE FROM public.agent_node_knowledge_bases
  WHERE knowledge_base_id = p_kb_id AND user_id = v_user;

  DELETE FROM public.knowledge_bases
  WHERE id = p_kb_id AND user_id = v_user;
END;
$$;

REVOKE ALL ON FUNCTION public.delete_knowledge_base_for_user(uuid) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.delete_knowledge_base_for_user(uuid) TO authenticated;
