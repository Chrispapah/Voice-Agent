UPDATE public.call_logs AS call_log
SET completed_at = session_row.updated_at
FROM public.sessions AS session_row
WHERE call_log.conversation_id = session_row.conversation_id
  AND call_log.completed_at IS NULL
  AND session_row.updated_at IS NOT NULL;

UPDATE public.call_logs
SET completed_at = started_at
WHERE completed_at IS NULL
  AND jsonb_array_length(transcript) > 0;
