ALTER TABLE public.call_logs
  ADD COLUMN IF NOT EXISTS call_quality character varying NOT NULL DEFAULT 'needs_attention';

ALTER TABLE public.call_logs
  DROP CONSTRAINT IF EXISTS call_logs_call_quality_check;

ALTER TABLE public.call_logs
  ADD CONSTRAINT call_logs_call_quality_check
  CHECK (call_quality IN ('satisfactory', 'unsatisfactory', 'needs_attention'));
