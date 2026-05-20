ALTER TABLE public.bot_configs
  ALTER COLUMN openai_realtime_model SET DEFAULT 'gpt-realtime';

UPDATE public.bot_configs
SET openai_realtime_model = 'gpt-realtime'
WHERE openai_realtime_model = 'gpt-4o-realtime-preview';
