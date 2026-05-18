-- Default new bots to Greek for Deepgram/browser STT language column.
ALTER TABLE public.bot_configs
  ALTER COLUMN deepgram_language SET DEFAULT 'el'::character varying;
