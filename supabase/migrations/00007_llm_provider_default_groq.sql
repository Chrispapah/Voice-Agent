-- New bot rows default to Groq (matches app ORM + SDRSettings).
alter table public.bot_configs
  alter column llm_provider set default 'groq';
