-- Default chat model matches Groq-capable id (conversation brain uses Groq for all LLM calls).
alter table public.bot_configs
  alter column llm_model_name set default 'llama-3.3-70b-versatile';
