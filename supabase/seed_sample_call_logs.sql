-- Sample call_logs for UI testing.
--
-- Run in Supabase: SQL Editor → New query → paste → Run.
-- Uses the database role that bypasses RLS (default in Dashboard SQL Editor).
--
-- Rows attach to your NEWEST bot (by created_at). Create a conversation in the
-- app first if bot_configs is empty. To target one bot, replace the subquery
-- with a literal UUID, e.g. WHERE id = 'xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx'::uuid

BEGIN;

INSERT INTO public.call_logs (
  bot_id,
  conversation_id,
  lead_id,
  started_at,
  completed_at,
  call_outcome,
  transcript,
  qualification_notes,
  meeting_booked,
  proposed_slot,
  follow_up_action
)
SELECT
  b.id,
  'seed-' || gen_random_uuid()::text,
  'lead-demo-001',
  now() - interval '2 days',
  now() - interval '2 days' + interval '4 minutes',
  'meeting_booked',
  '[
    {"role": "assistant", "content": "Hi, this is Alex calling from Acme. Is this a good time for a quick chat?"},
    {"role": "human", "content": "I have about five minutes."},
    {"role": "assistant", "content": "Perfect. We help sales teams cut no-shows in half. Would Tuesday at 2pm work for a 15-minute walkthrough?"},
    {"role": "human", "content": "Tuesday 2 works. Send the invite."},
    {"role": "assistant", "content": "Great—you will see a calendar invite shortly. Thanks for your time."}
  ]'::jsonb,
  '{"budget": "confirmed", "timeline": "Q2"}'::jsonb,
  true,
  'Tuesday 2:00 PM',
  NULL
FROM (SELECT id FROM public.bot_configs ORDER BY created_at DESC LIMIT 1) AS b;

INSERT INTO public.call_logs (
  bot_id,
  conversation_id,
  lead_id,
  started_at,
  completed_at,
  call_outcome,
  transcript,
  qualification_notes,
  meeting_booked,
  proposed_slot,
  follow_up_action
)
SELECT
  b.id,
  'seed-' || gen_random_uuid()::text,
  'lead-demo-002',
  now() - interval '1 day',
  now() - interval '1 day' + interval '6 minutes',
  'follow_up_needed',
  '[
    {"role": "assistant", "content": "Hey, it is Jordan from Acme. Am I catching you at an okay time?"},
    {"role": "human", "content": "We might be interested but I need to loop in our VP."},
    {"role": "assistant", "content": "Totally fair. Would it help if I sent a one-pager she can skim?"},
    {"role": "human", "content": "Yes, email me and I will forward it internally."},
    {"role": "assistant", "content": "Will do. I will follow up next week unless I hear from you sooner."}
  ]'::jsonb,
  '{"vp_approval": "pending"}'::jsonb,
  false,
  NULL,
  'Send deck; follow up in 1 week'
FROM (SELECT id FROM public.bot_configs ORDER BY created_at DESC LIMIT 1) AS b;

INSERT INTO public.call_logs (
  bot_id,
  conversation_id,
  lead_id,
  started_at,
  completed_at,
  call_outcome,
  transcript,
  qualification_notes,
  meeting_booked,
  proposed_slot,
  follow_up_action
)
SELECT
  b.id,
  'seed-' || gen_random_uuid()::text,
  'lead-demo-003',
  now() - interval '3 hours',
  now() - interval '3 hours' + interval '2 minutes',
  'not_interested',
  '[
    {"role": "assistant", "content": "Hi, this is Alex from Acme—did I catch you at a bad time?"},
    {"role": "human", "content": "We are not in market for anything new this year."},
    {"role": "assistant", "content": "Understood. Thanks for being upfront, and have a great day."}
  ]'::jsonb,
  '{}'::jsonb,
  false,
  NULL,
  NULL
FROM (SELECT id FROM public.bot_configs ORDER BY created_at DESC LIMIT 1) AS b;

COMMIT;

-- If INSERT 0 rows: you have no bot_configs. Create "My conversation" in the app, then re-run.
-- To attach to a specific bot only, use e.g.:
-- FROM (SELECT id FROM public.bot_configs WHERE id = 'YOUR-BOT-UUID'::uuid) AS b;
