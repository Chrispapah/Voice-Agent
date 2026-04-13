import type { User as SupabaseUser } from "@supabase/supabase-js";
import { assertSupabaseConfigured, supabase } from "./supabase";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:3000";

function mapUser(user: SupabaseUser) {
  return {
    id: user.id,
    email: user.email || "",
    display_name: user.user_metadata?.display_name || "",
  };
}

function throwIfSupabaseError(error: { message: string } | null): asserts error is null {
  if (error) {
    throw new Error(error.message);
  }
}

async function requireAccessToken(): Promise<string> {
  assertSupabaseConfigured();
  const {
    data: { session },
  } = await supabase.auth.getSession();
  if (!session?.access_token) {
    throw new Error("Not authenticated");
  }
  return session.access_token;
}

async function railwayRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const accessToken = await requireAccessToken();
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${accessToken}`,
      ...(init?.headers || {}),
    },
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `API error ${res.status}`);
  }
  if (res.status === 204) return undefined as unknown as T;
  return res.json();
}

async function requireCurrentUserId(): Promise<string> {
  assertSupabaseConfigured();
  const {
    data: { user },
    error,
  } = await supabase.auth.getUser();
  throwIfSupabaseError(error);
  if (!user) {
    throw new Error("Not authenticated");
  }
  return user.id;
}

// Auth
export async function register(email: string, password: string, display_name: string): Promise<boolean> {
  assertSupabaseConfigured();
  const { data, error } = await supabase.auth.signUp({
    email,
    password,
    options: {
      data: { display_name },
    },
  });
  throwIfSupabaseError(error);
  return Boolean(data.session);
}

export async function login(email: string, password: string): Promise<void> {
  assertSupabaseConfigured();
  const { error } = await supabase.auth.signInWithPassword({ email, password });
  throwIfSupabaseError(error);
}

export async function getMe(): Promise<{ id: string; email: string; display_name: string }> {
  assertSupabaseConfigured();
  const {
    data: { user },
    error,
  } = await supabase.auth.getUser();
  throwIfSupabaseError(error);
  if (!user) {
    throw new Error("Not authenticated");
  }
  return mapUser(user);
}

// Bots
export interface BotConfig {
  id: string;
  name: string;
  is_active: boolean;
  llm_provider: string;
  llm_model_name: string;
  llm_temperature: number;
  llm_max_tokens: number;
  openai_api_key: string | null;
  anthropic_api_key: string | null;
  groq_api_key: string | null;
  elevenlabs_api_key: string | null;
  elevenlabs_voice_id: string | null;
  elevenlabs_model_id: string;
  deepgram_api_key: string | null;
  deepgram_model: string;
  deepgram_language: string;
  twilio_account_sid: string | null;
  twilio_auth_token: string | null;
  twilio_phone_number: string | null;
  initial_greeting: string;
  max_call_turns: number;
  max_objection_attempts: number;
  max_qualify_attempts: number;
  max_booking_attempts: number;
  sales_rep_name: string;
  prompt_greeting: string | null;
  prompt_qualify: string | null;
  prompt_pitch: string | null;
  prompt_objection: string | null;
  prompt_booking: string | null;
  prompt_wrapup: string | null;
  created_at: string | null;
  updated_at: string | null;
}

const SECRET_BOT_FIELDS = [
  "openai_api_key",
  "anthropic_api_key",
  "groq_api_key",
  "elevenlabs_api_key",
  "deepgram_api_key",
  "twilio_account_sid",
  "twilio_auth_token",
] as const;

function sanitizeBotUpdate(fields: Partial<BotConfig>): Partial<BotConfig> {
  const sanitized: Partial<BotConfig> = {};

  for (const [key, value] of Object.entries(fields) as [keyof BotConfig, BotConfig[keyof BotConfig]][]) {
    if (value === undefined) continue;

    if (
      SECRET_BOT_FIELDS.includes(key as (typeof SECRET_BOT_FIELDS)[number]) &&
      typeof value === "string" &&
      value.includes("****")
    ) {
      continue;
    }

    Object.assign(sanitized, { [key]: value });
  }

  return sanitized;
}

export async function listBots(): Promise<BotConfig[]> {
  assertSupabaseConfigured();
  const { data, error } = await supabase
    .from("bot_configs_safe")
    .select("*")
    .order("created_at", { ascending: false });
  throwIfSupabaseError(error);
  return data as BotConfig[];
}

export async function getBot(id: string): Promise<BotConfig> {
  assertSupabaseConfigured();
  const { data, error } = await supabase
    .from("bot_configs_safe")
    .select("*")
    .eq("id", id)
    .single();
  throwIfSupabaseError(error);
  return data as BotConfig;
}

export async function createBot(name: string): Promise<BotConfig> {
  assertSupabaseConfigured();
  const userId = await requireCurrentUserId();
  const { data, error } = await supabase
    .from("bot_configs")
    .insert({ name, user_id: userId })
    .select("id")
    .single();
  throwIfSupabaseError(error);
  return getBot(data.id);
}

export async function updateBot(id: string, fields: Partial<BotConfig>): Promise<BotConfig> {
  assertSupabaseConfigured();
  const sanitizedFields = sanitizeBotUpdate(fields);
  if (Object.keys(sanitizedFields).length === 0) {
    return getBot(id);
  }

  const { error } = await supabase.from("bot_configs").update(sanitizedFields).eq("id", id);
  throwIfSupabaseError(error);
  return getBot(id);
}

export async function deleteBot(id: string): Promise<void> {
  assertSupabaseConfigured();
  const { error } = await supabase.from("bot_configs").delete().eq("id", id);
  throwIfSupabaseError(error);
}

// Leads
export interface Lead {
  id: string;
  lead_name: string;
  company: string;
  phone_number: string;
  lead_email: string;
  lead_context: string;
  lifecycle_stage: string;
  owner_name: string;
  calendar_id: string;
  created_at: string | null;
}

export async function listLeads(botId: string): Promise<Lead[]> {
  assertSupabaseConfigured();
  const { data, error } = await supabase
    .from("leads")
    .select("*")
    .eq("bot_id", botId)
    .order("created_at", { ascending: false });
  throwIfSupabaseError(error);
  return data as Lead[];
}

export async function createLead(
  botId: string,
  data: Omit<Lead, "id" | "lifecycle_stage" | "created_at">,
): Promise<Lead> {
  assertSupabaseConfigured();
  const { data: created, error } = await supabase
    .from("leads")
    .insert({ ...data, bot_id: botId })
    .select("*")
    .single();
  throwIfSupabaseError(error);
  return created as Lead;
}

// Calls
export interface CallLog {
  id: string;
  conversation_id: string;
  lead_id: string;
  started_at: string | null;
  completed_at: string | null;
  call_outcome: string;
  transcript: { role: string; content: string }[];
  qualification_notes: Record<string, unknown>;
  meeting_booked: boolean;
  proposed_slot: string | null;
  follow_up_action: string | null;
}

export async function listCalls(botId: string): Promise<CallLog[]> {
  assertSupabaseConfigured();
  const { data, error } = await supabase
    .from("call_logs")
    .select("*")
    .eq("bot_id", botId)
    .order("started_at", { ascending: false });
  throwIfSupabaseError(error);
  return data as CallLog[];
}

export async function getCall(callId: string): Promise<CallLog> {
  assertSupabaseConfigured();
  const { data, error } = await supabase.from("call_logs").select("*").eq("id", callId).single();
  throwIfSupabaseError(error);
  return data as CallLog;
}

// Test sessions
export const startTestSession = (botId: string, leadId: string) =>
  railwayRequest<{ conversation_id: string; agent_response: string; stage: string }>(
    `/api/bots/${botId}/test-session`,
    { method: "POST", body: JSON.stringify({ lead_id: leadId }) },
  );

export const sendTestTurn = (botId: string, sessionId: string, humanInput: string) =>
  railwayRequest<{ conversation_id: string; agent_response: string; stage: string; call_outcome: string }>(
    `/api/bots/${botId}/test-session/${sessionId}/turns`,
    { method: "POST", body: JSON.stringify({ human_input: humanInput }) },
  );
