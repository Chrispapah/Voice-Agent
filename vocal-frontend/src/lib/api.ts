import type { ConversationSpecV1 } from "./conversationSpec";
import {
  assertSupabaseConfigured,
  getSupabaseAccessToken,
  getSupabaseUrl,
  supabase,
} from "./supabase";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:3000";
const STATIC_BEARER_TOKEN = import.meta.env.VITE_API_BEARER_TOKEN;
const LOCAL_BEARER_TOKEN_KEY = "vocal-api-bearer-token";

export class AuthRequiredError extends Error {
  constructor() {
    super("Not authenticated");
    this.name = "AuthRequiredError";
  }
}

export interface BotConfig {
  id: string;
  user_id: string;
  name: string;
  folder_id: string | null;
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
  conversation_spec?: ConversationSpecV1 | null;
  created_at: string | null;
  updated_at: string | null;
}

export type AgentListItem = Pick<
  BotConfig,
  | "id"
  | "user_id"
  | "folder_id"
  | "name"
  | "elevenlabs_voice_id"
  | "twilio_phone_number"
  | "conversation_spec"
  | "created_at"
  | "updated_at"
>;

export interface AgentFolder {
  id: string;
  user_id: string;
  name: string;
  created_at: string | null;
  updated_at: string | null;
}

export interface Lead {
  id: string;
  bot_id: string;
  lead_name: string;
  company: string;
  phone_number: string;
  lead_email: string;
  lead_context: string;
  lifecycle_stage: string;
  timezone: string;
  owner_name: string;
  calendar_id: string;
  created_at: string | null;
}

export type ToolDefinitionKind = "http" | "webhook" | "custom";

export interface ToolDefinition {
  id: string;
  user_id: string;
  name: string;
  description: string;
  kind: ToolDefinitionKind;
  config_json: Record<string, unknown>;
  is_active: boolean;
  created_at: string | null;
  updated_at: string | null;
}

export type ToolDefinitionInput = Pick<ToolDefinition, "name" | "description" | "kind" | "config_json">;

export interface KnowledgeBase {
  id: string;
  user_id: string;
  name: string;
  description: string;
  created_at: string | null;
  updated_at: string | null;
}

export interface KnowledgeDocument {
  id: string;
  knowledge_base_id: string;
  user_id: string;
  title: string;
  source_url: string | null;
  storage_path: string | null;
  mime_type: string | null;
  status: string;
  metadata_json: Record<string, unknown>;
  created_at: string | null;
  updated_at: string | null;
}

export interface IngestKnowledgeDocumentInput {
  knowledge_base_id: string;
  title: string;
  content?: string;
  file?: File;
  source_url?: string | null;
  metadata_json?: Record<string, unknown>;
}

export interface IngestKnowledgeDocumentResponse {
  document_id: string;
  knowledge_base_id: string;
  chunks_created: number;
  status: string;
}

export interface AnswerKnowledgeQuestionInput {
  question: string;
  knowledge_base_ids?: string[];
  bot_id?: string;
  node_id?: string;
  match_count?: number;
  whole_kb_max_chunks?: number;
  whole_kb_max_context_chars?: number;
  hybrid_max_matches?: number;
  llm_provider?: string;
  answer_model?: string;
  embedding_model?: string;
  openai_api_key?: string;
}

export interface AnswerKnowledgeQuestionResponse {
  answer: string;
  matches: Array<{
    id: string;
    knowledge_base_id: string;
    document_id: string;
    chunk_index?: number;
    similarity: number;
    retrieval_score?: number;
    retrieval_reasons?: string[];
    keyword_hits?: number;
    content_preview: string;
    metadata_json: Record<string, unknown>;
  }>;
  knowledge_base_ids: string[];
  retrieval_mode?: "whole_kb" | "hybrid";
  retrieval_config?: {
    llm_provider?: string;
    answer_model?: string;
    embedding_model?: string;
    uses_request_openai_api_key?: boolean;
    match_count: number;
    whole_kb_max_chunks: number;
    whole_kb_max_context_chars: number;
    hybrid_max_matches: number;
  };
}

export interface TestSessionResponse {
  conversation_id: string;
  agent_response: string;
  stage: string;
  active_node: string;
  next_node: string;
  call_outcome?: string;
}

export interface CallLog {
  id: string;
  bot_id: string;
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

export interface AgentsPageData {
  agents: AgentListItem[];
  folders: AgentFolder[];
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

function throwIfSupabaseError(error: { message: string } | null): asserts error is null {
  if (error) {
    throw new Error(error.message);
  }
}

function tokenFromSupabaseStorage(): string | null {
  if (typeof window === "undefined") return null;
  for (let i = 0; i < window.localStorage.length; i += 1) {
    const key = window.localStorage.key(i);
    if (!key || !key.startsWith("sb-") || !key.endsWith("-auth-token")) continue;
    const raw = window.localStorage.getItem(key);
    if (!raw) continue;
    try {
      const parsed = JSON.parse(raw);
      const token = parsed?.access_token || parsed?.currentSession?.access_token;
      if (typeof token === "string" && token.length > 0) return token;
    } catch {
      // Ignore unrelated localStorage values.
    }
  }
  return null;
}

function localBearerToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(LOCAL_BEARER_TOKEN_KEY);
}

export function saveLocalBearerToken(token: string): void {
  window.localStorage.setItem(LOCAL_BEARER_TOKEN_KEY, token);
}

export function clearLocalBearerToken(): void {
  window.localStorage.removeItem(LOCAL_BEARER_TOKEN_KEY);
}

async function requireAccessToken(): Promise<string> {
  const token = STATIC_BEARER_TOKEN || localBearerToken() || await getSupabaseAccessToken() || tokenFromSupabaseStorage();
  if (!token) {
    throw new AuthRequiredError();
  }
  return token;
}

/** Resolves the same bearer token used for REST API calls (for WebSocket auth messages). */
export async function resolveAccessToken(): Promise<string> {
  return requireAccessToken();
}

/** `wss://` / `ws://` URL for the browser voice session WebSocket. */
export function getVoiceSessionWebSocketUrl(botId: string): string {
  const trimmed = API_BASE.trim().replace(/\/$/, "");
  const wsBase = trimmed.startsWith("https://")
    ? `wss://${trimmed.slice("https://".length)}`
    : trimmed.startsWith("http://")
      ? `ws://${trimmed.slice("http://".length)}`
      : trimmed;
  return `${wsBase}/api/bots/${botId}/voice-session`;
}

async function requireCurrentUserId(): Promise<string> {
  assertSupabaseConfigured();
  const {
    data: { user },
    error,
  } = await supabase.auth.getUser();
  throwIfSupabaseError(error);
  if (!user) {
    throw new AuthRequiredError();
  }
  return user.id;
}

async function apiRequest<T>(path: string, init?: RequestInit): Promise<T> {
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
  if (res.status === 204) return undefined as T;
  return res.json();
}

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

export function listBots(): Promise<BotConfig[]> {
  return listBotsFromSupabase();
}

export function listAgentsPageData(): Promise<AgentsPageData> {
  return listAgentsPageDataFromSupabase();
}

export function getBot(id: string): Promise<BotConfig> {
  return getBotFromSupabase(id);
}

export function createBot(name: string, folderId?: string | null): Promise<BotConfig> {
  return createBotInSupabase(name, folderId);
}

export function updateBot(id: string, fields: Partial<BotConfig>): Promise<BotConfig> {
  return updateBotInSupabase(id, fields);
}

export function deleteBot(id: string): Promise<void> {
  return deleteBotFromSupabase(id);
}

export function listAgentFolders(): Promise<AgentFolder[]> {
  return listAgentFoldersFromSupabase();
}

export function createAgentFolder(name: string): Promise<AgentFolder> {
  return createAgentFolderInSupabase(name);
}

export function listLeads(botId: string): Promise<Lead[]> {
  return listLeadsFromSupabase(botId);
}

export function listTools(): Promise<ToolDefinition[]> {
  return listToolsFromSupabase();
}

export function createTool(data: ToolDefinitionInput): Promise<ToolDefinition> {
  return createToolInSupabase(data);
}

export function updateTool(id: string, data: ToolDefinitionInput): Promise<ToolDefinition> {
  return updateToolInSupabase(id, data);
}

export function listKnowledgeBases(): Promise<KnowledgeBase[]> {
  return listKnowledgeBasesFromSupabase();
}

export function createKnowledgeBase(data: Pick<KnowledgeBase, "name" | "description">): Promise<KnowledgeBase> {
  return createKnowledgeBaseInSupabase(data);
}

export function listKnowledgeDocuments(knowledgeBaseId: string): Promise<KnowledgeDocument[]> {
  return listKnowledgeDocumentsFromSupabase(knowledgeBaseId);
}

export function ingestKnowledgeDocument(
  data: IngestKnowledgeDocumentInput,
): Promise<IngestKnowledgeDocumentResponse> {
  return ingestKnowledgeDocumentWithEdgeFunction(data);
}

export function answerKnowledgeQuestion(
  data: AnswerKnowledgeQuestionInput,
): Promise<AnswerKnowledgeQuestionResponse> {
  return answerKnowledgeQuestionWithEdgeFunction(data);
}

export function listBotKnowledgeBaseIds(botId: string): Promise<string[]> {
  return listBotKnowledgeBaseIdsFromSupabase(botId);
}

export function setBotKnowledgeBases(botId: string, knowledgeBaseIds: string[]): Promise<void> {
  return setBotKnowledgeBasesInSupabase(botId, knowledgeBaseIds);
}

export function listNodeKnowledgeBaseAssignments(botId: string): Promise<Record<string, string[]>> {
  return listNodeKnowledgeBaseAssignmentsFromSupabase(botId);
}

export function setNodeKnowledgeBaseAssignments(
  botId: string,
  assignments: Record<string, string[]>,
): Promise<void> {
  return setNodeKnowledgeBaseAssignmentsInSupabase(botId, assignments);
}

export function createLead(botId: string, data: Omit<Lead, "id" | "bot_id" | "lifecycle_stage" | "created_at">) {
  return createLeadInSupabase(botId, data);
}

export function listCalls(): Promise<CallLog[]> {
  return listCallsFromSupabase();
}

async function listBotsFromSupabase(): Promise<BotConfig[]> {
  assertSupabaseConfigured();
  const userId = await requireCurrentUserId();
  const { data, error } = await supabase
    .from("bot_configs_safe")
    .select("*")
    .eq("user_id", userId)
    .order("created_at", { ascending: false });
  throwIfSupabaseError(error);
  return data as BotConfig[];
}

async function listAgentsPageDataFromSupabase(): Promise<AgentsPageData> {
  assertSupabaseConfigured();
  const userId = await requireCurrentUserId();
  const [agentsResult, foldersResult] = await Promise.all([
    supabase
      .from("bot_configs_safe")
      .select("id,user_id,folder_id,name,elevenlabs_voice_id,twilio_phone_number,conversation_spec,created_at,updated_at")
      .eq("user_id", userId)
      .order("created_at", { ascending: false }),
    supabase
      .from("agent_folders")
      .select("id,user_id,name,created_at,updated_at")
      .eq("user_id", userId)
      .order("created_at", { ascending: true }),
  ]);
  throwIfSupabaseError(agentsResult.error);
  throwIfSupabaseError(foldersResult.error);
  return {
    agents: agentsResult.data as AgentListItem[],
    folders: foldersResult.data as AgentFolder[],
  };
}

async function getBotFromSupabase(id: string): Promise<BotConfig> {
  assertSupabaseConfigured();
  const userId = await requireCurrentUserId();
  const { data, error } = await supabase
    .from("bot_configs_safe")
    .select("*")
    .eq("id", id)
    .eq("user_id", userId)
    .single();
  throwIfSupabaseError(error);
  return data as BotConfig;
}

async function createBotInSupabase(name: string, folderId?: string | null): Promise<BotConfig> {
  assertSupabaseConfigured();
  const userId = await requireCurrentUserId();
  const { data, error } = await supabase
    .from("bot_configs")
    .insert({ name, user_id: userId, folder_id: folderId ?? null })
    .select("id")
    .single();
  throwIfSupabaseError(error);
  return getBot(data.id);
}

async function updateBotInSupabase(id: string, fields: Partial<BotConfig>): Promise<BotConfig> {
  assertSupabaseConfigured();
  const sanitizedFields = sanitizeBotUpdate(fields);
  if (Object.keys(sanitizedFields).length === 0) {
    return getBot(id);
  }
  const userId = await requireCurrentUserId();
  const { error } = await supabase
    .from("bot_configs")
    .update(sanitizedFields)
    .eq("id", id)
    .eq("user_id", userId);
  throwIfSupabaseError(error);
  return getBot(id);
}

async function deleteBotFromSupabase(id: string): Promise<void> {
  assertSupabaseConfigured();
  const userId = await requireCurrentUserId();
  const { error } = await supabase.from("bot_configs").delete().eq("id", id).eq("user_id", userId);
  throwIfSupabaseError(error);
}

async function listAgentFoldersFromSupabase(): Promise<AgentFolder[]> {
  assertSupabaseConfigured();
  const userId = await requireCurrentUserId();
  const { data, error } = await supabase
    .from("agent_folders")
    .select("*")
    .eq("user_id", userId)
    .order("created_at", { ascending: true });
  throwIfSupabaseError(error);
  return data as AgentFolder[];
}

async function createAgentFolderInSupabase(name: string): Promise<AgentFolder> {
  assertSupabaseConfigured();
  const userId = await requireCurrentUserId();
  const { data, error } = await supabase
    .from("agent_folders")
    .insert({ name, user_id: userId })
    .select("*")
    .single();
  throwIfSupabaseError(error);
  return data as AgentFolder;
}

async function listToolsFromSupabase(): Promise<ToolDefinition[]> {
  assertSupabaseConfigured();
  const userId = await requireCurrentUserId();
  const { data, error } = await supabase
    .from("agent_tools")
    .select("*")
    .eq("user_id", userId)
    .order("created_at", { ascending: false });
  throwIfSupabaseError(error);
  return data as ToolDefinition[];
}

async function createToolInSupabase(data: ToolDefinitionInput): Promise<ToolDefinition> {
  assertSupabaseConfigured();
  const userId = await requireCurrentUserId();
  const { data: created, error } = await supabase
    .from("agent_tools")
    .insert({ ...data, user_id: userId, is_active: true })
    .select("*")
    .single();
  throwIfSupabaseError(error);
  return created as ToolDefinition;
}

async function updateToolInSupabase(id: string, data: ToolDefinitionInput): Promise<ToolDefinition> {
  assertSupabaseConfigured();
  const userId = await requireCurrentUserId();
  const { data: updated, error } = await supabase
    .from("agent_tools")
    .update(data)
    .eq("id", id)
    .eq("user_id", userId)
    .select("*")
    .single();
  throwIfSupabaseError(error);
  return updated as ToolDefinition;
}

async function listKnowledgeBasesFromSupabase(): Promise<KnowledgeBase[]> {
  assertSupabaseConfigured();
  const userId = await requireCurrentUserId();
  const { data, error } = await supabase
    .from("knowledge_bases")
    .select("id,user_id,name,description,created_at,updated_at")
    .eq("user_id", userId)
    .order("created_at", { ascending: false });
  throwIfSupabaseError(error);
  return data as KnowledgeBase[];
}

async function createKnowledgeBaseInSupabase(
  data: Pick<KnowledgeBase, "name" | "description">,
): Promise<KnowledgeBase> {
  assertSupabaseConfigured();
  const userId = await requireCurrentUserId();
  const { data: created, error } = await supabase
    .from("knowledge_bases")
    .insert({ ...data, user_id: userId })
    .select("id,user_id,name,description,created_at,updated_at")
    .single();
  throwIfSupabaseError(error);
  return created as KnowledgeBase;
}

async function listKnowledgeDocumentsFromSupabase(knowledgeBaseId: string): Promise<KnowledgeDocument[]> {
  assertSupabaseConfigured();
  const userId = await requireCurrentUserId();
  const { data, error } = await supabase
    .from("knowledge_documents")
    .select("*")
    .eq("user_id", userId)
    .eq("knowledge_base_id", knowledgeBaseId)
    .order("created_at", { ascending: false });
  throwIfSupabaseError(error);
  return data as KnowledgeDocument[];
}

async function ingestKnowledgeDocumentWithEdgeFunction(
  data: IngestKnowledgeDocumentInput,
): Promise<IngestKnowledgeDocumentResponse> {
  assertSupabaseConfigured();
  const accessToken = await requireAccessToken();
  const body = data.file
    ? (() => {
        const form = new FormData();
        form.set("knowledge_base_id", data.knowledge_base_id);
        form.set("title", data.title);
        form.set("file", data.file);
        if (data.content) form.set("content", data.content);
        if (data.source_url) form.set("source_url", data.source_url);
        if (data.metadata_json) form.set("metadata_json", JSON.stringify(data.metadata_json));
        return form;
      })()
    : JSON.stringify(data);
  const response = await fetch(`${getSupabaseUrl()}/functions/v1/ingest-knowledge-document`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${accessToken}`,
      ...(data.file ? {} : { "Content-Type": "application/json" }),
    },
    body,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `Edge Function failed with ${response.status}`);
  }
  if (payload?.error) {
    throw new Error(String(payload.error));
  }
  return payload as IngestKnowledgeDocumentResponse;
}

async function answerKnowledgeQuestionWithEdgeFunction(
  data: AnswerKnowledgeQuestionInput,
): Promise<AnswerKnowledgeQuestionResponse> {
  assertSupabaseConfigured();
  const accessToken = await requireAccessToken();
  const response = await fetch(`${getSupabaseUrl()}/functions/v1/answer-knowledge-question`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${accessToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `Edge Function failed with ${response.status}`);
  }
  if (payload?.error) {
    throw new Error(String(payload.error));
  }
  return payload as AnswerKnowledgeQuestionResponse;
}

async function listBotKnowledgeBaseIdsFromSupabase(botId: string): Promise<string[]> {
  assertSupabaseConfigured();
  const userId = await requireCurrentUserId();
  const { data, error } = await supabase
    .from("bot_knowledge_bases")
    .select("knowledge_base_id")
    .eq("user_id", userId)
    .eq("bot_id", botId);
  throwIfSupabaseError(error);
  return (data as { knowledge_base_id: string }[]).map((row) => row.knowledge_base_id);
}

async function setBotKnowledgeBasesInSupabase(botId: string, knowledgeBaseIds: string[]): Promise<void> {
  assertSupabaseConfigured();
  const userId = await requireCurrentUserId();
  const uniqueIds = Array.from(new Set(knowledgeBaseIds));
  const { error: deleteError } = await supabase
    .from("bot_knowledge_bases")
    .delete()
    .eq("user_id", userId)
    .eq("bot_id", botId);
  throwIfSupabaseError(deleteError);
  if (uniqueIds.length === 0) return;
  const { error: insertError } = await supabase
    .from("bot_knowledge_bases")
    .insert(uniqueIds.map((knowledge_base_id) => ({ bot_id: botId, knowledge_base_id, user_id: userId })));
  throwIfSupabaseError(insertError);
}

async function listNodeKnowledgeBaseAssignmentsFromSupabase(botId: string): Promise<Record<string, string[]>> {
  assertSupabaseConfigured();
  const userId = await requireCurrentUserId();
  const { data, error } = await supabase
    .from("agent_node_knowledge_bases")
    .select("node_id,knowledge_base_id")
    .eq("user_id", userId)
    .eq("bot_id", botId);
  throwIfSupabaseError(error);
  const assignments: Record<string, string[]> = {};
  for (const row of data as { node_id: string; knowledge_base_id: string }[]) {
    assignments[row.node_id] = [...(assignments[row.node_id] ?? []), row.knowledge_base_id];
  }
  return assignments;
}

async function setNodeKnowledgeBaseAssignmentsInSupabase(
  botId: string,
  assignments: Record<string, string[]>,
): Promise<void> {
  assertSupabaseConfigured();
  const userId = await requireCurrentUserId();
  const { error: deleteError } = await supabase
    .from("agent_node_knowledge_bases")
    .delete()
    .eq("user_id", userId)
    .eq("bot_id", botId);
  throwIfSupabaseError(deleteError);
  const rows = Object.entries(assignments).flatMap(([node_id, ids]) =>
    Array.from(new Set(ids)).map((knowledge_base_id) => ({
      bot_id: botId,
      node_id,
      knowledge_base_id,
      user_id: userId,
    })),
  );
  if (rows.length === 0) return;
  const { error: insertError } = await supabase.from("agent_node_knowledge_bases").insert(rows);
  throwIfSupabaseError(insertError);
}

async function listLeadsFromSupabase(botId: string): Promise<Lead[]> {
  assertSupabaseConfigured();
  const { data, error } = await supabase
    .from("leads")
    .select("*")
    .eq("bot_id", botId)
    .order("created_at", { ascending: false });
  throwIfSupabaseError(error);
  return data as Lead[];
}

async function createLeadInSupabase(
  botId: string,
  data: Omit<Lead, "id" | "bot_id" | "lifecycle_stage" | "created_at">,
): Promise<Lead> {
  assertSupabaseConfigured();
  const { data: created, error } = await supabase
    .from("leads")
    .insert({ ...data, bot_id: botId, lifecycle_stage: "follow_up" })
    .select("*")
    .single();
  throwIfSupabaseError(error);
  return created as Lead;
}

async function listCallsFromSupabase(): Promise<CallLog[]> {
  assertSupabaseConfigured();
  const bots = await listBotsFromSupabase();
  const botIds = bots.map((bot) => bot.id);
  if (botIds.length === 0) return [];

  const { data, error } = await supabase
    .from("call_logs")
    .select("*")
    .in("bot_id", botIds)
    .order("started_at", { ascending: false });
  throwIfSupabaseError(error);
  return data as CallLog[];
}

export function startTestSession(botId: string, leadId: string): Promise<TestSessionResponse> {
  return apiRequest<TestSessionResponse>(`/api/bots/${botId}/test-session`, {
    method: "POST",
    body: JSON.stringify({ lead_id: leadId }),
  });
}

export function sendTestTurn(botId: string, sessionId: string, humanInput: string): Promise<TestSessionResponse> {
  return apiRequest<TestSessionResponse>(`/api/bots/${botId}/test-session/${sessionId}/turns`, {
    method: "POST",
    body: JSON.stringify({ human_input: humanInput }),
  });
}

export function stopTestSession(botId: string, sessionId: string, keepalive = false): Promise<void> {
  return apiRequest<void>(`/api/bots/${botId}/test-session/${sessionId}`, {
    method: "DELETE",
    keepalive,
  });
}
