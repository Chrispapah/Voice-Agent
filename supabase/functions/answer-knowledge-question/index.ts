// @ts-nocheck - Supabase Edge Functions run in Deno and use URL/npm imports.
import { serve } from "https://deno.land/std@0.224.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.45.4";

type AnswerRequest = {
  question?: string;
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
};

type Match = {
  id: string;
  knowledge_base_id: string;
  document_id: string;
  chunk_index?: number;
  content: string;
  metadata_json: Record<string, unknown>;
  similarity: number;
};

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};

const defaultEmbeddingModel = Deno.env.get("OPENAI_EMBEDDING_MODEL") || "text-embedding-3-small";
const defaultAnswerModel = Deno.env.get("OPENAI_ANSWER_MODEL") || "gpt-4o-mini";
const defaultWholeKbMaxChunks = readPositiveIntEnv("KNOWLEDGE_WHOLE_KB_MAX_CHUNKS", 80);
const defaultWholeKbMaxContextChars = readPositiveIntEnv("KNOWLEDGE_WHOLE_KB_MAX_CONTEXT_CHARS", 90000);
const defaultHybridMaxMatches = readPositiveIntEnv("KNOWLEDGE_HYBRID_MAX_MATCHES", 18);

function logStep(step: string, payload: Record<string, unknown> = {}): void {
  console.log(JSON.stringify({ step, ...payload }));
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      ...corsHeaders,
      "Content-Type": "application/json",
    },
  });
}

function requireEnv(name: string): string {
  const value = Deno.env.get(name);
  if (!value) throw new Error(`Missing ${name}`);
  return value;
}

function readPositiveIntEnv(name: string, fallback: number): number {
  const value = Number(Deno.env.get(name) || fallback);
  if (!Number.isFinite(value) || value < 1) return fallback;
  return Math.floor(value);
}

function readBoundedPositiveInt(value: unknown, fallback: number, min: number, max: number): number {
  const numericValue = Number(value ?? fallback);
  if (!Number.isFinite(numericValue)) return fallback;
  return Math.max(min, Math.min(Math.floor(numericValue), max));
}

function readOptionalText(value: unknown, fallback: string): string {
  return typeof value === "string" && value.trim() ? value.trim() : fallback;
}

function extractSearchTerms(question: string): string[] {
  const stopWords = new Set([
    "about",
    "what",
    "when",
    "where",
    "which",
    "tell",
    "give",
    "with",
    "from",
    "τους",
    "των",
    "την",
    "της",
    "και",
    "για",
    "μου",
    "στο",
    "στη",
    "στα",
  ]);
  const terms = question.match(/[\p{L}\p{N}]{4,}/gu) ?? [];
  return Array.from(
    new Set(
      terms
        .map((term) => term.toLocaleLowerCase())
        .filter((term) => !stopWords.has(term)),
    ),
  ).slice(0, 8);
}

async function fetchWholeKbMatches({
  adminClient,
  requestId,
  userId,
  knowledgeBaseIds,
  maxChunks,
  maxContextChars,
}: {
  adminClient: ReturnType<typeof createClient>;
  requestId: string;
  userId: string;
  knowledgeBaseIds: string[];
  maxChunks: number;
  maxContextChars: number;
}): Promise<Match[] | null> {
  const { count, error: countError } = await adminClient
    .from("knowledge_chunks")
    .select("id", { count: "exact", head: true })
    .eq("user_id", userId)
    .in("knowledge_base_id", knowledgeBaseIds);

  if (countError) {
    logStep("whole_kb:checked", {
      request_id: requestId,
      retrieval_mode: "hybrid",
      reason: "count_error",
      error: countError.message,
    });
    return null;
  }

  const chunkCount = count ?? 0;
  if (chunkCount > maxChunks) {
    logStep("whole_kb:checked", {
      request_id: requestId,
      retrieval_mode: "hybrid",
      reason: "too_many_chunks",
      chunk_count: chunkCount,
      max_chunks: maxChunks,
    });
    return null;
  }

  const { data, error } = await adminClient
    .from("knowledge_chunks")
    .select("id,knowledge_base_id,document_id,chunk_index,content,metadata_json")
    .eq("user_id", userId)
    .in("knowledge_base_id", knowledgeBaseIds)
    .order("knowledge_base_id", { ascending: true })
    .order("document_id", { ascending: true })
    .order("chunk_index", { ascending: true })
    .limit(Math.max(maxChunks, 1));

  if (error) {
    logStep("whole_kb:checked", {
      request_id: requestId,
      retrieval_mode: "hybrid",
      reason: "fetch_error",
      chunk_count: chunkCount,
      error: error.message,
    });
    return null;
  }

  const matches = ((data ?? []) as Omit<Match, "similarity">[]).map((match) => ({
    ...match,
    similarity: 1,
  }));
  const contextChars = matches.reduce((total, match) => total + (match.content?.length ?? 0), 0);

  if (contextChars > maxContextChars) {
    logStep("whole_kb:checked", {
      request_id: requestId,
      retrieval_mode: "hybrid",
      reason: "too_many_context_chars",
      chunk_count: chunkCount,
      context_chars: contextChars,
      max_context_chars: maxContextChars,
    });
    return null;
  }

  logStep("whole_kb:checked", {
    request_id: requestId,
    retrieval_mode: "whole_kb",
    chunk_count: chunkCount,
    returned_count: matches.length,
    context_chars: contextChars,
    max_chunks: maxChunks,
    max_context_chars: maxContextChars,
    top_preview: matches[0]?.content?.slice(0, 160) ?? null,
  });
  return matches;
}

async function fetchKeywordMatches({
  adminClient,
  requestId,
  userId,
  question,
  knowledgeBaseIds,
  limit,
}: {
  adminClient: ReturnType<typeof createClient>;
  requestId: string;
  userId: string;
  question: string;
  knowledgeBaseIds: string[];
  limit: number;
}): Promise<Match[]> {
  const searchTerms = extractSearchTerms(question);
  if (searchTerms.length === 0) {
    logStep("keyword_match:done", {
      request_id: requestId,
      reason: "no_search_terms",
      terms: searchTerms,
      returned_count: 0,
      top_preview: null,
    });
    return [];
  }

  const orFilter = searchTerms.map((term) => `content.ilike.%${term}%`).join(",");
  const { data, error } = await adminClient
    .from("knowledge_chunks")
    .select("id,knowledge_base_id,document_id,chunk_index,content,metadata_json")
    .eq("user_id", userId)
    .in("knowledge_base_id", knowledgeBaseIds)
    .or(orFilter)
    .order("chunk_index", { ascending: true })
    .limit(limit);

  if (error) {
    logStep("keyword_match:error", {
      request_id: requestId,
      terms: searchTerms,
      error: error.message,
    });
    return [];
  }

  const matches = ((data ?? []) as Omit<Match, "similarity">[]).map((match) => ({
    ...match,
    similarity: 0,
  }));
  logStep("keyword_match:done", {
    request_id: requestId,
    terms: searchTerms,
    returned_count: matches.length,
    top_preview: matches[0]?.content?.slice(0, 160) ?? null,
  });
  return matches;
}

function mergeMatches({
  matches,
  limit,
}: {
  matches: Match[];
  limit: number;
}): Match[] {
  const seen = new Set<string>();
  const merged: Match[] = [];

  for (const match of matches) {
    if (seen.has(match.id)) continue;
    seen.add(match.id);
    merged.push(match);
    if (merged.length >= limit) break;
  }

  return merged;
}

function buildNeighborRangeFilter(matches: Match[]): string | null {
  const rangesByDocument = new Map<string, { start: number; end: number }[]>();

  for (const match of matches) {
    if (typeof match.chunk_index !== "number") continue;
    const ranges = rangesByDocument.get(match.document_id) ?? [];
    ranges.push({
      start: Math.max(0, match.chunk_index - 1),
      end: match.chunk_index + 1,
    });
    rangesByDocument.set(match.document_id, ranges);
  }

  const filters: string[] = [];
  for (const [documentId, ranges] of rangesByDocument.entries()) {
    const mergedRanges = ranges
      .sort((left, right) => left.start - right.start)
      .reduce<{ start: number; end: number }[]>((merged, range) => {
        const previous = merged[merged.length - 1];
        if (!previous || range.start > previous.end + 1) {
          merged.push({ ...range });
          return merged;
        }

        previous.end = Math.max(previous.end, range.end);
        return merged;
      }, []);

    filters.push(
      ...mergedRanges.map((range) =>
        `and(document_id.eq.${documentId},chunk_index.gte.${range.start},chunk_index.lte.${range.end})`
      ),
    );
  }

  return filters.length > 0 ? filters.join(",") : null;
}

async function fetchNeighborMatches({
  adminClient,
  requestId,
  userId,
  knowledgeBaseIds,
  matches,
}: {
  adminClient: ReturnType<typeof createClient>;
  requestId: string;
  userId: string;
  knowledgeBaseIds: string[];
  matches: Match[];
}): Promise<Match[]> {
  const rangeFilter = buildNeighborRangeFilter(matches);
  if (!rangeFilter) {
    logStep("neighbor_match:done", {
      request_id: requestId,
      seed_count: matches.length,
      returned_count: 0,
      batched: true,
      top_preview: null,
    });
    return [];
  }

  const { data, error } = await adminClient
    .from("knowledge_chunks")
    .select("id,knowledge_base_id,document_id,chunk_index,content,metadata_json")
    .eq("user_id", userId)
    .in("knowledge_base_id", knowledgeBaseIds)
    .or(rangeFilter)
    .order("document_id", { ascending: true })
    .order("chunk_index", { ascending: true });

  if (error) {
    logStep("neighbor_match:error", {
      request_id: requestId,
      seed_count: matches.length,
      batched: true,
      error: error.message,
    });
    return [];
  }

  const neighborSimilarityByKey = new Map<string, number>();
  for (const match of matches) {
    if (typeof match.chunk_index !== "number") continue;
    for (let chunkIndex = Math.max(0, match.chunk_index - 1); chunkIndex <= match.chunk_index + 1; chunkIndex += 1) {
      const key = `${match.document_id}:${chunkIndex}`;
      neighborSimilarityByKey.set(key, Math.max(neighborSimilarityByKey.get(key) ?? 0, match.similarity ?? 0));
    }
  }

  const neighbors = ((data ?? []) as Omit<Match, "similarity">[]).map((neighbor) => ({
    ...neighbor,
    similarity: neighborSimilarityByKey.get(`${neighbor.document_id}:${neighbor.chunk_index}`) ?? 0,
  }));

  logStep("neighbor_match:done", {
    request_id: requestId,
    seed_count: matches.length,
    returned_count: neighbors.length,
    batched: true,
    top_preview: neighbors[0]?.content?.slice(0, 160) ?? null,
  });
  return neighbors;
}

function countTermHits(content: string, searchTerms: string[]): number {
  const normalizedContent = content.toLocaleLowerCase();
  return searchTerms.reduce((hits, term) => hits + (normalizedContent.includes(term) ? 1 : 0), 0);
}

function scoreAndRankMatches({
  question,
  vectorMatches,
  keywordMatches,
  neighborMatches,
  limit,
}: {
  question: string;
  vectorMatches: Match[];
  keywordMatches: Match[];
  neighborMatches: Match[];
  limit: number;
}): Match[] {
  const searchTerms = extractSearchTerms(question);
  const vectorById = new Map(vectorMatches.map((match, index) => [match.id, { match, rank: index }]));
  const keywordById = new Map(keywordMatches.map((match, index) => [match.id, { match, rank: index }]));
  const neighborIds = new Set(neighborMatches.map((match) => match.id));
  const candidatesById = new Map<string, Match>();

  for (const match of [...vectorMatches, ...keywordMatches, ...neighborMatches]) {
    if (!candidatesById.has(match.id)) candidatesById.set(match.id, match);
  }

  return Array.from(candidatesById.values())
    .map((match) => {
      const vector = vectorById.get(match.id);
      const keyword = keywordById.get(match.id);
      const termHits = searchTerms.length > 0 ? countTermHits(match.content, searchTerms) : 0;
      const vectorSimilarity = vector?.match.similarity ?? 0;
      const vectorRankBoost = vector ? Math.max(0, 1 - vector.rank / Math.max(vectorMatches.length, 1)) : 0;
      const keywordCoverage = searchTerms.length > 0 ? termHits / searchTerms.length : 0;
      const keywordRankBoost = keyword ? Math.max(0, 1 - keyword.rank / Math.max(keywordMatches.length, 1)) : 0;
      const neighborBoost = neighborIds.has(match.id) && !vector && !keyword ? 1 : 0;
      const retrievalScore =
        vectorSimilarity * 0.55 +
        vectorRankBoost * 0.15 +
        keywordCoverage * 0.2 +
        keywordRankBoost * 0.07 +
        neighborBoost * 0.03;

      return {
        ...match,
        similarity: vector?.match.similarity ?? match.similarity ?? 0,
        retrieval_score: Number(retrievalScore.toFixed(4)),
        retrieval_reasons: [
          vector ? "vector" : null,
          keyword || termHits > 0 ? "keyword" : null,
          neighborIds.has(match.id) ? "neighbor" : null,
        ].filter(Boolean),
        keyword_hits: termHits,
      };
    })
    .sort((left, right) => {
      if (right.retrieval_score !== left.retrieval_score) return right.retrieval_score - left.retrieval_score;
      if (left.document_id !== right.document_id) return left.document_id.localeCompare(right.document_id);
      return (left.chunk_index ?? 0) - (right.chunk_index ?? 0);
    })
    .slice(0, limit);
}

async function createEmbedding(question: string, openAiApiKey: string, model: string): Promise<number[]> {
  logStep("embedding:start", {
    model,
    question_chars: question.length,
  });
  const response = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${openAiApiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model,
      input: question,
    }),
  });

  if (!response.ok) {
    const errorBody = await response.text();
    throw new Error(`OpenAI embedding failed (${response.status}): ${errorBody}`);
  }

  const payload = await response.json();
  const embedding = payload?.data?.[0]?.embedding;
  if (!Array.isArray(embedding)) throw new Error("OpenAI embedding response missing embedding");
  if (embedding.length !== 1536) {
    throw new Error(`Embedding model "${model}" returned ${embedding.length} dimensions, but this index expects 1536`);
  }
  logStep("embedding:done", {
    dimensions: embedding.length,
  });
  return embedding;
}

async function answerFromContext({
  question,
  matches,
  openAiApiKey,
  model,
}: {
  question: string;
  matches: Match[];
  openAiApiKey: string;
  model: string;
}): Promise<string> {
  const context = matches
    .map((match, index) => {
      const source = typeof match.metadata_json?.title === "string" ? match.metadata_json.title : match.document_id;
      return `[${index + 1}] Source: ${source}\n${match.content}`;
    })
    .join("\n\n---\n\n");
  logStep("answer:start", {
    model,
    match_count: matches.length,
    context_chars: context.length,
    top_similarity: matches[0]?.similarity ?? null,
  });

  const response = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${openAiApiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model,
      temperature: 0,
      messages: [
        {
          role: "system",
          content:
            "You answer questions using only the provided knowledge base context. " +
            "If the answer is not present in the context, say you do not know. " +
            "Answer in the same language as the question. " +
            "Be concise and cite source numbers like [1] when useful.",
        },
        {
          role: "user",
          content: `Knowledge base context:\n${context || "(no context found)"}\n\nQuestion: ${question}`,
        },
      ],
    }),
  });

  if (!response.ok) {
    const errorBody = await response.text();
    throw new Error(`OpenAI answer failed (${response.status}): ${errorBody}`);
  }

  const payload = await response.json();
  const answer = payload?.choices?.[0]?.message?.content;
  if (typeof answer !== "string") throw new Error("OpenAI answer response missing content");
  logStep("answer:done", {
    answer_chars: answer.length,
    answer_preview: answer.slice(0, 160),
  });
  return answer;
}

async function resolveKnowledgeBaseIds({
  adminClient,
  userId,
  request,
}: {
  adminClient: ReturnType<typeof createClient>;
  userId: string;
  request: AnswerRequest;
}): Promise<string[]> {
  if (Array.isArray(request.knowledge_base_ids) && request.knowledge_base_ids.length > 0) {
    const uniqueIds = Array.from(new Set(request.knowledge_base_ids.filter(Boolean)));
    logStep("kb:resolve:explicit:start", {
      requested_count: uniqueIds.length,
      requested_ids: uniqueIds,
    });
    const { data, error } = await adminClient
      .from("knowledge_bases")
      .select("id")
      .eq("user_id", userId)
      .in("id", uniqueIds);
    if (error) throw new Error(error.message);
    const resolved = (data as { id: string }[]).map((row) => row.id);
    logStep("kb:resolve:explicit:done", {
      resolved_count: resolved.length,
      resolved_ids: resolved,
    });
    return resolved;
  }

  if (!request.bot_id) return [];

  if (request.node_id) {
    logStep("kb:resolve:node:start", {
      bot_id: request.bot_id,
      node_id: request.node_id,
    });
    const { data: nodeRows, error: nodeError } = await adminClient
      .from("agent_node_knowledge_bases")
      .select("knowledge_base_id")
      .eq("user_id", userId)
      .eq("bot_id", request.bot_id)
      .eq("node_id", request.node_id);
    if (nodeError) throw new Error(nodeError.message);
    const nodeIds = (nodeRows as { knowledge_base_id: string }[]).map((row) => row.knowledge_base_id);
    if (nodeIds.length > 0) {
      const resolved = Array.from(new Set(nodeIds));
      logStep("kb:resolve:node:done", {
        resolved_count: resolved.length,
        resolved_ids: resolved,
      });
      return resolved;
    }
  }

  logStep("kb:resolve:bot:start", {
    bot_id: request.bot_id,
  });
  const { data: botRows, error: botError } = await adminClient
    .from("bot_knowledge_bases")
    .select("knowledge_base_id")
    .eq("user_id", userId)
    .eq("bot_id", request.bot_id);
  if (botError) throw new Error(botError.message);
  const resolved = Array.from(new Set((botRows as { knowledge_base_id: string }[]).map((row) => row.knowledge_base_id)));
  logStep("kb:resolve:bot:done", {
    resolved_count: resolved.length,
    resolved_ids: resolved,
  });
  return resolved;
}

async function logKnowledgeBaseDiagnostics({
  adminClient,
  requestId,
  userId,
  knowledgeBaseIds,
}: {
  adminClient: ReturnType<typeof createClient>;
  requestId: string;
  userId: string;
  knowledgeBaseIds: string[];
}): Promise<void> {
  const { data: documents, error: documentsError } = await adminClient
    .from("knowledge_documents")
    .select("id,title,status,mime_type,storage_path,metadata_json")
    .eq("user_id", userId)
    .in("knowledge_base_id", knowledgeBaseIds)
    .order("created_at", { ascending: false })
    .limit(5);

  if (documentsError) {
    logStep("kb:diagnostics:documents:error", {
      request_id: requestId,
      error: documentsError.message,
    });
  } else {
    logStep("kb:diagnostics:documents", {
      request_id: requestId,
      document_count_sample: documents?.length ?? 0,
      documents: (documents ?? []).map((document) => ({
        id: document.id,
        title: document.title,
        status: document.status,
        mime_type: document.mime_type,
        storage_path: document.storage_path,
        metadata_json: document.metadata_json,
      })),
    });
  }

  const { count: chunkCount, error: chunkCountError } = await adminClient
    .from("knowledge_chunks")
    .select("id", { count: "exact", head: true })
    .eq("user_id", userId)
    .in("knowledge_base_id", knowledgeBaseIds);

  const { count: embeddingCount, error: embeddingCountError } = await adminClient
    .from("knowledge_chunks")
    .select("id", { count: "exact", head: true })
    .eq("user_id", userId)
    .in("knowledge_base_id", knowledgeBaseIds)
    .not("embedding", "is", null);

  const { data: sampleChunks, error: sampleChunksError } = await adminClient
    .from("knowledge_chunks")
    .select("id,document_id,chunk_index,content,metadata_json")
    .eq("user_id", userId)
    .in("knowledge_base_id", knowledgeBaseIds)
    .order("chunk_index", { ascending: true })
    .limit(3);

  logStep("kb:diagnostics:chunks", {
    request_id: requestId,
    chunk_count: chunkCount ?? null,
    chunk_count_error: chunkCountError?.message ?? null,
    embedding_count: embeddingCount ?? null,
    embedding_count_error: embeddingCountError?.message ?? null,
    sample_error: sampleChunksError?.message ?? null,
    samples: (sampleChunks ?? []).map((chunk) => ({
      id: chunk.id,
      document_id: chunk.document_id,
      chunk_index: chunk.chunk_index,
      content_chars: chunk.content?.length ?? 0,
      content_preview: chunk.content?.slice(0, 240) ?? null,
      metadata_json: chunk.metadata_json,
    })),
  });
}

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }

  if (req.method !== "POST") {
    return jsonResponse({ error: "Method not allowed" }, 405);
  }

  try {
    const requestId = crypto.randomUUID();
    logStep("request:start", {
      request_id: requestId,
      method: req.method,
    });
    const supabaseUrl = requireEnv("SUPABASE_URL");
    const supabaseAnonKey = requireEnv("SUPABASE_ANON_KEY");
    const supabaseServiceRoleKey = requireEnv("SUPABASE_SERVICE_ROLE_KEY");
    const authorization = req.headers.get("Authorization");

    if (!authorization) {
      logStep("auth:missing", { request_id: requestId });
      return jsonResponse({ error: "Missing Authorization header" }, 401);
    }

    const userClient = createClient(supabaseUrl, supabaseAnonKey, {
      global: { headers: { Authorization: authorization } },
    });
    const adminClient = createClient(supabaseUrl, supabaseServiceRoleKey);

    const {
      data: { user },
      error: userError,
    } = await userClient.auth.getUser();

    if (userError || !user) {
      logStep("auth:failed", {
        request_id: requestId,
        error: userError?.message ?? null,
      });
      return jsonResponse({ error: "Not authenticated" }, 401);
    }
    logStep("auth:ok", {
      request_id: requestId,
      user_id: user.id,
    });

    const body = (await req.json()) as AnswerRequest;
    const question = body.question?.trim();
    const matchCount = Math.max(1, Math.min(Number(body.match_count || 5), 12));
    const wholeKbMaxChunks = readBoundedPositiveInt(body.whole_kb_max_chunks, defaultWholeKbMaxChunks, 1, 500);
    const wholeKbMaxContextChars = readBoundedPositiveInt(
      body.whole_kb_max_context_chars,
      defaultWholeKbMaxContextChars,
      1000,
      300000,
    );
    const hybridMaxMatches = readBoundedPositiveInt(body.hybrid_max_matches, defaultHybridMaxMatches, 1, 40);
    const llmProvider = readOptionalText(body.llm_provider, "openai").toLowerCase();
    const answerModel = readOptionalText(body.answer_model, defaultAnswerModel);
    const embeddingModel = readOptionalText(body.embedding_model, defaultEmbeddingModel);
    const openAiApiKey = readOptionalText(body.openai_api_key, Deno.env.get("OPENAI_API_KEY") || "");
    logStep("request:body", {
      request_id: requestId,
      question_chars: question?.length ?? 0,
      match_count: matchCount,
      whole_kb_max_chunks: wholeKbMaxChunks,
      whole_kb_max_context_chars: wholeKbMaxContextChars,
      hybrid_max_matches: hybridMaxMatches,
      llm_provider: llmProvider,
      answer_model: answerModel,
      embedding_model: embeddingModel,
      has_request_openai_api_key: typeof body.openai_api_key === "string" && body.openai_api_key.trim().length > 0,
      has_explicit_kbs: Array.isArray(body.knowledge_base_ids),
      explicit_kb_count: body.knowledge_base_ids?.length ?? 0,
      bot_id: body.bot_id ?? null,
      node_id: body.node_id ?? null,
    });

    if (!question) {
      logStep("request:invalid", { request_id: requestId, reason: "missing_question" });
      return jsonResponse({ error: "question is required" }, 400);
    }

    if (llmProvider !== "openai") {
      logStep("request:invalid", { request_id: requestId, reason: "unsupported_llm_provider", llm_provider: llmProvider });
      return jsonResponse({ error: `Unsupported llm_provider "${llmProvider}". Only "openai" is currently supported.` }, 400);
    }

    if (!openAiApiKey) {
      throw new Error("Missing OPENAI_API_KEY or request openai_api_key");
    }

    const knowledgeBaseIds = await resolveKnowledgeBaseIds({
      adminClient,
      userId: user.id,
      request: body,
    });

    if (knowledgeBaseIds.length === 0) {
      logStep("kb:resolve:none", { request_id: requestId });
      return jsonResponse({
        answer: "I do not know because no knowledge base is assigned.",
        matches: [],
        knowledge_base_ids: [],
      });
    }

    await logKnowledgeBaseDiagnostics({
      adminClient,
      requestId,
      userId: user.id,
      knowledgeBaseIds,
    });

    const wholeKbMatches = await fetchWholeKbMatches({
      adminClient,
      requestId,
      userId: user.id,
      knowledgeBaseIds,
      maxChunks: wholeKbMaxChunks,
      maxContextChars: wholeKbMaxContextChars,
    });

    if (wholeKbMatches) {
      const answer = await answerFromContext({
        question,
        matches: wholeKbMatches,
        openAiApiKey,
        model: answerModel,
      });

      logStep("request:done", {
        request_id: requestId,
        retrieval_mode: "whole_kb",
        returned_matches: wholeKbMatches.length,
      });
      return jsonResponse({
        answer,
        matches: wholeKbMatches.map((match) => ({
          id: match.id,
          knowledge_base_id: match.knowledge_base_id,
          document_id: match.document_id,
          chunk_index: match.chunk_index,
          similarity: match.similarity,
          content_preview: match.content.slice(0, 400),
          metadata_json: match.metadata_json,
        })),
        knowledge_base_ids: knowledgeBaseIds,
        retrieval_mode: "whole_kb",
        retrieval_config: {
          llm_provider: llmProvider,
          answer_model: answerModel,
          embedding_model: embeddingModel,
          uses_request_openai_api_key: typeof body.openai_api_key === "string" && body.openai_api_key.trim().length > 0,
          match_count: matchCount,
          whole_kb_max_chunks: wholeKbMaxChunks,
          whole_kb_max_context_chars: wholeKbMaxContextChars,
          hybrid_max_matches: hybridMaxMatches,
        },
      });
    }

    const embedding = await createEmbedding(question, openAiApiKey, embeddingModel);
    logStep("match:start", {
      request_id: requestId,
      knowledge_base_ids: knowledgeBaseIds,
      match_count: matchCount,
      client: "service_role",
      rpc: "match_knowledge_chunks_for_user",
    });
    const { data: matches, error: matchError } = await adminClient.rpc("match_knowledge_chunks_for_user", {
      query_embedding: `[${embedding.join(",")}]`,
      match_user_id: user.id,
      match_count: matchCount,
      knowledge_base_ids: knowledgeBaseIds,
    });

    if (matchError) throw new Error(matchError.message);

    const typedMatches = (matches || []) as Match[];
    const keywordMatches = await fetchKeywordMatches({
      adminClient,
      requestId,
      userId: user.id,
      question,
      knowledgeBaseIds,
      limit: matchCount,
    });
    const seedMatches = mergeMatches({
      matches: [...typedMatches, ...keywordMatches],
      limit: Math.min(matchCount + 4, 12),
    });
    const neighborMatches = await fetchNeighborMatches({
      adminClient,
      requestId,
      userId: user.id,
      knowledgeBaseIds,
      matches: seedMatches,
    });
    const combinedMatches = scoreAndRankMatches({
      question,
      vectorMatches: typedMatches,
      keywordMatches,
      neighborMatches,
      limit: hybridMaxMatches,
    });
    logStep("rerank:done", {
      request_id: requestId,
      strategy: "weighted_scoring",
      candidate_count: mergeMatches({
        matches: [...typedMatches, ...keywordMatches, ...neighborMatches],
        limit: Number.MAX_SAFE_INTEGER,
      }).length,
      returned_count: combinedMatches.length,
      top_score: combinedMatches[0]?.retrieval_score ?? null,
      top_reasons: combinedMatches[0]?.retrieval_reasons ?? null,
      top_preview: combinedMatches[0]?.content?.slice(0, 160) ?? null,
    });
    logStep("match:done", {
      request_id: requestId,
      retrieval_mode: "hybrid",
      returned_count: typedMatches.length,
      top_similarity: typedMatches[0]?.similarity ?? null,
      top_document_id: typedMatches[0]?.document_id ?? null,
      top_preview: typedMatches[0]?.content?.slice(0, 160) ?? null,
      keyword_count: keywordMatches.length,
      neighbor_count: neighborMatches.length,
      combined_count: combinedMatches.length,
      top_retrieval_score: combinedMatches[0]?.retrieval_score ?? null,
      top_retrieval_reasons: combinedMatches[0]?.retrieval_reasons ?? null,
      combined_top_preview: combinedMatches[0]?.content?.slice(0, 160) ?? null,
    });
    const answer = await answerFromContext({
      question,
      matches: combinedMatches,
      openAiApiKey,
      model: answerModel,
    });

    logStep("request:done", {
      request_id: requestId,
      retrieval_mode: "hybrid",
      returned_matches: combinedMatches.length,
    });
    return jsonResponse({
      answer,
      matches: combinedMatches.map((match) => ({
        id: match.id,
        knowledge_base_id: match.knowledge_base_id,
        document_id: match.document_id,
        chunk_index: match.chunk_index,
        similarity: match.similarity,
        retrieval_score: match.retrieval_score,
        retrieval_reasons: match.retrieval_reasons,
        keyword_hits: match.keyword_hits,
        content_preview: match.content.slice(0, 400),
        metadata_json: match.metadata_json,
      })),
      knowledge_base_ids: knowledgeBaseIds,
      retrieval_mode: "hybrid",
      retrieval_config: {
        llm_provider: llmProvider,
        answer_model: answerModel,
        embedding_model: embeddingModel,
        uses_request_openai_api_key: typeof body.openai_api_key === "string" && body.openai_api_key.trim().length > 0,
        match_count: matchCount,
        whole_kb_max_chunks: wholeKbMaxChunks,
        whole_kb_max_context_chars: wholeKbMaxContextChars,
        hybrid_max_matches: hybridMaxMatches,
      },
    });
  } catch (error) {
    logStep("request:error", {
      error: error instanceof Error ? error.message : "Unknown error",
    });
    return jsonResponse(
      {
        error: error instanceof Error ? error.message : "Unknown error",
      },
      500,
    );
  }
});
