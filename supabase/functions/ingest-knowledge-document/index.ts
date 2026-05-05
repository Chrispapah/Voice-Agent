// @ts-nocheck - Supabase Edge Functions run in Deno and use URL/npm imports.
import { serve } from "https://deno.land/std@0.224.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2.45.4";
import { extractText, getDocumentProxy } from "npm:unpdf";

type IngestRequest = {
  knowledge_base_id?: string;
  title?: string;
  content?: string;
  source_url?: string | null;
  metadata_json?: Record<string, unknown>;
};

type ParsedRequest = {
  knowledgeBaseId?: string;
  title?: string;
  content?: string;
  file?: File;
  fileName?: string;
  sourceUrl?: string | null;
  mimeType: string;
  metadataJson: Record<string, unknown>;
};

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};

const embeddingModel = Deno.env.get("OPENAI_EMBEDDING_MODEL") || "text-embedding-3-small";
const maxChunkChars = Number(Deno.env.get("KNOWLEDGE_CHUNK_CHARS") || 1400);
const chunkOverlapChars = Number(Deno.env.get("KNOWLEDGE_CHUNK_OVERLAP_CHARS") || 200);

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

async function extractPdfText(file: File): Promise<string> {
  const buffer = await file.arrayBuffer();
  const pdf = await getDocumentProxy(new Uint8Array(buffer));
  const result = await extractText(pdf, { mergePages: true });
  return result.text.trim();
}

async function parseRequest(req: Request): Promise<ParsedRequest> {
  const contentType = req.headers.get("content-type") || "";
  if (contentType.includes("multipart/form-data")) {
    const form = await req.formData();
    const file = form.get("file");
    const title = String(form.get("title") || "").trim();
    const knowledgeBaseId = String(form.get("knowledge_base_id") || "").trim();
    const sourceUrl = String(form.get("source_url") || "").trim() || null;
    const metadataRaw = String(form.get("metadata_json") || "{}");
    const metadataJson = JSON.parse(metadataRaw) as Record<string, unknown>;

    if (file instanceof File && file.size > 0) {
      const mimeType = file.type || "application/octet-stream";
      const content = mimeType === "application/pdf" || file.name.toLowerCase().endsWith(".pdf")
        ? await extractPdfText(file)
        : await file.text();
      return {
        knowledgeBaseId,
        title: title || file.name,
        content,
        file,
        fileName: file.name,
        sourceUrl,
        mimeType,
        metadataJson: { ...metadataJson, file_name: file.name, file_size: file.size },
      };
    }

    return {
      knowledgeBaseId,
      title,
      content: String(form.get("content") || "").trim(),
      sourceUrl,
      mimeType: "text/plain",
      metadataJson,
    };
  }

  const body = (await req.json()) as IngestRequest;
  return {
    knowledgeBaseId: body.knowledge_base_id?.trim(),
    title: body.title?.trim(),
    content: body.content?.trim(),
    sourceUrl: body.source_url ?? null,
    mimeType: "text/plain",
    metadataJson: body.metadata_json ?? {},
  };
}

function sanitizePathPart(value: string): string {
  const normalized = value
    .normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^\x20-\x7E]/g, "")
    .trim()
    .replace(/[/\\?%*:|"<>]/g, "-")
    .replace(/[^A-Za-z0-9._-]/g, "-")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^[.-]+|[.-]+$/g, "")
    .slice(0, 160);
  return normalized || "document";
}

function buildStoragePath(userId: string, knowledgeBaseId: string, fileName: string): string {
  const safeName = sanitizePathPart(fileName);
  const suffix = crypto.randomUUID();
  return `${userId}/${knowledgeBaseId}/${suffix}-${safeName}`;
}

async function uploadOriginalFile({
  adminClient,
  userId,
  knowledgeBaseId,
  file,
  mimeType,
}: {
  adminClient: ReturnType<typeof createClient>;
  userId: string;
  knowledgeBaseId: string;
  file: File;
  mimeType: string;
}): Promise<string> {
  const bucketName = Deno.env.get("KNOWLEDGE_BASE_BUCKET") || "Knowledge Base";
  const { data: buckets, error: bucketsError } = await adminClient.storage.listBuckets();
  if (bucketsError) throw new Error(`Failed to list storage buckets: ${bucketsError.message}`);
  if (!buckets?.some((bucket) => bucket.name === bucketName || bucket.id === bucketName)) {
    const { error: createBucketError } = await adminClient.storage.createBucket(bucketName, {
      public: false,
    });
    if (createBucketError) {
      throw new Error(`Failed to create storage bucket "${bucketName}": ${createBucketError.message}`);
    }
  }

  const storagePath = buildStoragePath(userId, knowledgeBaseId, file.name);
  const fileBytes = new Uint8Array(await file.arrayBuffer());
  const { error } = await adminClient.storage
    .from(bucketName)
    .upload(storagePath, fileBytes, {
      contentType: mimeType || file.type || "application/octet-stream",
      upsert: false,
    });
  if (error) throw new Error(`Failed to upload original file: ${error.message}`);
  return storagePath;
}

function chunkText(content: string): string[] {
  const normalized = content.replace(/\r\n/g, "\n").replace(/\n{3,}/g, "\n\n").trim();
  if (!normalized) return [];

  const paragraphs = normalized.split(/\n\s*\n/).map((part) => part.trim()).filter(Boolean);
  const chunks: string[] = [];
  let current = "";

  for (const paragraph of paragraphs) {
    if (!current) {
      current = paragraph;
      continue;
    }

    if (`${current}\n\n${paragraph}`.length <= maxChunkChars) {
      current = `${current}\n\n${paragraph}`;
      continue;
    }

    chunks.push(current);
    current = paragraph;
  }

  if (current) chunks.push(current);

  const splitChunks = chunks.flatMap((chunk) => {
    if (chunk.length <= maxChunkChars) return [chunk];
    const parts: string[] = [];
    let start = 0;
    while (start < chunk.length) {
      const end = Math.min(start + maxChunkChars, chunk.length);
      parts.push(chunk.slice(start, end).trim());
      if (end === chunk.length) break;
      start = Math.max(end - chunkOverlapChars, start + 1);
    }
    return parts.filter(Boolean);
  });

  return splitChunks.filter((chunk) => chunk.length > 0);
}

async function createEmbeddings(chunks: string[], openAiApiKey: string): Promise<number[][]> {
  const response = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${openAiApiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: embeddingModel,
      input: chunks,
    }),
  });

  if (!response.ok) {
    const errorBody = await response.text();
    throw new Error(`OpenAI embeddings failed (${response.status}): ${errorBody}`);
  }

  const payload = await response.json();
  const embeddings = payload?.data;
  if (!Array.isArray(embeddings) || embeddings.length !== chunks.length) {
    throw new Error("OpenAI embeddings response did not match chunk count");
  }

  return embeddings.map((item: { embedding?: number[] }) => {
    if (!Array.isArray(item.embedding)) throw new Error("OpenAI embedding item missing embedding array");
    return item.embedding;
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
    const supabaseUrl = requireEnv("SUPABASE_URL");
    const supabaseAnonKey = requireEnv("SUPABASE_ANON_KEY");
    const supabaseServiceRoleKey = requireEnv("SUPABASE_SERVICE_ROLE_KEY");
    const openAiApiKey = requireEnv("OPENAI_API_KEY");
    const authorization = req.headers.get("Authorization");

    if (!authorization) {
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
      return jsonResponse({ error: "Not authenticated" }, 401);
    }

    const parsed = await parseRequest(req);
    const knowledgeBaseId = parsed.knowledgeBaseId;
    const title = parsed.title;
    const content = parsed.content?.trim();

    if (!knowledgeBaseId) return jsonResponse({ error: "knowledge_base_id is required" }, 400);
    if (!title) return jsonResponse({ error: "title is required" }, 400);
    if (!content) return jsonResponse({ error: "content is required" }, 400);

    const { data: knowledgeBase, error: knowledgeBaseError } = await adminClient
      .from("knowledge_bases")
      .select("id")
      .eq("id", knowledgeBaseId)
      .eq("user_id", user.id)
      .single();

    if (knowledgeBaseError || !knowledgeBase) {
      return jsonResponse({ error: "Knowledge base not found" }, 404);
    }

    const chunks = chunkText(content);
    if (chunks.length === 0) {
      return jsonResponse({ error: "content did not produce any chunks" }, 400);
    }

    const storagePath = parsed.file
      ? await uploadOriginalFile({
          adminClient,
          userId: user.id,
          knowledgeBaseId,
          file: parsed.file,
          mimeType: parsed.mimeType,
        })
      : null;

    const { data: document, error: documentError } = await adminClient
      .from("knowledge_documents")
      .insert({
        knowledge_base_id: knowledgeBaseId,
        user_id: user.id,
        title,
        source_url: parsed.sourceUrl,
        storage_path: storagePath,
        mime_type: parsed.mimeType,
        status: "processing",
        metadata_json: {
          ...parsed.metadataJson,
          storage_bucket: storagePath ? Deno.env.get("KNOWLEDGE_BASE_BUCKET") || "Knowledge Base" : null,
        },
      })
      .select("*")
      .single();

    if (documentError || !document) {
      throw new Error(documentError?.message || "Failed to create knowledge document");
    }

    try {
      const embeddings = await createEmbeddings(chunks, openAiApiKey);
      const rows = chunks.map((chunk, index) => ({
        knowledge_base_id: knowledgeBaseId,
        document_id: document.id,
        user_id: user.id,
        chunk_index: index,
        content: chunk,
        metadata_json: {
          ...parsed.metadataJson,
          title,
          source_url: parsed.sourceUrl,
        },
        embedding: `[${embeddings[index].join(",")}]`,
      }));

      const { error: chunksError } = await adminClient.from("knowledge_chunks").insert(rows);
      if (chunksError) throw new Error(chunksError.message);

      const { error: updateError } = await adminClient
        .from("knowledge_documents")
        .update({ status: "ready" })
        .eq("id", document.id)
        .eq("user_id", user.id);
      if (updateError) throw new Error(updateError.message);

      return jsonResponse({
        document_id: document.id,
        knowledge_base_id: knowledgeBaseId,
        chunks_created: rows.length,
        status: "ready",
      });
    } catch (error) {
      await adminClient
        .from("knowledge_documents")
        .update({
          status: "failed",
          metadata_json: {
            ...parsed.metadataJson,
            error: error instanceof Error ? error.message : "Unknown ingestion error",
          },
        })
        .eq("id", document.id)
        .eq("user_id", user.id);
      throw error;
    }
  } catch (error) {
    return jsonResponse(
      {
        error: error instanceof Error ? error.message : "Unknown error",
      },
      500,
    );
  }
});
