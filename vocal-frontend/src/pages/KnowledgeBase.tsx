import { ChangeEvent, FormEvent, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { BookOpen, FileText, Plus, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  answerKnowledgeQuestion,
  AuthRequiredError,
  createKnowledgeBase,
  deleteKnowledgeBase,
  ingestKnowledgeDocument,
  listKnowledgeBases,
  listKnowledgeDocuments,
  type KnowledgeBase,
  type KnowledgeDocument,
} from "@/lib/api";

export default function KnowledgeBasePage() {
  const navigate = useNavigate();
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBase[]>([]);
  const [documents, setDocuments] = useState<KnowledgeDocument[]>([]);
  const [selectedKnowledgeBaseId, setSelectedKnowledgeBaseId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingDocuments, setLoadingDocuments] = useState(false);
  const [creating, setCreating] = useState(false);
  const [ingesting, setIngesting] = useState(false);
  const [deletingKb, setDeletingKb] = useState(false);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [showDocumentForm, setShowDocumentForm] = useState(false);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [documentTitle, setDocumentTitle] = useState("");
  const [documentSourceUrl, setDocumentSourceUrl] = useState("");
  const [documentContent, setDocumentContent] = useState("");
  const [documentFile, setDocumentFile] = useState<File | null>(null);
  const [question, setQuestion] = useState("What is this document about?");
  const [matchCount, setMatchCount] = useState(5);
  const [wholeKbMaxChunks, setWholeKbMaxChunks] = useState(80);
  const [wholeKbMaxContextChars, setWholeKbMaxContextChars] = useState(90000);
  const [hybridMaxMatches, setHybridMaxMatches] = useState(18);
  const [llmProvider, setLlmProvider] = useState("openai");
  const [answerModel, setAnswerModel] = useState("gpt-4.1-nano");
  const [embeddingModel, setEmbeddingModel] = useState("text-embedding-3-small");
  const [openAiApiKey, setOpenAiApiKey] = useState("");
  const [answer, setAnswer] = useState("");
  const [asking, setAsking] = useState(false);
  const [error, setError] = useState("");
  const [kbPendingDelete, setKbPendingDelete] = useState<KnowledgeBase | null>(null);

  const selectedKnowledgeBase = useMemo(
    () => knowledgeBases.find((kb) => kb.id === selectedKnowledgeBaseId) ?? null,
    [knowledgeBases, selectedKnowledgeBaseId],
  );

  useEffect(() => {
    listKnowledgeBases()
      .then((loaded) => {
        setKnowledgeBases(loaded);
        setSelectedKnowledgeBaseId(loaded[0]?.id ?? null);
      })
      .catch((err: unknown) => {
        if (err instanceof AuthRequiredError) {
          navigate("/auth");
          return;
        }
        setError(err instanceof Error ? err.message : "Failed to load knowledge bases");
      })
      .finally(() => setLoading(false));
  }, [navigate]);

  useEffect(() => {
    if (!selectedKnowledgeBaseId) {
      setDocuments([]);
      return;
    }
    setLoadingDocuments(true);
    listKnowledgeDocuments(selectedKnowledgeBaseId)
      .then(setDocuments)
      .catch((err: unknown) => setError(err instanceof Error ? err.message : "Failed to load documents"))
      .finally(() => setLoadingDocuments(false));
  }, [selectedKnowledgeBaseId]);

  async function handleCreateKnowledgeBase(e: FormEvent) {
    e.preventDefault();
    if (!name.trim()) return;
    setCreating(true);
    setError("");
    try {
      const created = await createKnowledgeBase({
        name: name.trim(),
        description: description.trim(),
      });
      setKnowledgeBases((current) => [created, ...current]);
      setSelectedKnowledgeBaseId(created.id);
      setName("");
      setDescription("");
      setShowCreateForm(false);
    } catch (err: unknown) {
      if (err instanceof AuthRequiredError) {
        navigate("/auth");
        return;
      }
      setError(err instanceof Error ? err.message : "Failed to create knowledge base");
    } finally {
      setCreating(false);
    }
  }

  async function refreshDocuments(knowledgeBaseId: string) {
    setLoadingDocuments(true);
    try {
      const loaded = await listKnowledgeDocuments(knowledgeBaseId);
      setDocuments(loaded);
    } finally {
      setLoadingDocuments(false);
    }
  }

  async function handleIngestDocument(e: FormEvent) {
    e.preventDefault();
    if (!selectedKnowledgeBaseId || !documentTitle.trim() || (!documentContent.trim() && !documentFile)) return;
    setIngesting(true);
    setError("");
    try {
      await ingestKnowledgeDocument({
        knowledge_base_id: selectedKnowledgeBaseId,
        title: documentTitle.trim(),
        content: documentContent.trim() || undefined,
        file: documentFile ?? undefined,
        source_url: documentSourceUrl.trim() || null,
      });
      setDocumentTitle("");
      setDocumentSourceUrl("");
      setDocumentContent("");
      setDocumentFile(null);
      setShowDocumentForm(false);
      await refreshDocuments(selectedKnowledgeBaseId);
    } catch (err: unknown) {
      if (err instanceof AuthRequiredError) {
        navigate("/auth");
        return;
      }
      setError(err instanceof Error ? err.message : "Failed to ingest document");
    } finally {
      setIngesting(false);
    }
  }

  async function handleDocumentFile(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setDocumentTitle((current) => current || file.name);
    setDocumentSourceUrl("");
    setDocumentFile(file);
    if (file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) {
      setDocumentContent(await file.text());
      return;
    }
    setDocumentContent("");
  }

  async function handleAskQuestion(e: FormEvent) {
    e.preventDefault();
    if (!selectedKnowledgeBaseId || !question.trim()) return;
    setAsking(true);
    setError("");
    setAnswer("");
    try {
      const response = await answerKnowledgeQuestion({
        question: question.trim(),
        knowledge_base_ids: [selectedKnowledgeBaseId],
        match_count: matchCount,
        whole_kb_max_chunks: wholeKbMaxChunks,
        whole_kb_max_context_chars: wholeKbMaxContextChars,
        hybrid_max_matches: hybridMaxMatches,
        llm_provider: llmProvider,
        answer_model: answerModel.trim() || undefined,
        embedding_model: embeddingModel.trim() || undefined,
        openai_api_key: openAiApiKey.trim() || undefined,
      });
      setAnswer(response.answer);
    } catch (err: unknown) {
      if (err instanceof AuthRequiredError) {
        navigate("/auth");
        return;
      }
      setError(err instanceof Error ? err.message : "Failed to answer question");
    } finally {
      setAsking(false);
    }
  }

  async function confirmDeleteKnowledgeBase() {
    if (!kbPendingDelete) return;
    setDeletingKb(true);
    setError("");
    try {
      await deleteKnowledgeBase(kbPendingDelete.id);
      setKnowledgeBases((current) => {
        const next = current.filter((kb) => kb.id !== kbPendingDelete.id);
        if (selectedKnowledgeBaseId === kbPendingDelete.id) {
          setSelectedKnowledgeBaseId(next[0]?.id ?? null);
        }
        return next;
      });
      setKbPendingDelete(null);
    } catch (err: unknown) {
      if (err instanceof AuthRequiredError) {
        navigate("/auth");
        return;
      }
      setError(err instanceof Error ? err.message : "Failed to delete knowledge base");
    } finally {
      setDeletingKb(false);
    }
  }

  return (
    <div className="flex flex-1 min-h-0">
      <aside className="hidden lg:flex flex-col w-72 border-r border-border bg-surface-muted/40">
        <div className="flex items-center justify-between px-4 py-4 border-b border-border">
          <div className="flex items-center gap-2 text-sm font-medium">
            <BookOpen className="w-4 h-4" /> Knowledge bases
          </div>
          <button
            type="button"
            onClick={() => setShowCreateForm((visible) => !visible)}
            className="w-7 h-7 rounded-md bg-foreground text-background flex items-center justify-center hover:opacity-90"
            aria-label="Create knowledge base"
          >
            <Plus className="w-4 h-4" />
          </button>
        </div>
        <div className="flex-1 overflow-y-auto p-3 space-y-2">
          {showCreateForm && (
            <form onSubmit={handleCreateKnowledgeBase} className="rounded-lg border border-border bg-card p-3">
              <label className="text-[11px] font-semibold tracking-wide text-muted-foreground">NAME</label>
              <input
                autoFocus
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Product docs"
                className="mt-2 w-full rounded-md border border-border bg-background px-2 py-1.5 text-sm"
              />
              <label className="mt-3 block text-[11px] font-semibold tracking-wide text-muted-foreground">DESCRIPTION</label>
              <textarea
                rows={3}
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="What this knowledge base contains"
                className="mt-2 w-full resize-none rounded-md border border-border bg-background px-2 py-1.5 text-sm"
              />
              <div className="mt-3 flex gap-2">
                <Button type="submit" size="sm" disabled={creating || !name.trim()}>
                  {creating ? "Creating..." : "Create"}
                </Button>
                <Button type="button" size="sm" variant="outline" onClick={() => setShowCreateForm(false)} disabled={creating}>
                  Cancel
                </Button>
              </div>
            </form>
          )}
          {loading && <p className="px-2 py-3 text-xs text-muted-foreground">Loading knowledge bases...</p>}
          {!loading && knowledgeBases.length === 0 && (
            <div className="grid h-full place-items-center text-xs text-muted-foreground">
              No knowledge bases yet
            </div>
          )}
          {knowledgeBases.map((kb) => (
            <div
              key={kb.id}
              className={`flex gap-1 rounded-lg border p-1 transition ${
                selectedKnowledgeBaseId === kb.id ? "border-primary bg-primary/10" : "border-border bg-card"
              }`}
            >
              <button
                type="button"
                onClick={() => setSelectedKnowledgeBaseId(kb.id)}
                className="min-w-0 flex-1 rounded-md px-2 py-2 text-left hover:bg-primary/5"
              >
                <div className="text-sm font-medium truncate">{kb.name}</div>
                <div className="mt-1 line-clamp-2 text-xs text-muted-foreground">{kb.description || "No description"}</div>
              </button>
              <button
                type="button"
                onClick={() => setKbPendingDelete(kb)}
                className="shrink-0 self-center rounded-md p-2 text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
                aria-label={`Delete ${kb.name}`}
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>
          ))}
        </div>
      </aside>

      <div className="flex-1 overflow-auto px-4 py-5 sm:px-8 sm:py-6">
        <div className="mx-auto max-w-4xl">
          {error && (
            <div className="mb-4 rounded-lg border border-destructive/20 bg-destructive/10 px-4 py-3 text-sm text-destructive">
              {error}
            </div>
          )}

          <div className="lg:hidden mb-4 space-y-2">
            <label className="text-[11px] font-semibold text-muted-foreground tracking-wide">KNOWLEDGE BASE</label>
            <div className="flex gap-2">
              <select
                value={selectedKnowledgeBaseId ?? ""}
                onChange={(e) => setSelectedKnowledgeBaseId(e.target.value || null)}
                className="flex-1 rounded-lg border border-border bg-background px-3 py-2 text-sm"
                disabled={loading || knowledgeBases.length === 0}
              >
                {knowledgeBases.length === 0 ? <option value="">None yet</option> : null}
                {knowledgeBases.map((kb) => (
                  <option key={kb.id} value={kb.id}>
                    {kb.name}
                  </option>
                ))}
              </select>
              <Button type="button" size="icon" variant="outline" onClick={() => setShowCreateForm(true)} aria-label="Create knowledge base">
                <Plus className="w-4 h-4" />
              </Button>
            </div>
          </div>

          {showCreateForm && (
            <div className="lg:hidden mb-4">
              <form onSubmit={handleCreateKnowledgeBase} className="rounded-lg border border-border bg-card p-3">
                <label className="text-[11px] font-semibold tracking-wide text-muted-foreground">NAME</label>
                <input
                  autoFocus
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Product docs"
                  className="mt-2 w-full rounded-md border border-border bg-background px-2 py-1.5 text-sm"
                />
                <label className="mt-3 block text-[11px] font-semibold tracking-wide text-muted-foreground">DESCRIPTION</label>
                <textarea
                  rows={3}
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="What this knowledge base contains"
                  className="mt-2 w-full resize-none rounded-md border border-border bg-background px-2 py-1.5 text-sm"
                />
                <div className="mt-3 flex gap-2">
                  <Button type="submit" size="sm" disabled={creating || !name.trim()}>
                    {creating ? "Creating..." : "Create"}
                  </Button>
                  <Button type="button" size="sm" variant="outline" onClick={() => setShowCreateForm(false)} disabled={creating}>
                    Cancel
                  </Button>
                </div>
              </form>
            </div>
          )}

          {!selectedKnowledgeBase ? (
            <div className="grid min-h-[420px] place-items-center rounded-xl border border-border bg-card">
              <div className="text-center px-4">
                <div className="w-12 h-12 mx-auto rounded-xl bg-surface-muted border border-border flex items-center justify-center">
                  <BookOpen className="w-5 h-5 text-muted-foreground" />
                </div>
                <p className="mt-4 text-sm text-muted-foreground">You have no knowledge bases yet.</p>
                <Button
                  size="sm"
                  className="mt-4 gap-1.5 bg-gradient-primary text-primary-foreground shadow-elegant"
                  onClick={() => setShowCreateForm(true)}
                >
                  <Plus className="w-4 h-4" /> Create knowledge base
                </Button>
              </div>
            </div>
          ) : (
            <div className="rounded-xl border border-border bg-card p-5 sm:p-6 shadow-soft">
              <div className="mb-6 flex flex-wrap items-start justify-between gap-3">
                <div className="flex items-start gap-3 min-w-0">
                  <div className="rounded-xl bg-gradient-soft p-3 shrink-0">
                    <BookOpen className="h-5 w-5 text-primary" />
                  </div>
                  <div className="min-w-0">
                    <h1 className="text-xl font-semibold tracking-tight">{selectedKnowledgeBase.name}</h1>
                    <p className="mt-1 text-sm text-muted-foreground">
                      {selectedKnowledgeBase.description ||
                        "Attach this knowledge base to agents or flow nodes from the Flow Builder."}
                    </p>
                  </div>
                </div>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="gap-1.5 text-destructive hover:bg-destructive/10 hover:text-destructive shrink-0"
                  onClick={() => setKbPendingDelete(selectedKnowledgeBase)}
                >
                  <Trash2 className="w-4 h-4" /> Delete
                </Button>
              </div>
              <div className="rounded-lg border border-dashed border-border bg-surface-muted/30 p-4">
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-2 text-sm font-semibold">
                    <FileText className="h-4 w-4" /> Documents
                  </div>
                  <Button type="button" size="sm" variant="outline" onClick={() => setShowDocumentForm((visible) => !visible)}>
                    <Plus className="h-4 w-4" /> Add document
                  </Button>
                </div>
                {showDocumentForm && (
                  <form onSubmit={handleIngestDocument} className="mt-4 rounded-lg border border-border bg-card p-3">
                    <label className="text-[11px] font-semibold tracking-wide text-muted-foreground">TITLE</label>
                    <input
                      value={documentTitle}
                      onChange={(e) => setDocumentTitle(e.target.value)}
                      placeholder="Product FAQ"
                      className="mt-2 w-full rounded-md border border-border bg-background px-2 py-1.5 text-sm"
                    />
                    <label className="mt-3 block text-[11px] font-semibold tracking-wide text-muted-foreground">UPLOAD FILE</label>
                    <input
                      type="file"
                      accept=".pdf,.txt,.md,.csv,.json,application/pdf,text/*"
                      onChange={handleDocumentFile}
                      className="mt-2 w-full rounded-md border border-border bg-background px-2 py-1.5 text-sm"
                    />
                    <p className="mt-1 text-[11px] text-muted-foreground">
                      PDFs are processed automatically. For text files, you can review the content below before saving.
                    </p>
                    <label className="mt-3 block text-[11px] font-semibold tracking-wide text-muted-foreground">SOURCE URL</label>
                    <input
                      value={documentSourceUrl}
                      onChange={(e) => setDocumentSourceUrl(e.target.value)}
                      placeholder="Optional"
                      className="mt-2 w-full rounded-md border border-border bg-background px-2 py-1.5 text-sm"
                    />
                    <label className="mt-3 block text-[11px] font-semibold tracking-wide text-muted-foreground">CONTENT</label>
                    <textarea
                      rows={10}
                      value={documentContent}
                      onChange={(e) => setDocumentContent(e.target.value)}
                      placeholder={
                        documentFile?.type === "application/pdf" || documentFile?.name.toLowerCase().endsWith(".pdf")
                          ? "Text is extracted from the PDF when you save."
                          : "Paste document text here…"
                      }
                      className="mt-2 w-full resize-none rounded-md border border-border bg-background px-2 py-1.5 text-sm"
                    />
                    <div className="mt-3 flex gap-2">
                      <Button type="submit" size="sm" disabled={ingesting || !documentTitle.trim() || (!documentContent.trim() && !documentFile)}>
                        {ingesting ? "Saving…" : "Save document"}
                      </Button>
                      <Button type="button" size="sm" variant="outline" onClick={() => setShowDocumentForm(false)} disabled={ingesting}>
                        Cancel
                      </Button>
                    </div>
                  </form>
                )}
                {loadingDocuments ? (
                  <p className="mt-3 text-sm text-muted-foreground">Loading documents...</p>
                ) : documents.length === 0 ? (
                  <p className="mt-3 text-sm text-muted-foreground">
                    No documents yet. Use Add document to upload or paste your first one.
                  </p>
                ) : (
                  <div className="mt-3 divide-y divide-border rounded-lg border border-border bg-card">
                    {documents.map((document) => (
                      <div key={document.id} className="px-3 py-2 text-sm">
                        <div className="font-medium">{document.title}</div>
                        <div className="text-xs text-muted-foreground">{document.status}</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
              <form onSubmit={handleAskQuestion} className="mt-4 rounded-lg border border-border bg-surface-muted/30 p-4">
                <div className="mb-3 flex items-center justify-between gap-3">
                  <div className="text-sm font-semibold">Ask this knowledge base</div>
                  <Button type="submit" size="sm" disabled={asking || !question.trim() || documents.length === 0}>
                    {asking ? "Asking..." : "Ask"}
                  </Button>
                </div>
                <textarea
                  rows={3}
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  placeholder="Ask a question about the indexed documents..."
                  className="w-full resize-none rounded-md border border-border bg-background px-3 py-2 text-sm"
                />
                <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                  <div>
                    <label className="text-[11px] font-semibold tracking-wide text-muted-foreground">VECTOR MATCHES</label>
                    <input
                      type="number"
                      min={1}
                      max={12}
                      value={matchCount}
                      onChange={(e) => setMatchCount(Number(e.target.value))}
                      className="mt-1 w-full rounded-md border border-border bg-background px-2 py-1.5 text-sm"
                    />
                  </div>
                  <div>
                    <label className="text-[11px] font-semibold tracking-wide text-muted-foreground">WHOLE-KB CHUNKS</label>
                    <input
                      type="number"
                      min={1}
                      max={500}
                      value={wholeKbMaxChunks}
                      onChange={(e) => setWholeKbMaxChunks(Number(e.target.value))}
                      className="mt-1 w-full rounded-md border border-border bg-background px-2 py-1.5 text-sm"
                    />
                  </div>
                  <div>
                    <label className="text-[11px] font-semibold tracking-wide text-muted-foreground">WHOLE-KB CHARS</label>
                    <input
                      type="number"
                      min={1000}
                      max={300000}
                      step={1000}
                      value={wholeKbMaxContextChars}
                      onChange={(e) => setWholeKbMaxContextChars(Number(e.target.value))}
                      className="mt-1 w-full rounded-md border border-border bg-background px-2 py-1.5 text-sm"
                    />
                  </div>
                  <div>
                    <label className="text-[11px] font-semibold tracking-wide text-muted-foreground">FINAL MATCHES</label>
                    <input
                      type="number"
                      min={1}
                      max={40}
                      value={hybridMaxMatches}
                      onChange={(e) => setHybridMaxMatches(Number(e.target.value))}
                      className="mt-1 w-full rounded-md border border-border bg-background px-2 py-1.5 text-sm"
                    />
                  </div>
                </div>
                <p className="mt-2 text-[11px] text-muted-foreground">
                  Small knowledge bases can be read in full; larger ones use vector search and ranking to pick the best excerpts.
                </p>
                <div className="mt-4 grid gap-3 sm:grid-cols-2">
                  <div>
                    <label className="text-[11px] font-semibold tracking-wide text-muted-foreground">LLM PROVIDER</label>
                    <select
                      value={llmProvider}
                      onChange={(e) => setLlmProvider(e.target.value)}
                      className="mt-1 w-full rounded-md border border-border bg-background px-2 py-1.5 text-sm"
                    >
                      <option value="openai">OpenAI</option>
                    </select>
                  </div>
                  <div>
                    <label className="text-[11px] font-semibold tracking-wide text-muted-foreground">OPENAI API KEY</label>
                    <input
                      type="password"
                      value={openAiApiKey}
                      onChange={(e) => setOpenAiApiKey(e.target.value)}
                      placeholder="Optional — uses your workspace default when blank"
                      className="mt-1 w-full rounded-md border border-border bg-background px-2 py-1.5 text-sm"
                    />
                  </div>
                  <div>
                    <label className="text-[11px] font-semibold tracking-wide text-muted-foreground">ANSWER MODEL</label>
                    <input
                      value={answerModel}
                      onChange={(e) => setAnswerModel(e.target.value)}
                      placeholder="gpt-4.1-nano"
                      className="mt-1 w-full rounded-md border border-border bg-background px-2 py-1.5 text-sm"
                    />
                  </div>
                  <div>
                    <label className="text-[11px] font-semibold tracking-wide text-muted-foreground">EMBEDDING MODEL</label>
                    <input
                      value={embeddingModel}
                      onChange={(e) => setEmbeddingModel(e.target.value)}
                      placeholder="text-embedding-3-small"
                      className="mt-1 w-full rounded-md border border-border bg-background px-2 py-1.5 text-sm"
                    />
                  </div>
                </div>
                <p className="mt-2 text-[11px] text-muted-foreground">
                  Use an embedding model with 1536 dimensions that matches how documents were indexed. Keys are sent only with this
                  request and are not stored in the app.
                </p>
                {documents.length === 0 && (
                  <p className="mt-2 text-xs text-muted-foreground">Add at least one document before asking questions.</p>
                )}
                {answer && (
                  <div className="mt-3 rounded-lg border border-border bg-card p-3 text-sm">
                    {answer}
                  </div>
                )}
              </form>
            </div>
          )}
        </div>
      </div>

      <AlertDialog open={!!kbPendingDelete} onOpenChange={(open) => !open && setKbPendingDelete(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete this knowledge base?</AlertDialogTitle>
            <AlertDialogDescription>
              {kbPendingDelete
                ? `“${kbPendingDelete.name}” and all of its documents and search index data will be permanently removed. Links from agents and flows will stop working.`
                : ""}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={deletingKb}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              onClick={(e) => {
                e.preventDefault();
                void confirmDeleteKnowledgeBase();
              }}
              disabled={deletingKb}
            >
              {deletingKb ? "Deleting…" : "Delete"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
