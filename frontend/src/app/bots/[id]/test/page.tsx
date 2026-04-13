"use client";

import { FormEvent, useEffect, useRef, useState, use } from "react";
import Link from "next/link";
import { useAuth } from "@/lib/auth";
import { getBot, listLeads, startTestSession, sendTestTurn, Lead } from "@/lib/api";
import { ArrowLeft, Send } from "lucide-react";

interface ChatMessage {
  role: "human" | "agent";
  content: string;
}

export default function TestConsolePage({ params }: { params: Promise<{ id: string }> }) {
  const { id: botId } = use(params);
  const { user, loading: authLoading } = useAuth();
  const [botName, setBotName] = useState("");
  const [leads, setLeads] = useState<Lead[]>([]);
  const [selectedLead, setSelectedLead] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [stage, setStage] = useState("");
  const [outcome, setOutcome] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const messagesEnd = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (authLoading || !user) return;
    Promise.all([getBot(botId), listLeads(botId)])
      .then(([b, l]) => {
        setBotName(b.name);
        setLeads(l);
        if (l.length > 0) setSelectedLead(l[0].id);
      })
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : "Failed to load test console");
      })
      .finally(() => setLoading(false));
  }, [botId, user, authLoading]);

  useEffect(() => {
    messagesEnd.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleStart() {
    if (!selectedLead) return;
    setSending(true);
    setError("");
    try {
      const res = await startTestSession(botId, selectedLead);
      setSessionId(res.conversation_id);
      setStage(res.stage);
      setMessages([{ role: "agent", content: res.agent_response }]);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to start test session");
    } finally {
      setSending(false);
    }
  }

  async function handleSend(e: FormEvent) {
    e.preventDefault();
    if (!sessionId || !input.trim()) return;
    const text = input.trim();
    setInput("");
    setMessages((prev) => [...prev, { role: "human", content: text }]);
    setSending(true);
    setError("");
    try {
      const res = await sendTestTurn(botId, sessionId, text);
      setMessages((prev) => [...prev, { role: "agent", content: res.agent_response }]);
      setStage(res.stage);
      setOutcome(res.call_outcome);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to get agent response");
      setMessages((prev) => [
        ...prev,
        { role: "agent", content: "[Error: could not get response]" },
      ]);
    } finally {
      setSending(false);
    }
  }

  return (
    <div className="mx-auto flex h-screen max-w-3xl flex-col px-4 py-6">
      {/* Header */}
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Link href={`/bots/${botId}/settings`} className="rounded p-1 hover:bg-[var(--secondary)]">
            <ArrowLeft size={20} />
          </Link>
          <div>
            <h1 className="text-lg font-bold">Test: {botName}</h1>
            <div className="flex gap-3 text-xs text-[var(--muted-foreground)]">
              {stage && <span>Stage: {stage}</span>}
              {outcome && <span>Outcome: {outcome}</span>}
            </div>
          </div>
        </div>
      </div>

      {error && (
        <div className="mb-4 rounded-md bg-red-50 p-3 text-sm text-red-700 dark:bg-red-900/30 dark:text-red-300">
          {error}
        </div>
      )}

      {/* Chat area or start form */}
      {authLoading || loading ? (
        <div className="flex flex-1 items-center justify-center text-[var(--muted-foreground)]">Loading...</div>
      ) : !sessionId ? (
        <div className="flex flex-1 flex-col items-center justify-center">
          <h2 className="mb-4 text-lg font-semibold">Start a test conversation</h2>
          {leads.length === 0 ? (
            <div className="text-center">
              <p className="mb-2 text-sm text-[var(--muted-foreground)]">
                No leads found. Add a lead first.
              </p>
              <Link
                href={`/bots/${botId}/leads`}
                className="text-sm font-medium text-[var(--primary)] hover:underline"
              >
                Add leads
              </Link>
            </div>
          ) : (
            <div className="w-full max-w-sm space-y-4">
              <div>
                <label className="mb-1 block text-sm font-medium">Select Lead</label>
                <select
                  value={selectedLead}
                  onChange={(e) => setSelectedLead(e.target.value)}
                  className="w-full rounded-md border border-[var(--border)] bg-[var(--background)] px-3 py-2 text-sm"
                >
                  {leads.map((l) => (
                    <option key={l.id} value={l.id}>
                      {l.lead_name} - {l.company}
                    </option>
                  ))}
                </select>
              </div>
              <button
                onClick={handleStart}
                disabled={sending}
                className="w-full rounded-md bg-[var(--primary)] px-4 py-2 text-sm font-medium text-[var(--primary-foreground)] hover:opacity-90 disabled:opacity-50"
              >
                {sending ? "Starting..." : "Start Conversation"}
              </button>
            </div>
          )}
        </div>
      ) : (
        <>
          {/* Messages */}
          <div className="flex-1 space-y-3 overflow-y-auto rounded-lg border border-[var(--border)] bg-[var(--secondary)] p-4">
            {messages.map((msg, i) => (
              <div key={i} className={`flex ${msg.role === "human" ? "justify-end" : "justify-start"}`}>
                <div
                  className={`max-w-[80%] rounded-lg px-4 py-2 text-sm ${
                    msg.role === "human"
                      ? "bg-[var(--primary)] text-[var(--primary-foreground)]"
                      : "bg-[var(--card)] border border-[var(--border)]"
                  }`}
                >
                  {msg.content}
                </div>
              </div>
            ))}
            <div ref={messagesEnd} />
          </div>

          {/* Input */}
          <form onSubmit={handleSend} className="mt-3 flex gap-2">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type your response..."
              disabled={sending}
              className="flex-1 rounded-md border border-[var(--border)] bg-[var(--background)] px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-[var(--ring)]"
            />
            <button
              type="submit"
              disabled={sending || !input.trim()}
              className="flex items-center gap-1.5 rounded-md bg-[var(--primary)] px-4 py-2 text-sm font-medium text-[var(--primary-foreground)] hover:opacity-90 disabled:opacity-50"
            >
              <Send size={16} />
            </button>
          </form>
        </>
      )}
    </div>
  );
}
