"use client";

import { useEffect, useState, use } from "react";
import Link from "next/link";
import { useAuth } from "@/lib/auth";
import { CallLog, getBot, listCalls } from "@/lib/api";
import { ArrowLeft, ChevronDown, ChevronUp, Phone } from "lucide-react";

export default function CallsPage({ params }: { params: Promise<{ id: string }> }) {
  const { id: botId } = use(params);
  const { user, loading: authLoading } = useAuth();
  const [botName, setBotName] = useState("");
  const [calls, setCalls] = useState<CallLog[]>([]);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    if (authLoading || !user) return;
    Promise.all([getBot(botId), listCalls(botId)])
      .then(([b, loadedCalls]) => {
        setBotName(b.name);
        setCalls(loadedCalls);
      })
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : "Failed to load calls");
      })
      .finally(() => setLoading(false));
  }, [botId, user, authLoading]);

  const outcomeColor: Record<string, string> = {
    meeting_booked: "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300",
    follow_up_needed: "bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-300",
    not_interested: "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300",
    no_answer: "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400",
    voicemail: "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400",
  };

  return (
    <div className="mx-auto max-w-4xl px-4 py-8">
      <div className="mb-6 flex items-center gap-3">
        <Link href={`/bots/${botId}/settings`} className="rounded p-1 hover:bg-[var(--secondary)]">
          <ArrowLeft size={20} />
        </Link>
        <h1 className="text-xl font-bold">Call Logs - {botName}</h1>
      </div>

      {error && (
        <div className="mb-6 rounded-md bg-red-50 p-3 text-sm text-red-700 dark:bg-red-900/30 dark:text-red-300">
          {error}
        </div>
      )}

      {authLoading || loading ? (
        <div className="rounded-lg border border-[var(--border)] p-8 text-center text-[var(--muted-foreground)]">
          Loading...
        </div>
      ) : null}

      {!loading && calls.length === 0 ? (
        <div className="rounded-lg border border-dashed border-[var(--border)] p-12 text-center">
          <Phone size={48} className="mx-auto mb-4 text-[var(--muted-foreground)]" />
          <p className="text-[var(--muted-foreground)]">No calls yet.</p>
        </div>
      ) : !loading ? (
        <div className="space-y-2">
          {calls.map((call) => (
            <div key={call.id} className="rounded-lg border border-[var(--border)] bg-[var(--card)]">
              <button
                onClick={() => setExpanded(expanded === call.id ? null : call.id)}
                className="flex w-full items-center justify-between px-4 py-3 text-left"
              >
                <div className="flex items-center gap-4">
                  <span
                    className={`rounded-full px-2 py-0.5 text-xs font-medium ${
                      outcomeColor[call.call_outcome] || outcomeColor.no_answer
                    }`}
                  >
                    {call.call_outcome.replace(/_/g, " ")}
                  </span>
                  <span className="text-sm font-medium">{call.conversation_id}</span>
                  <span className="text-xs text-[var(--muted-foreground)]">
                    {call.started_at ? new Date(call.started_at).toLocaleString() : ""}
                  </span>
                </div>
                {expanded === call.id ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
              </button>
              {expanded === call.id && (
                <div className="border-t border-[var(--border)] px-4 py-4">
                  <div className="mb-3 flex gap-4 text-xs text-[var(--muted-foreground)]">
                    <span>Meeting: {call.meeting_booked ? "Yes" : "No"}</span>
                    {call.proposed_slot && <span>Slot: {call.proposed_slot}</span>}
                    {call.follow_up_action && <span>Action: {call.follow_up_action}</span>}
                  </div>
                  <div className="space-y-2">
                    {call.transcript.map((msg, i) => (
                      <div key={i} className={`flex ${msg.role === "human" ? "justify-end" : "justify-start"}`}>
                        <div
                          className={`max-w-[80%] rounded-lg px-3 py-2 text-sm ${
                            msg.role === "human"
                              ? "bg-[var(--primary)] text-[var(--primary-foreground)]"
                              : "bg-[var(--secondary)]"
                          }`}
                        >
                          <span className="mb-0.5 block text-xs font-medium opacity-70">
                            {msg.role === "human" ? "Prospect" : "Agent"}
                          </span>
                          {msg.content}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}
