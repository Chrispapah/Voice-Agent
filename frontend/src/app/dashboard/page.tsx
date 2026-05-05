"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { useAuth } from "@/lib/auth";
import { BotConfig, createBot, deleteBot, listBots } from "@/lib/api";
import { botExecutionLabel } from "@/lib/conversationSpec";
import { Plus, Settings, Trash2, Bot, Phone } from "lucide-react";

export default function DashboardPage() {
  const { user, loading: authLoading, logout } = useAuth();
  const [bots, setBots] = useState<BotConfig[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const router = useRouter();

  useEffect(() => {
    if (authLoading || !user) return;
    listBots()
      .then(setBots)
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : "Failed to load bots");
      })
      .finally(() => setLoading(false));
  }, [user, authLoading]);

  async function handleCreate() {
    const name = prompt("Conversation name:", "My conversation");
    if (!name) return;
    setError("");
    try {
      const bot = await createBot(name);
      router.push(`/bots/${bot.id}/settings`);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to create bot");
    }
  }

  async function handleDelete(id: string) {
    if (!confirm("Delete this bot? This cannot be undone.")) return;
    setError("");
    try {
      await deleteBot(id);
      setBots((prev) => prev.filter((b) => b.id !== id));
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to delete bot");
    }
  }

  if (authLoading || loading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <p className="text-[var(--muted-foreground)]">Loading...</p>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Your conversations</h1>
          <p className="text-sm text-[var(--muted-foreground)]">
            Welcome back, {user?.display_name || user?.email}
          </p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={handleCreate}
            className="flex items-center gap-2 rounded-md bg-[var(--primary)] px-4 py-2 text-sm font-medium text-[var(--primary-foreground)] hover:opacity-90"
          >
            <Plus size={16} />
            New conversation
          </button>
          <button
            onClick={logout}
            className="rounded-md border border-[var(--border)] px-4 py-2 text-sm hover:bg-[var(--secondary)]"
          >
            Sign out
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-6 rounded-md bg-red-50 p-3 text-sm text-red-700 dark:bg-red-900/30 dark:text-red-300">
          {error}
        </div>
      )}

      {bots.length === 0 ? (
        <div className="rounded-lg border border-dashed border-[var(--border)] p-12 text-center">
          <Bot size={48} className="mx-auto mb-4 text-[var(--muted-foreground)]" />
          <h2 className="text-lg font-semibold">No conversations yet</h2>
          <p className="mb-4 text-sm text-[var(--muted-foreground)]">
            Create a voice conversation and configure it in the agent builder.
          </p>
          <button
            onClick={handleCreate}
            className="rounded-md bg-[var(--primary)] px-4 py-2 text-sm font-medium text-[var(--primary-foreground)] hover:opacity-90"
          >
            Create conversation
          </button>
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {bots.map((bot) => (
            <div
              key={bot.id}
              className="rounded-lg border border-[var(--border)] bg-[var(--card)] p-5 transition-shadow hover:shadow-md"
            >
              <div className="mb-3 flex items-start justify-between">
                <div>
                  <h3 className="font-semibold">{bot.name}</h3>
                  <span
                    className={`mt-1 inline-block rounded-full px-2 py-0.5 text-xs font-medium ${
                      bot.is_active
                        ? "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300"
                        : "bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400"
                    }`}
                  >
                    {bot.is_active ? "Active" : "Inactive"}
                  </span>
                </div>
                <button
                  onClick={() => handleDelete(bot.id)}
                  className="rounded p-1 text-[var(--muted-foreground)] hover:bg-[var(--secondary)] hover:text-[var(--destructive)]"
                >
                  <Trash2 size={16} />
                </button>
              </div>
              <div className="mb-4 space-y-1 text-xs text-[var(--muted-foreground)]">
                <p>{botExecutionLabel(bot.conversation_spec)}</p>
                <p>LLM: {bot.llm_provider} / {bot.llm_model_name}</p>
                <p>Voice: {bot.elevenlabs_voice_id || "Not set"}</p>
                <p>Turns: {bot.max_call_turns} max</p>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <Link
                  href={`/bots/${bot.id}/settings`}
                  className="flex items-center justify-center gap-1.5 rounded-md border border-[var(--border)] px-3 py-1.5 text-xs font-medium hover:bg-[var(--secondary)]"
                >
                  <Settings size={14} />
                  Configure
                </Link>
                <Link
                  href={`/bots/${bot.id}/test`}
                  className="flex items-center justify-center gap-1.5 rounded-md bg-[var(--primary)] px-3 py-1.5 text-xs font-medium text-[var(--primary-foreground)] hover:opacity-90"
                >
                  Test
                </Link>
                <Link
                  href={`/bots/${bot.id}/calls`}
                  className="col-span-2 flex items-center justify-center gap-1.5 rounded-md border border-[var(--border)] px-3 py-1.5 text-xs font-medium hover:bg-[var(--secondary)]"
                >
                  <Phone size={14} />
                  Call logs
                </Link>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
