"use client";

import { useEffect, useMemo, useState, use } from "react";
import Link from "next/link";
import { useAuth } from "@/lib/auth";
import { AgentBuilder } from "@/components/AgentBuilder";
import { BotConfig, getBot, updateBot } from "@/lib/api";
import { isClassicSdrBuiltInGraph } from "@/lib/conversationSpec";
import { ArrowLeft, Save, MessageSquare, FileText, Phone } from "lucide-react";

const TABS = [
  { id: "general", label: "General" },
  { id: "agent_builder", label: "Agent builder" },
  { id: "ai", label: "AI Model" },
  { id: "voice", label: "Voice & STT" },
  { id: "conversation", label: "SDR funnel" },
  { id: "prompts", label: "SDR stage prompts" },
  { id: "keys", label: "API Keys" },
  { id: "telephony", label: "Telephony" },
] as const;

type TabId = (typeof TABS)[number]["id"];

export default function BotSettingsPage({ params }: { params: Promise<{ id: string }> }) {
  const { id: botId } = use(params);
  const { user, loading: authLoading } = useAuth();
  const [bot, setBot] = useState<BotConfig | null>(null);
  const [tab, setTab] = useState<TabId>("general");
  const [saving, setSaving] = useState(false);
  const [dirty, setDirty] = useState<Partial<BotConfig>>({});
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (authLoading || !user) return;
    getBot(botId)
      .then(setBot)
      .catch((err: unknown) => {
        setMessage(err instanceof Error ? err.message : "Failed to load bot");
      })
      .finally(() => setLoading(false));
  }, [botId, user, authLoading]);

  function update<K extends keyof BotConfig>(key: K, value: BotConfig[K]) {
    setDirty((prev) => ({ ...prev, [key]: value }));
  }

  function current<K extends keyof BotConfig>(key: K): BotConfig[K] {
    if (key in dirty) return dirty[key] as BotConfig[K];
    return bot?.[key] as BotConfig[K];
  }

  const conversationSpec = useMemo(
    () =>
      ("conversation_spec" in dirty ? dirty.conversation_spec : bot?.conversation_spec) as
        | BotConfig["conversation_spec"]
        | undefined,
    [dirty, bot?.conversation_spec],
  );
  const showClassicSdrTabs = isClassicSdrBuiltInGraph(conversationSpec ?? null);
  const visibleTabs = useMemo(
    () =>
      showClassicSdrTabs
        ? [...TABS]
        : TABS.filter((t) => t.id !== "conversation" && t.id !== "prompts"),
    [showClassicSdrTabs],
  );

  useEffect(() => {
    if (!showClassicSdrTabs && (tab === "conversation" || tab === "prompts")) {
      setTab("agent_builder");
    }
  }, [showClassicSdrTabs, tab]);

  async function handleSave() {
    if (!bot || Object.keys(dirty).length === 0) return;
    setSaving(true);
    setMessage("");
    try {
      const updated = await updateBot(bot.id, dirty);
      setBot(updated);
      setDirty({});
      setMessage("Saved!");
      setTimeout(() => setMessage(""), 2000);
    } catch (err: unknown) {
      setMessage(err instanceof Error ? err.message : "Save failed");
    } finally {
      setSaving(false);
    }
  }

  if (authLoading || loading) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <p className="text-[var(--muted-foreground)]">Loading...</p>
      </div>
    );
  }

  if (!bot) {
    return (
      <div className="flex min-h-screen items-center justify-center px-4">
        <div className="rounded-md bg-red-50 p-4 text-sm text-red-700 dark:bg-red-900/30 dark:text-red-300">
          {message || "Bot not found"}
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-4xl px-4 py-8">
      {/* Header */}
      <div className="mb-6 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Link href="/dashboard" className="rounded p-1 hover:bg-[var(--secondary)]">
            <ArrowLeft size={20} />
          </Link>
          <div>
            <h1 className="text-xl font-bold">{bot.name}</h1>
            <div className="flex gap-3 text-xs text-[var(--muted-foreground)]">
              <Link href={`/bots/${bot.id}/test`} className="flex items-center gap-1 hover:underline">
                <MessageSquare size={12} /> Test
              </Link>
              <Link href={`/bots/${bot.id}/leads`} className="flex items-center gap-1 hover:underline">
                <FileText size={12} /> Leads
              </Link>
              <Link href={`/bots/${bot.id}/calls`} className="flex items-center gap-1 hover:underline">
                <Phone size={12} /> Calls
              </Link>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {message && (
            <span
              className={`text-sm ${
                message === "Saved!" ? "text-green-600" : "text-red-600 dark:text-red-300"
              }`}
            >
              {message}
            </span>
          )}
          <button
            type="button"
            onClick={handleSave}
            disabled={saving || Object.keys(dirty).length === 0}
            className="flex items-center gap-2 rounded-md bg-[var(--primary)] px-4 py-2 text-sm font-medium text-[var(--primary-foreground)] hover:opacity-90 disabled:opacity-50"
          >
            <Save size={16} />
            {saving ? "Saving..." : "Save"}
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="mb-6 flex gap-1 overflow-x-auto rounded-lg border border-[var(--border)] bg-[var(--secondary)] p-1">
        {visibleTabs.map((t) => (
          <button
            type="button"
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`whitespace-nowrap rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
              tab === t.id
                ? "bg-[var(--background)] shadow-sm"
                : "text-[var(--muted-foreground)] hover:text-[var(--foreground)]"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="space-y-5 rounded-lg border border-[var(--border)] bg-[var(--card)] p-6">
        {tab === "agent_builder" && (
          <AgentBuilder
            botId={bot.id}
            value={
              ("conversation_spec" in dirty
                ? dirty.conversation_spec
                : bot.conversation_spec) as BotConfig["conversation_spec"]
            }
            onChange={(next) => update("conversation_spec", next)}
          />
        )}

        {tab === "general" && (
          <>
            <Field label="Bot Name">
              <input
                value={current("name") || ""}
                onChange={(e) => update("name", e.target.value)}
                className="input-field"
              />
            </Field>
            <Field label="Sales Rep Name">
              <input
                value={current("sales_rep_name") || ""}
                onChange={(e) => update("sales_rep_name", e.target.value)}
                className="input-field"
              />
            </Field>
            <Field label="Initial Greeting">
              <textarea
                rows={3}
                value={current("initial_greeting") || ""}
                onChange={(e) => update("initial_greeting", e.target.value)}
                className="input-field"
              />
            </Field>
            <Field label="Active">
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={current("is_active") ?? true}
                  onChange={(e) => update("is_active", e.target.checked)}
                  className="h-4 w-4 rounded"
                />
                <span className="text-sm">Bot is active and can receive calls</span>
              </label>
            </Field>
          </>
        )}

        {tab === "ai" && (
          <>
            <Field label="LLM Provider">
              <select
                value={current("llm_provider") || "openai"}
                onChange={(e) => update("llm_provider", e.target.value)}
                className="input-field"
              >
                <option value="openai">OpenAI</option>
                <option value="anthropic">Anthropic</option>
                <option value="groq">Groq</option>
              </select>
            </Field>
            <Field label="Model Name">
              <input
                value={current("llm_model_name") || ""}
                onChange={(e) => update("llm_model_name", e.target.value)}
                className="input-field"
                placeholder="e.g. gpt-4o-mini"
              />
            </Field>
            <Field label={`Temperature: ${current("llm_temperature") ?? 0.4}`}>
              <input
                type="range"
                min="0"
                max="2"
                step="0.1"
                value={current("llm_temperature") ?? 0.4}
                onChange={(e) => update("llm_temperature", parseFloat(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-[var(--muted-foreground)]">
                <span>Precise (0)</span>
                <span>Creative (2)</span>
              </div>
            </Field>
            <Field label={`Max Tokens: ${current("llm_max_tokens") ?? 300}`}>
              <input
                type="range"
                min="128"
                max="4096"
                step="64"
                value={current("llm_max_tokens") ?? 300}
                onChange={(e) => update("llm_max_tokens", parseInt(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-[var(--muted-foreground)]">
                <span>128</span>
                <span>4096</span>
              </div>
            </Field>
          </>
        )}

        {tab === "voice" && (
          <>
            <Field label="ElevenLabs Voice ID">
              <input
                value={current("elevenlabs_voice_id") || ""}
                onChange={(e) => update("elevenlabs_voice_id", e.target.value)}
                className="input-field"
                placeholder="e.g. 21m00Tcm4TlvDq8ikWAM"
              />
            </Field>
            <Field label="ElevenLabs Model">
              <select
                value={current("elevenlabs_model_id") || "eleven_turbo_v2"}
                onChange={(e) => update("elevenlabs_model_id", e.target.value)}
                className="input-field"
              >
                <option value="eleven_turbo_v2">Turbo v2 (fastest)</option>
                <option value="eleven_multilingual_v2">Multilingual v2</option>
                <option value="eleven_monolingual_v1">Monolingual v1</option>
              </select>
            </Field>
            <Field label="Deepgram Model">
              <select
                value={current("deepgram_model") || "nova-2"}
                onChange={(e) => update("deepgram_model", e.target.value)}
                className="input-field"
              >
                <option value="nova-2">Nova 2</option>
                <option value="nova-3">Nova 3</option>
                <option value="phonecall">Phonecall</option>
              </select>
            </Field>
            <Field label="Language">
              <select
                value={current("deepgram_language") || "en-US"}
                onChange={(e) => update("deepgram_language", e.target.value)}
                className="input-field"
              >
                <option value="en-US">English (US)</option>
                <option value="en-GB">English (UK)</option>
                <option value="es">Spanish</option>
                <option value="fr">French</option>
                <option value="de">German</option>
              </select>
            </Field>
          </>
        )}

        {tab === "conversation" && (
          <>
            <Field label={`Max Call Turns: ${current("max_call_turns") ?? 12}`}>
              <input
                type="range"
                min="5"
                max="30"
                value={current("max_call_turns") ?? 12}
                onChange={(e) => update("max_call_turns", parseInt(e.target.value))}
                className="w-full"
              />
            </Field>
            <Field label={`Max Objection Attempts: ${current("max_objection_attempts") ?? 2}`}>
              <input
                type="range"
                min="1"
                max="5"
                value={current("max_objection_attempts") ?? 2}
                onChange={(e) => update("max_objection_attempts", parseInt(e.target.value))}
                className="w-full"
              />
            </Field>
            <Field label={`Max Qualify Attempts: ${current("max_qualify_attempts") ?? 3}`}>
              <input
                type="range"
                min="1"
                max="5"
                value={current("max_qualify_attempts") ?? 3}
                onChange={(e) => update("max_qualify_attempts", parseInt(e.target.value))}
                className="w-full"
              />
            </Field>
            <Field label={`Max Booking Attempts: ${current("max_booking_attempts") ?? 3}`}>
              <input
                type="range"
                min="1"
                max="5"
                value={current("max_booking_attempts") ?? 3}
                onChange={(e) => update("max_booking_attempts", parseInt(e.target.value))}
                className="w-full"
              />
            </Field>
          </>
        )}

        {tab === "prompts" && (
          <>
            <p className="text-sm text-[var(--muted-foreground)]">
              Customize the system prompts for each conversation stage. Leave blank to use defaults.
              Available variables: <code className="rounded bg-[var(--secondary)] px-1">{"{lead_name}"}</code>,{" "}
              <code className="rounded bg-[var(--secondary)] px-1">{"{company}"}</code>,{" "}
              <code className="rounded bg-[var(--secondary)] px-1">{"{lead_context}"}</code>,{" "}
              <code className="rounded bg-[var(--secondary)] px-1">{"{pain_points}"}</code>,{" "}
              <code className="rounded bg-[var(--secondary)] px-1">{"{sales_rep_name}"}</code>
            </p>
            {(
              [
                ["prompt_greeting", "Greeting"],
                ["prompt_qualify", "Qualification"],
                ["prompt_pitch", "Pitch"],
                ["prompt_objection", "Objection Handling"],
                ["prompt_booking", "Meeting Booking"],
                ["prompt_wrapup", "Wrap-up"],
              ] as const
            ).map(([key, label]) => (
              <Field key={key} label={label}>
                <textarea
                  rows={4}
                  value={(current(key) as string) || ""}
                  onChange={(e) => update(key, e.target.value || null)}
                  className="input-field font-mono text-xs"
                  placeholder="Leave blank to use default prompt"
                />
              </Field>
            ))}
          </>
        )}

        {tab === "keys" && (
          <>
            <p className="text-sm text-[var(--muted-foreground)]">
              Enter your API keys. These are stored encrypted and never exposed in full.
            </p>
            <Field label="OpenAI API Key">
              <input
                type="password"
                placeholder={bot.openai_api_key || "sk-..."}
                onChange={(e) => update("openai_api_key", e.target.value)}
                className="input-field"
              />
            </Field>
            <Field label="Anthropic API Key">
              <input
                type="password"
                placeholder={bot.anthropic_api_key || "sk-ant-..."}
                onChange={(e) => update("anthropic_api_key", e.target.value)}
                className="input-field"
              />
            </Field>
            <Field label="Groq API Key">
              <input
                type="password"
                placeholder={bot.groq_api_key || "gsk_..."}
                onChange={(e) => update("groq_api_key", e.target.value)}
                className="input-field"
              />
            </Field>
            <Field label="ElevenLabs API Key">
              <input
                type="password"
                placeholder={bot.elevenlabs_api_key || "Enter key..."}
                onChange={(e) => update("elevenlabs_api_key", e.target.value)}
                className="input-field"
              />
            </Field>
            <Field label="Deepgram API Key">
              <input
                type="password"
                placeholder={bot.deepgram_api_key || "Enter key..."}
                onChange={(e) => update("deepgram_api_key", e.target.value)}
                className="input-field"
              />
            </Field>
          </>
        )}

        {tab === "telephony" && (
          <>
            <Field label="Twilio Account SID">
              <input
                type="password"
                placeholder={bot.twilio_account_sid || "AC..."}
                onChange={(e) => update("twilio_account_sid", e.target.value)}
                className="input-field"
              />
            </Field>
            <Field label="Twilio Auth Token">
              <input
                type="password"
                placeholder={bot.twilio_auth_token || "Enter token..."}
                onChange={(e) => update("twilio_auth_token", e.target.value)}
                className="input-field"
              />
            </Field>
            <Field label="Twilio Phone Number">
              <input
                value={current("twilio_phone_number") || ""}
                onChange={(e) => update("twilio_phone_number", e.target.value)}
                className="input-field"
                placeholder="+1234567890"
              />
            </Field>
          </>
        )}
      </div>

      <style jsx>{`
        :global(.input-field) {
          width: 100%;
          border-radius: 0.375rem;
          border: 1px solid var(--border);
          background: var(--background);
          padding: 0.5rem 0.75rem;
          font-size: 0.875rem;
          outline: none;
        }
        :global(.input-field:focus) {
          box-shadow: 0 0 0 2px var(--ring);
        }
      `}</style>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="mb-1.5 block text-sm font-medium">{label}</label>
      {children}
    </div>
  );
}
