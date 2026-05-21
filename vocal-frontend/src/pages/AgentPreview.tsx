import { useEffect, useMemo, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import { Mic, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  getPublicAgentPreview,
  getPublicPreviewVoiceWebSocketUrl,
  startPublicAgentPreviewSession,
  type PublicAgentPreview,
} from "@/lib/api";
import { VoiceSession, type VoiceSessionCallbacks } from "@/lib/voiceSession";

type ChatMessage = { role: "human" | "agent"; content: string };

function upsertAgentBubble(messages: ChatMessage[], text: string): ChatMessage[] {
  const last = messages[messages.length - 1];
  if (last?.role === "agent") {
    return [...messages.slice(0, -1), { role: "agent", content: text }];
  }
  return [...messages, { role: "agent", content: text }];
}

export default function AgentPreviewPage() {
  const { token } = useParams();
  const [preview, setPreview] = useState<PublicAgentPreview | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(true);
  const [starting, setStarting] = useState(false);
  const [active, setActive] = useState(false);
  const [error, setError] = useState("");
  const [conversationId, setConversationId] = useState<string | null>(null);
  const voiceRef = useRef<VoiceSession | null>(null);

  useEffect(() => {
    if (!token) {
      setError("Preview link is missing a token.");
      setLoading(false);
      return;
    }
    getPublicAgentPreview(token)
      .then(setPreview)
      .catch((err: unknown) => setError(err instanceof Error ? err.message : "Preview unavailable"))
      .finally(() => setLoading(false));
  }, [token]);

  useEffect(() => {
    return () => {
      void voiceRef.current?.stop();
    };
  }, []);

  const canStart = useMemo(() => Boolean(token && preview && !active && !starting), [active, preview, starting, token]);

  async function startConversation() {
    if (!token || !preview || active) return;
    setError("");
    setStarting(true);
    try {
      const session = await startPublicAgentPreviewSession(token);
      const callbacks: VoiceSessionCallbacks = {
        onReady: (cid) => setConversationId(cid),
        onTranscriptFinal: (text) => setMessages((current) => [...current, { role: "human", content: text }]),
        onAgentText: (text) => setMessages((current) => upsertAgentBubble(current, text)),
        onError: setError,
        onClose: () => {
          voiceRef.current = null;
          setActive(false);
        },
      };
      const voice = new VoiceSession(callbacks);
      voiceRef.current = voice;
      await voice.startWithOptions({
        wsUrl: getPublicPreviewVoiceWebSocketUrl(token),
        authMessage: null,
        leadId: session.lead_id,
        conversationId: session.conversation_id,
      });
      setActive(true);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Could not start the voice preview.");
      voiceRef.current = null;
    } finally {
      setStarting(false);
    }
  }

  async function stopConversation() {
    await voiceRef.current?.stop();
    voiceRef.current = null;
    setActive(false);
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      <main className="mx-auto flex min-h-screen max-w-4xl flex-col px-6 py-10">
        <div className="mb-8 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="grid h-10 w-10 place-items-center rounded-xl bg-primary/10">
              <img src="/akoi-logo-no-back.png" alt="Akoi" className="h-7 w-7 object-contain" />
            </div>
            <div>
              <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Live AI Preview</div>
              <h1 className="text-2xl font-semibold">{preview?.title || "Agent preview"}</h1>
            </div>
          </div>
          {conversationId && <div className="font-mono text-xs text-muted-foreground">{conversationId}</div>}
        </div>

        {loading && (
          <div className="grid flex-1 place-items-center rounded-2xl border border-border bg-card text-sm text-muted-foreground">
            Loading preview...
          </div>
        )}

        {!loading && error && !preview && (
          <div className="grid flex-1 place-items-center rounded-2xl border border-border bg-card px-6 text-center">
            <div>
              <div className="text-lg font-semibold">Preview unavailable</div>
              <p className="mt-2 text-sm text-muted-foreground">{error}</p>
            </div>
          </div>
        )}

        {!loading && preview && (
          <div className="flex flex-1 flex-col rounded-2xl border border-border bg-card shadow-sm">
            <div className="border-b border-border p-5">
              <h2 className="text-lg font-semibold">{preview.agent_name}</h2>
              <p className="mt-1 text-sm text-muted-foreground">{preview.welcome_message}</p>
              {error && (
                <div className="mt-3 rounded-lg border border-destructive/20 bg-destructive/10 px-3 py-2 text-sm text-destructive">
                  {error}
                </div>
              )}
            </div>

            <div className="min-h-[380px] flex-1 space-y-3 overflow-y-auto bg-surface-muted/30 p-5">
              {messages.length === 0 ? (
                <div className="grid h-full min-h-[320px] place-items-center text-center text-sm text-muted-foreground">
                  <div>
                    <p>Click start and allow microphone access.</p>
                    <p className="mt-1">Then speak naturally with the agent.</p>
                  </div>
                </div>
              ) : (
                messages.map((message, index) => (
                  <div key={index} className={`flex ${message.role === "human" ? "justify-end" : "justify-start"}`}>
                    <div
                      className={`max-w-[82%] rounded-xl px-4 py-3 text-sm leading-6 ${
                        message.role === "human"
                          ? "bg-primary text-primary-foreground"
                          : "border border-border bg-background"
                      }`}
                    >
                      {message.content}
                    </div>
                  </div>
                ))
              )}
            </div>

            <div className="flex items-center justify-between gap-3 border-t border-border p-5">
              <div className="text-xs text-muted-foreground">
                {active ? "Microphone is live." : "Your microphone starts only after you click Start."}
              </div>
              {!active ? (
                <Button onClick={() => void startConversation()} disabled={!canStart}>
                  <Mic className="mr-2 h-4 w-4" />
                  {starting ? "Starting..." : "Start conversation"}
                </Button>
              ) : (
                <Button variant="outline" onClick={() => void stopConversation()}>
                  <Square className="mr-2 h-4 w-4" />
                  Stop
                </Button>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
