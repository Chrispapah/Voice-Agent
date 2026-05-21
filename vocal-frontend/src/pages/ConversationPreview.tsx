import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { CalendarClock, CheckCircle2, MessageSquare, PhoneCall } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { getPublicConversationPreview, type PublicConversationPreview } from "@/lib/api";

function formatDate(value: string | null): string {
  if (!value) return "-";
  return new Date(value).toLocaleString();
}

function duration(startedAt: string | null, completedAt: string | null): string {
  if (!startedAt || !completedAt) return "-";
  const ms = new Date(completedAt).getTime() - new Date(startedAt).getTime();
  if (!Number.isFinite(ms) || ms < 0) return "-";
  const totalSeconds = Math.round(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

function roleLabel(role: string): string {
  const normalized = role.toLowerCase();
  if (["human", "user", "lead", "caller"].includes(normalized)) return "Client";
  if (["assistant", "agent", "ai"].includes(normalized)) return "Agent";
  return role;
}

export default function ConversationPreviewPage() {
  const { token } = useParams();
  const [preview, setPreview] = useState<PublicConversationPreview | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!token) {
      setError("Preview link is missing a token.");
      setLoading(false);
      return;
    }

    getPublicConversationPreview(token)
      .then(setPreview)
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : "This preview link is unavailable.");
      })
      .finally(() => setLoading(false));
  }, [token]);

  const stats = useMemo(() => {
    if (!preview) return [];
    const conversation = preview.conversation;
    return [
      { label: "Started", value: formatDate(conversation.started_at), icon: CalendarClock },
      { label: "Duration", value: duration(conversation.started_at, conversation.completed_at), icon: PhoneCall },
      { label: "Outcome", value: conversation.call_outcome.replace(/_/g, " "), icon: CheckCircle2 },
      { label: "Turns", value: String(conversation.transcript.length), icon: MessageSquare },
    ];
  }, [preview]);

  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="border-b border-border bg-surface">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-5">
          <div>
            <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Conversation Preview
            </div>
            <h1 className="mt-1 text-2xl font-semibold">Client-ready transcript</h1>
          </div>
          <Link to="/" className="text-sm font-medium text-primary">
            Akoi
          </Link>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-6 py-8">
        {loading && (
          <Card>
            <CardContent className="py-12 text-center text-sm text-muted-foreground">
              Loading conversation preview...
            </CardContent>
          </Card>
        )}

        {!loading && error && (
          <Card>
            <CardContent className="py-12 text-center">
              <div className="text-base font-semibold">Preview unavailable</div>
              <p className="mt-2 text-sm text-muted-foreground">
                The link may be invalid, expired, or revoked.
              </p>
            </CardContent>
          </Card>
        )}

        {!loading && !error && preview && (
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                  <div>
                    <CardTitle>{preview.conversation.agent_name}</CardTitle>
                    <p className="mt-1 font-mono text-xs text-muted-foreground">
                      {preview.conversation.conversation_id}
                    </p>
                  </div>
                  <Badge variant={preview.conversation.meeting_booked ? "default" : "secondary"}>
                    {preview.conversation.meeting_booked ? "Meeting booked" : "No meeting booked"}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid gap-3 md:grid-cols-4">
                  {stats.map((item) => (
                    <div key={item.label} className="rounded-lg border border-border bg-surface-muted/40 p-3">
                      <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                        <item.icon className="h-3.5 w-3.5" />
                        {item.label}
                      </div>
                      <div className="mt-1 text-sm font-medium capitalize">{item.value}</div>
                    </div>
                  ))}
                </div>
                {(preview.conversation.proposed_slot || preview.conversation.follow_up_action) && (
                  <div className="mt-4 rounded-lg border border-border p-4 text-sm">
                    {preview.conversation.proposed_slot && (
                      <div>
                        <span className="font-medium">Proposed slot:</span> {preview.conversation.proposed_slot}
                      </div>
                    )}
                    {preview.conversation.follow_up_action && (
                      <div className="mt-1">
                        <span className="font-medium">Follow-up:</span> {preview.conversation.follow_up_action}
                      </div>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Transcript</CardTitle>
              </CardHeader>
              <CardContent>
                {preview.conversation.transcript.length === 0 ? (
                  <div className="rounded-lg border border-dashed border-border py-10 text-center text-sm text-muted-foreground">
                    No transcript turns were captured for this conversation.
                  </div>
                ) : (
                  <div className="space-y-3">
                    {preview.conversation.transcript.map((turn, index) => {
                      const isClient = roleLabel(turn.role) === "Client";
                      return (
                        <div
                          key={`${turn.role}-${index}`}
                          className={`rounded-xl border border-border p-4 ${
                            isClient ? "bg-card" : "bg-primary/5"
                          }`}
                        >
                          <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                            {roleLabel(turn.role)}
                          </div>
                          <p className="whitespace-pre-wrap text-sm leading-6">{turn.content}</p>
                        </div>
                      );
                    })}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        )}
      </main>
    </div>
  );
}
