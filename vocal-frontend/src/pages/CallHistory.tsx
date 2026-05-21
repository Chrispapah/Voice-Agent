import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { PhoneCall, Calendar, Filter, Upload, History, Share2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { toast } from "@/components/ui/sonner";
import { AuthRequiredError, createConversationShare, listCalls, type CallLog } from "@/lib/api";

const dot = (color: string) => <span className={`inline-block w-1.5 h-1.5 rounded-full mr-1.5 ${color}`} />;

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

export default function CallHistoryPage() {
  const navigate = useNavigate();
  const [calls, setCalls] = useState<CallLog[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [sharingCallId, setSharingCallId] = useState<string | null>(null);

  useEffect(() => {
    listCalls()
      .then(setCalls)
      .catch((err: unknown) => {
        if (err instanceof AuthRequiredError) {
          navigate("/auth");
          return;
        }
        setError(err instanceof Error ? err.message : "Failed to load call history");
      })
      .finally(() => setLoading(false));
  }, [navigate]);

  const rows = useMemo(
    () =>
      calls.map((call) => ({
        ...call,
        displayStartedAt: formatDate(call.started_at),
        displayDuration: duration(call.started_at, call.completed_at),
        transcriptCount: call.transcript?.length || 0,
      })),
    [calls],
  );

  const shareConversation = async (call: CallLog) => {
    if (!call.transcript?.length) {
      toast.info("No transcript yet", { description: "Only conversations with transcript turns can be shared." });
      return;
    }

    setSharingCallId(call.id);
    try {
      const share = await createConversationShare(call.id);
      await navigator.clipboard.writeText(share.preview_url);
      toast.success("Preview link copied", { description: "You can send this read-only link to your client." });
    } catch (err: unknown) {
      if (err instanceof AuthRequiredError) {
        navigate("/auth");
        return;
      }
      toast.error("Could not create preview link", {
        description: err instanceof Error ? err.message : "Please try again.",
      });
    } finally {
      setSharingCallId(null);
    }
  };

  return (
    <div className="flex flex-col flex-1 min-h-0">
      <header className="flex items-center justify-between px-8 py-5 border-b border-border">
        <div className="flex items-center gap-2">
          <PhoneCall className="w-4 h-4" />
          <h1 className="text-base font-semibold">Call History</h1>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" className="gap-1.5"><History className="w-4 h-4" /></Button>
          <Button variant="outline" size="sm" className="gap-1.5"><Upload className="w-4 h-4" /> Export</Button>
          <Button variant="outline" size="sm">Customize View</Button>
          <Button variant="outline" size="sm">Custom Attributes</Button>
        </div>
      </header>
      <div className="flex items-center gap-2 px-8 py-3 border-b border-border">
        <Button variant="outline" size="sm" className="gap-1.5"><Calendar className="w-4 h-4" /> Date Range</Button>
        <Button variant="outline" size="sm" className="gap-1.5"><Filter className="w-4 h-4" /> Filter</Button>
      </div>
      <div className="flex-1 overflow-auto">
        {error && (
          <div className="m-4 rounded-lg border border-destructive/20 bg-destructive/10 px-4 py-3 text-sm text-destructive">
            {error}
          </div>
        )}
        <table className="w-full text-sm">
          <thead className="bg-surface-muted/60 text-muted-foreground sticky top-0">
            <tr className="text-left">
              {["Time","Duration","Conversation ID","Lead ID","Outcome","Meeting Booked","Proposed Slot","Follow-Up Action","Transcript Turns","Preview Link"].map(h => (
                <th key={h} className="px-4 py-3 font-medium whitespace-nowrap">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {loading && (
              <tr>
                <td className="px-4 py-8 text-center text-muted-foreground" colSpan={10}>
                  Loading call history...
                </td>
              </tr>
            )}
            {!loading && rows.length === 0 && (
              <tr>
                <td className="px-4 py-8 text-center text-muted-foreground" colSpan={10}>
                  No call logs yet. Test conversations and completed calls will appear here.
                </td>
              </tr>
            )}
            {!loading &&
              rows.map((r) => (
                <tr key={r.id} className="border-t border-border hover:bg-surface-muted/40">
                  <td className="px-4 py-3 whitespace-nowrap">{r.displayStartedAt}</td>
                  <td className="px-4 py-3">{r.displayDuration}</td>
                  <td className="px-4 py-3 text-muted-foreground font-mono text-xs">{r.conversation_id}</td>
                  <td className="px-4 py-3 text-muted-foreground font-mono text-xs">{r.lead_id}</td>
                  <td className="px-4 py-3">{dot(r.call_outcome === "meeting_booked" ? "bg-success" : "bg-muted-foreground")}{r.call_outcome}</td>
                  <td className="px-4 py-3">{r.meeting_booked ? "Yes" : "No"}</td>
                  <td className="px-4 py-3">{r.proposed_slot || "-"}</td>
                  <td className="px-4 py-3">{r.follow_up_action || "-"}</td>
                  <td className="px-4 py-3">{r.transcriptCount}</td>
                  <td className="px-4 py-3">
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      className="gap-1.5"
                      disabled={sharingCallId === r.id || r.transcriptCount === 0}
                      onClick={() => void shareConversation(r)}
                    >
                      <Share2 className="w-3.5 h-3.5" />
                      {sharingCallId === r.id ? "Creating..." : "Copy"}
                    </Button>
                  </td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
      <footer className="px-8 py-3 border-t border-border text-xs text-muted-foreground">
        Page 1 of 1 · Total Sessions: {rows.length}
      </footer>
    </div>
  );
}
