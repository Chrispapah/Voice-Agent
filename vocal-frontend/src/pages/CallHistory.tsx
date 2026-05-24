import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { PhoneCall, Calendar, Filter, Upload, History, MessageSquare } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { AuthRequiredError, listCalls, type CallLog } from "@/lib/api";

type CallQuality = NonNullable<CallLog["call_quality"]>;

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
  if (["assistant", "agent", "ai"].includes(normalized)) return "Voicebot";
  return role;
}

function callQuality(call: CallLog): CallQuality {
  return call.call_quality ?? "needs_attention";
}

function qualityLabel(value: CallQuality): string {
  return value.replace(/_/g, " ");
}

function qualityBadgeVariant(value: CallQuality): "default" | "secondary" | "destructive" | "outline" {
  if (value === "satisfactory") return "default";
  if (value === "unsatisfactory") return "destructive";
  return "secondary";
}

export default function CallHistoryPage() {
  const navigate = useNavigate();
  const [calls, setCalls] = useState<CallLog[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [selectedCall, setSelectedCall] = useState<CallLog | null>(null);
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [agentFilter, setAgentFilter] = useState("all");
  const [qualityFilter, setQualityFilter] = useState<"all" | CallQuality>("all");
  const [searchText, setSearchText] = useState("");

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

  const allRows = useMemo(
    () =>
      calls.map((call) => ({
        ...call,
        displayStartedAt: formatDate(call.started_at),
        displayDuration: duration(call.started_at, call.completed_at),
        transcriptCount: call.transcript?.length || 0,
      })),
    [calls],
  );

  const agentOptions = useMemo(
    () =>
      Array.from(new Set(allRows.map((call) => call.agent_name || "Unknown agent"))).sort((a, b) =>
        a.localeCompare(b),
      ),
    [allRows],
  );

  const rows = useMemo(() => {
    const fromMs = dateFrom ? new Date(`${dateFrom}T00:00:00`).getTime() : null;
    const toMs = dateTo ? new Date(`${dateTo}T23:59:59.999`).getTime() : null;
    const query = searchText.trim().toLowerCase();

    return allRows.filter((call) => {
      const startedMs = call.started_at ? new Date(call.started_at).getTime() : null;
      if (fromMs !== null && (startedMs === null || startedMs < fromMs)) return false;
      if (toMs !== null && (startedMs === null || startedMs > toMs)) return false;
      if (agentFilter !== "all" && (call.agent_name || "Unknown agent") !== agentFilter) return false;
      if (qualityFilter !== "all" && callQuality(call) !== qualityFilter) return false;
      if (!query) return true;
      return [
        call.agent_name,
        call.conversation_id,
        call.lead_id,
      ].some((value) => (value || "").toLowerCase().includes(query));
    });
  }, [agentFilter, allRows, dateFrom, dateTo, qualityFilter, searchText]);

  const hasActiveFilters = Boolean(dateFrom || dateTo || agentFilter !== "all" || qualityFilter !== "all" || searchText);

  function clearFilters(): void {
    setDateFrom("");
    setDateTo("");
    setAgentFilter("all");
    setQualityFilter("all");
    setSearchText("");
  }

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
      <div className="flex flex-wrap items-end gap-3 px-8 py-3 border-b border-border">
        <div>
          <label className="mb-1 flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
            <Calendar className="w-3.5 h-3.5" /> From
          </label>
          <input
            type="date"
            value={dateFrom}
            onChange={(event) => setDateFrom(event.target.value)}
            className="h-9 rounded-md border border-input bg-background px-3 text-sm"
          />
        </div>
        <div>
          <label className="mb-1 block text-xs font-medium text-muted-foreground">To</label>
          <input
            type="date"
            value={dateTo}
            onChange={(event) => setDateTo(event.target.value)}
            className="h-9 rounded-md border border-input bg-background px-3 text-sm"
          />
        </div>
        <div>
          <label className="mb-1 flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
            <Filter className="w-3.5 h-3.5" /> Agent
          </label>
          <select
            value={agentFilter}
            onChange={(event) => setAgentFilter(event.target.value)}
            className="h-9 min-w-40 rounded-md border border-input bg-background px-3 text-sm"
          >
            <option value="all">All agents</option>
            {agentOptions.map((agent) => (
              <option key={agent} value={agent}>{agent}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="mb-1 block text-xs font-medium text-muted-foreground">Call Quality</label>
          <select
            value={qualityFilter}
            onChange={(event) => setQualityFilter(event.target.value as "all" | CallQuality)}
            className="h-9 rounded-md border border-input bg-background px-3 text-sm"
          >
            <option value="all">All quality</option>
            <option value="satisfactory">Satisfactory</option>
            <option value="needs_attention">Needs attention</option>
            <option value="unsatisfactory">Unsatisfactory</option>
          </select>
        </div>
        <div>
          <label className="mb-1 block text-xs font-medium text-muted-foreground">Search</label>
          <input
            type="search"
            value={searchText}
            onChange={(event) => setSearchText(event.target.value)}
            placeholder="Agent, lead, conversation..."
            className="h-9 w-56 rounded-md border border-input bg-background px-3 text-sm"
          />
        </div>
        {hasActiveFilters && (
          <Button variant="ghost" size="sm" onClick={clearFilters}>
            Clear filters
          </Button>
        )}
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
              {["Time","Duration","Agent","Conversation ID","Lead ID","Call Quality","Transcript Turns"].map(h => (
                <th key={h} className="px-4 py-3 font-medium whitespace-nowrap">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {loading && (
              <tr>
                <td className="px-4 py-8 text-center text-muted-foreground" colSpan={7}>
                  Loading call history...
                </td>
              </tr>
            )}
            {!loading && rows.length === 0 && (
              <tr>
                <td className="px-4 py-8 text-center text-muted-foreground" colSpan={7}>
                  {hasActiveFilters
                    ? "No call logs match the selected filters."
                    : "No call logs yet. Test conversations and completed calls will appear here."}
                </td>
              </tr>
            )}
            {!loading &&
              rows.map((r) => (
                <tr key={r.id} className="border-t border-border hover:bg-surface-muted/40">
                  <td className="px-4 py-3 whitespace-nowrap">{r.displayStartedAt}</td>
                  <td className="px-4 py-3">{r.displayDuration}</td>
                  <td className="px-4 py-3 font-medium">{r.agent_name || "Unknown agent"}</td>
                  <td className="px-4 py-3 text-muted-foreground font-mono text-xs">{r.conversation_id}</td>
                  <td className="px-4 py-3 text-muted-foreground font-mono text-xs">{r.lead_id}</td>
                  <td className="px-4 py-3">
                    <Badge variant={qualityBadgeVariant(callQuality(r))} className="capitalize">
                      {qualityLabel(callQuality(r))}
                    </Badge>
                  </td>
                  <td className="px-4 py-3">
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="h-8 gap-1.5 px-2"
                      disabled={r.transcriptCount === 0}
                      onClick={() => setSelectedCall(r)}
                    >
                      <MessageSquare className="h-3.5 w-3.5" />
                      {r.transcriptCount === 0 ? "No transcript" : `View (${r.transcriptCount})`}
                    </Button>
                  </td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
      <footer className="px-8 py-3 border-t border-border text-xs text-muted-foreground">
        Page 1 of 1 · Showing {rows.length} of {allRows.length} sessions
      </footer>
      <Dialog open={selectedCall !== null} onOpenChange={(open) => !open && setSelectedCall(null)}>
        <DialogContent className="max-h-[85vh] max-w-3xl overflow-hidden">
          <DialogHeader>
            <DialogTitle>Conversation Transcript</DialogTitle>
            <DialogDescription>
              {selectedCall ? (
                <span className="font-mono text-xs">
                  {selectedCall.conversation_id}
                </span>
              ) : null}
            </DialogDescription>
          </DialogHeader>

          {selectedCall && (
            <div className="space-y-4">
              <div className="grid gap-3 rounded-lg border border-border bg-surface-muted/40 p-3 text-sm md:grid-cols-5">
                <div>
                  <div className="text-xs text-muted-foreground">Started</div>
                  <div className="mt-1">{formatDate(selectedCall.started_at)}</div>
                </div>
                <div>
                  <div className="text-xs text-muted-foreground">Duration</div>
                  <div className="mt-1">{duration(selectedCall.started_at, selectedCall.completed_at)}</div>
                </div>
                <div>
                  <div className="text-xs text-muted-foreground">Agent</div>
                  <div className="mt-1">{selectedCall.agent_name || "Unknown agent"}</div>
                </div>
                <div>
                  <div className="text-xs text-muted-foreground">Call Quality</div>
                  <div className="mt-1 capitalize">{qualityLabel(callQuality(selectedCall))}</div>
                </div>
                <div>
                  <div className="text-xs text-muted-foreground">Turns</div>
                  <div className="mt-1">{selectedCall.transcript.length}</div>
                </div>
              </div>

              <div className="max-h-[55vh] space-y-3 overflow-y-auto pr-2">
                {selectedCall.transcript.length === 0 ? (
                  <div className="rounded-lg border border-dashed border-border py-10 text-center text-sm text-muted-foreground">
                    No transcript turns were captured for this conversation.
                  </div>
                ) : (
                  selectedCall.transcript.map((turn, index) => {
                    const label = roleLabel(turn.role);
                    const isClient = label === "Client";
                    return (
                      <div
                        key={`${turn.role}-${index}`}
                        className={`rounded-xl border border-border p-4 ${
                          isClient ? "bg-card" : "bg-primary/5"
                        }`}
                      >
                        <div className="mb-2">
                          <Badge variant={isClient ? "secondary" : "default"}>{label}</Badge>
                        </div>
                        <p className="whitespace-pre-wrap text-sm leading-6">{turn.content || "-"}</p>
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
