import { FormEvent, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ChevronLeft, Plus, Wrench } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  AuthRequiredError,
  createTool,
  listTools,
  updateTool,
  type ToolDefinition,
  type ToolDefinitionKind,
} from "@/lib/api";

const toolKinds: ToolDefinitionKind[] = ["http", "webhook", "custom"];

function endpointFromConfig(config: Record<string, unknown>): string {
  return typeof config.endpoint_url === "string" ? config.endpoint_url : "";
}

export default function ToolsPage() {
  const navigate = useNavigate();
  const [tools, setTools] = useState<ToolDefinition[]>([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [selectedToolId, setSelectedToolId] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [kind, setKind] = useState<ToolDefinitionKind>("http");
  const [endpoint, setEndpoint] = useState("");

  useEffect(() => {
    listTools()
      .then(setTools)
      .catch((err: unknown) => {
        if (err instanceof AuthRequiredError) {
          navigate("/auth");
          return;
        }
        setError(err instanceof Error ? err.message : "Failed to load tools");
      })
      .finally(() => setLoading(false));
  }, [navigate]);

  function resetForm() {
    setSelectedToolId(null);
    setName("");
    setDescription("");
    setEndpoint("");
    setKind("http");
    setError("");
  }

  function selectTool(tool: ToolDefinition) {
    setSelectedToolId(tool.id);
    setName(tool.name);
    setDescription(tool.description);
    setKind(tool.kind);
    setEndpoint(endpointFromConfig(tool.config_json));
    setError("");
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!name.trim()) return;
    setSubmitting(true);
    setError("");
    const existingConfig = selectedToolId
      ? tools.find((tool) => tool.id === selectedToolId)?.config_json ?? {}
      : {};
    const payload = {
      name: name.trim(),
      description: description.trim(),
      kind,
      config_json: {
        ...existingConfig,
        endpoint_url: endpoint.trim(),
        method: typeof existingConfig.method === "string" ? existingConfig.method : "POST",
        status: typeof existingConfig.status === "string" ? existingConfig.status : "placeholder",
      },
    };
    try {
      if (selectedToolId) {
        const updated = await updateTool(selectedToolId, payload);
        setTools((current) => current.map((tool) => (tool.id === updated.id ? updated : tool)));
        selectTool(updated);
      } else {
        const created = await createTool(payload);
        setTools((current) => [created, ...current]);
        resetForm();
      }
    } catch (err: unknown) {
      if (err instanceof AuthRequiredError) {
        navigate("/auth");
        return;
      }
      setError(
        err instanceof Error
          ? err.message
          : selectedToolId
            ? "Failed to update tool"
            : "Failed to create tool",
      );
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="flex flex-1 min-h-0">
      <aside className="hidden lg:flex flex-col w-72 border-r border-border bg-surface-muted/40">
        <div className="flex items-center justify-between px-4 py-4 border-b border-border">
          <div className="flex items-center gap-2 text-sm font-medium">
            <Wrench className="w-4 h-4" /> Tools
          </div>
          <button
            type="button"
            onClick={resetForm}
            className="w-7 h-7 rounded-md bg-foreground text-background flex items-center justify-center hover:opacity-90"
            aria-label="Create a new tool"
          >
            <Plus className="w-4 h-4" />
          </button>
        </div>
        <div className="flex-1 overflow-y-auto p-3 space-y-2">
          {loading && <p className="px-2 py-3 text-xs text-muted-foreground">Loading tools...</p>}
          {!loading && tools.length === 0 && (
            <div className="grid h-full place-items-center text-xs text-muted-foreground">
              No tools defined yet
            </div>
          )}
          {tools.map((tool) => (
            <button
              key={tool.id}
              type="button"
              onClick={() => selectTool(tool)}
              className={`w-full rounded-lg border px-3 py-2 text-left transition hover:border-primary/50 hover:bg-primary/5 ${
                selectedToolId === tool.id ? "border-primary bg-primary/10" : "border-border bg-card"
              }`}
            >
              <div className="text-sm font-medium">{tool.name}</div>
              <div className="mt-1 text-[11px] uppercase tracking-wide text-muted-foreground">{tool.kind}</div>
            </button>
          ))}
        </div>
      </aside>

      <div className="flex-1 overflow-auto px-8 py-6">
        <div className="mx-auto max-w-3xl">
          <button
            type="button"
            onClick={() => navigate("/agents")}
            className="mb-5 inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground"
          >
            <ChevronLeft className="w-4 h-4" /> Back to agents
          </button>

          <div className="rounded-xl border border-border bg-card p-6 shadow-soft">
            <div className="mb-5 flex items-start gap-3">
              <div className="rounded-xl bg-gradient-soft p-3">
                <Wrench className="h-5 w-5 text-primary" />
              </div>
              <div>
                <h1 className="text-xl font-semibold tracking-tight">
                  {selectedToolId ? "Edit tool" : "Tools"}
                </h1>
                <p className="mt-1 text-sm text-muted-foreground">
                  {selectedToolId
                    ? "Update this reusable tool, then attach it to a single prompt agent or individual flow nodes."
                    : "Define reusable tools here, then attach them to a single prompt agent or individual flow nodes."}
                </p>
              </div>
            </div>

            {error && (
              <div className="mb-4 rounded-lg border border-destructive/20 bg-destructive/10 px-4 py-3 text-sm text-destructive">
                {error}
              </div>
            )}

            <form onSubmit={handleSubmit} className="grid gap-4">
              <div>
                <label className="text-[11px] font-semibold text-muted-foreground tracking-wide">TOOL NAME</label>
                <input
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Check calendar availability"
                  className="mt-2 w-full rounded-lg border border-border bg-background px-3 py-2 text-sm"
                />
              </div>
              <div>
                <label className="text-[11px] font-semibold text-muted-foreground tracking-wide">DESCRIPTION</label>
                <textarea
                  rows={3}
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="Explain when the agent should call this tool."
                  className="mt-2 w-full rounded-lg border border-border bg-background px-3 py-2 text-sm resize-none"
                />
              </div>
              <div className="grid gap-4 sm:grid-cols-2">
                <div>
                  <label className="text-[11px] font-semibold text-muted-foreground tracking-wide">TYPE</label>
                  <select
                    value={kind}
                    onChange={(e) => setKind(e.target.value as ToolDefinitionKind)}
                    className="mt-2 w-full rounded-lg border border-border bg-background px-3 py-2 text-sm"
                  >
                    {toolKinds.map((toolKind) => (
                      <option key={toolKind} value={toolKind}>{toolKind}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="text-[11px] font-semibold text-muted-foreground tracking-wide">ENDPOINT PLACEHOLDER</label>
                  <input
                    value={endpoint}
                    onChange={(e) => setEndpoint(e.target.value)}
                    placeholder="https://api.example.com/tool"
                    className="mt-2 w-full rounded-lg border border-border bg-background px-3 py-2 text-sm"
                  />
                </div>
              </div>
              <div className="flex gap-2">
                <Button
                  type="submit"
                  size="sm"
                  className="w-fit gap-1.5 bg-gradient-primary text-primary-foreground shadow-elegant"
                  disabled={submitting || !name.trim()}
                >
                  {!selectedToolId && <Plus className="w-4 h-4" />}
                  {submitting
                    ? selectedToolId
                      ? "Saving..."
                      : "Creating..."
                    : selectedToolId
                      ? "Save changes"
                      : "Create tool"}
                </Button>
                {selectedToolId && (
                  <Button type="button" size="sm" variant="outline" onClick={resetForm} disabled={submitting}>
                    Cancel
                  </Button>
                )}
              </div>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}
