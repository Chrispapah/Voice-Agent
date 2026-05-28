import { FormEvent, useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { ChevronLeft, Plus, Trash2, Wrench } from "lucide-react";
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
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import WorkspaceEnvVarsPanel from "@/components/WorkspaceEnvVarsPanel";
import {
  AuthRequiredError,
  createTool,
  deleteTool,
  listAuthConnections,
  listTools,
  updateTool,
  type AuthConnection,
  type ToolDefinition,
  type ToolDefinitionKind,
} from "@/lib/api";
import {
  configToJson,
  defaultHttpToolConfig,
  parseToolConfig,
  syncPathParameters,
  type HttpToolConfigV1,
  type ToolHeader,
  type ToolParameterDef,
} from "@/lib/toolConfig";

const toolKinds: ToolDefinitionKind[] = ["http", "webhook", "custom"];

const labelClass = "text-[11px] font-semibold text-muted-foreground tracking-wide";

function FieldLabel({ children }: { children: React.ReactNode }) {
  return <label className={labelClass}>{children}</label>;
}

function textInput(className = "") {
  return `mt-2 w-full rounded-lg border border-border bg-background px-3 py-2 text-sm ${className}`;
}

export default function ToolsPage() {
  const navigate = useNavigate();
  const [tools, setTools] = useState<ToolDefinition[]>([]);
  const [authConnections, setAuthConnections] = useState<AuthConnection[]>([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState("");
  const [selectedToolId, setSelectedToolId] = useState<string | null>(null);
  const [toolPendingDelete, setToolPendingDelete] = useState<ToolDefinition | null>(null);
  const [envDialogOpen, setEnvDialogOpen] = useState(false);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [kind, setKind] = useState<ToolDefinitionKind>("http");
  const [config, setConfig] = useState<HttpToolConfigV1>(defaultHttpToolConfig());
  const [parametersJson, setParametersJson] = useState(
    JSON.stringify(defaultHttpToolConfig().parameters, null, 2),
  );

  useEffect(() => {
    Promise.all([listTools(), listAuthConnections().catch(() => [])])
      .then(([toolList, connections]) => {
        setTools(toolList);
        setAuthConnections(connections);
      })
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
    setKind("http");
    const d = defaultHttpToolConfig();
    setConfig(d);
    setParametersJson(JSON.stringify(d.parameters, null, 2));
    setError("");
  }

  function selectTool(tool: ToolDefinition) {
    setSelectedToolId(tool.id);
    setName(tool.name);
    setDescription(tool.description);
    setKind(tool.kind);
    const parsed = parseToolConfig(tool.config_json);
    setConfig(parsed);
    setParametersJson(JSON.stringify(parsed.parameters ?? defaultHttpToolConfig().parameters, null, 2));
    setError("");
  }

  function updateConfig(patch: Partial<HttpToolConfigV1>) {
    setConfig((current) => {
      const next = { ...current, ...patch };
      if (patch.url !== undefined) {
        next.path_parameters = syncPathParameters(patch.url, current.path_parameters);
      }
      return next;
    });
  }

  function setHeaders(headers: ToolHeader[]) {
    updateConfig({ headers });
  }

  function setQueryParams(query_parameters: ToolParameterDef[]) {
    updateConfig({ query_parameters });
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!name.trim()) return;
    if (kind === "custom") {
      setError("Custom tools are not supported at runtime yet.");
      return;
    }
    let parameters: Record<string, unknown> | null = null;
    try {
      parameters = JSON.parse(parametersJson) as Record<string, unknown>;
    } catch {
      setError("LLM parameters must be valid JSON Schema.");
      return;
    }
    setSubmitting(true);
    setError("");
    const payload = {
      name: name.trim(),
      description: description.trim(),
      kind,
      config_json: configToJson({ ...config, parameters }),
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
      setError(err instanceof Error ? err.message : "Failed to save tool");
    } finally {
      setSubmitting(false);
    }
  }

  async function confirmDeleteTool() {
    if (!toolPendingDelete) return;
    setDeleting(true);
    setError("");
    try {
      await deleteTool(toolPendingDelete.id);
      setTools((current) => current.filter((t) => t.id !== toolPendingDelete.id));
      if (selectedToolId === toolPendingDelete.id) resetForm();
      setToolPendingDelete(null);
    } catch (err: unknown) {
      if (err instanceof AuthRequiredError) {
        navigate("/auth");
        return;
      }
      setError(err instanceof Error ? err.message : "Failed to delete tool");
    } finally {
      setDeleting(false);
    }
  }

  const isHttpKind = kind === "http" || kind === "webhook";

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
            <div className="grid h-full place-items-center text-xs text-muted-foreground">No tools defined yet</div>
          )}
          {tools.map((tool) => {
            const c = parseToolConfig(tool.config_json);
            return (
              <div
                key={tool.id}
                className={`flex gap-1 rounded-lg border p-1 transition ${
                  selectedToolId === tool.id ? "border-primary bg-primary/10" : "border-border bg-card"
                }`}
              >
                <button
                  type="button"
                  onClick={() => selectTool(tool)}
                  className="min-w-0 flex-1 rounded-md px-2 py-2 text-left hover:bg-primary/5"
                >
                  <div className="text-sm font-medium truncate">{tool.name}</div>
                  <div className="mt-1 text-[11px] text-muted-foreground truncate">
                    {tool.kind} · {c.method} {c.url ? c.url.slice(0, 40) : "no URL"}
                  </div>
                </button>
                <button
                  type="button"
                  onClick={() => setToolPendingDelete(tool)}
                  className="shrink-0 self-center rounded-md p-2 text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
                  aria-label={`Delete ${tool.name}`}
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            );
          })}
        </div>
      </aside>

      <div className="flex-1 overflow-auto px-4 py-5 sm:px-8 sm:py-6">
        <div className="mx-auto max-w-3xl">
          <button
            type="button"
            onClick={() => navigate("/agents")}
            className="mb-5 inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground"
          >
            <ChevronLeft className="w-4 h-4" /> Back to agents
          </button>

          <div className="rounded-xl border border-border bg-card p-5 sm:p-6 shadow-soft">
            <div className="mb-5 flex items-start justify-between gap-3">
              <div>
                <h1 className="text-xl font-semibold tracking-tight">
                  {selectedToolId ? "Edit tool" : "Add webhook tool"}
                </h1>
                <p className="mt-1 text-sm text-muted-foreground">
                  Configure HTTP tools for your agents. Requires Groq as the LLM provider when attached.
                </p>
              </div>
              <Button type="button" variant="outline" size="sm" onClick={() => setEnvDialogOpen(true)}>
                Env variables
              </Button>
            </div>

            {error && (
              <div className="mb-4 rounded-lg border border-destructive/20 bg-destructive/10 px-4 py-3 text-sm text-destructive">
                {error}
              </div>
            )}

            <form onSubmit={handleSubmit} className="grid gap-4">
              <div>
                <FieldLabel>NAME</FieldLabel>
                <input value={name} onChange={(e) => setName(e.target.value)} className={textInput()} placeholder="check_calendar" />
              </div>
              <div>
                <FieldLabel>DESCRIPTION (how/when to use)</FieldLabel>
                <textarea
                  rows={3}
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  className={`${textInput()} resize-none`}
                  placeholder="Call when the user asks about appointment availability."
                />
              </div>
              <div className="grid gap-4 sm:grid-cols-2">
                <div>
                  <FieldLabel>TYPE</FieldLabel>
                  <select
                    value={kind}
                    onChange={(e) => setKind(e.target.value as ToolDefinitionKind)}
                    className={textInput()}
                  >
                    {toolKinds.map((k) => (
                      <option key={k} value={k}>{k}</option>
                    ))}
                  </select>
                </div>
                {isHttpKind && (
                  <div>
                    <FieldLabel>METHOD</FieldLabel>
                    <select
                      value={config.method}
                      onChange={(e) => updateConfig({ method: e.target.value as HttpToolConfigV1["method"] })}
                      className={textInput()}
                    >
                      {(["GET", "POST", "PUT", "PATCH", "DELETE"] as const).map((m) => (
                        <option key={m} value={m}>{m}</option>
                      ))}
                    </select>
                  </div>
                )}
              </div>

              {kind === "custom" && (
                <p className="text-sm text-muted-foreground rounded-lg border border-border bg-muted/30 p-3">
                  Custom server-side plugins are coming soon. Use HTTP or webhook for now.
                </p>
              )}

              {isHttpKind && (
                <>
                  <div>
                    <FieldLabel>URL</FieldLabel>
                    <input
                      value={config.url}
                      onChange={(e) => updateConfig({ url: e.target.value })}
                      className={textInput()}
                      placeholder="https://api.example.com/users/{userId}"
                    />
                    <p className="mt-1 text-xs text-muted-foreground">
                      Type <code className="rounded bg-muted px-1">{`{{VAR}}`}</code> for environment variables.{" "}
                      <button type="button" className="text-primary hover:underline" onClick={() => setEnvDialogOpen(true)}>
                        Manage variables
                      </button>
                    </p>
                  </div>

                  <div className="grid gap-4 sm:grid-cols-2">
                    <div>
                      <FieldLabel>RESPONSE TIMEOUT (seconds)</FieldLabel>
                      <input
                        type="number"
                        min={1}
                        max={120}
                        value={config.response_timeout_seconds}
                        onChange={(e) => updateConfig({ response_timeout_seconds: Number(e.target.value) || 20 })}
                        className={textInput()}
                      />
                    </div>
                    <div className="flex items-end pb-2">
                      <label className="flex items-center gap-2 text-sm">
                        <input
                          type="checkbox"
                          checked={config.disable_interruptions}
                          onChange={(e) => updateConfig({ disable_interruptions: e.target.checked })}
                        />
                        Disable interruptions while running
                      </label>
                    </div>
                  </div>

                  <div className="grid gap-4 sm:grid-cols-2">
                    <div>
                      <FieldLabel>PRE-TOOL SPEECH</FieldLabel>
                      <select
                        value={config.pre_tool_speech}
                        onChange={(e) =>
                          updateConfig({ pre_tool_speech: e.target.value as HttpToolConfigV1["pre_tool_speech"] })
                        }
                        className={textInput()}
                      >
                        <option value="auto">Auto</option>
                        <option value="force">Force</option>
                        <option value="disabled">Disabled</option>
                      </select>
                    </div>
                    <div>
                      <FieldLabel>EXECUTION MODE</FieldLabel>
                      <select
                        value={config.execution_mode}
                        onChange={(e) =>
                          updateConfig({ execution_mode: e.target.value as HttpToolConfigV1["execution_mode"] })
                        }
                        className={textInput()}
                      >
                        <option value="default">Default</option>
                        <option value="blocking">Blocking</option>
                      </select>
                    </div>
                  </div>

                  {config.pre_tool_speech === "force" && (
                    <div>
                      <FieldLabel>PRE-TOOL SPEECH TEXT</FieldLabel>
                      <input
                        value={config.pre_tool_speech_text ?? ""}
                        onChange={(e) => updateConfig({ pre_tool_speech_text: e.target.value })}
                        className={textInput()}
                        placeholder="One moment while I check that."
                      />
                    </div>
                  )}

                  <div className="grid gap-4 sm:grid-cols-2">
                    <div>
                      <FieldLabel>TOOL CALL SOUND</FieldLabel>
                      <select
                        value={config.tool_call_sound}
                        onChange={(e) =>
                          updateConfig({ tool_call_sound: e.target.value as HttpToolConfigV1["tool_call_sound"] })
                        }
                        className={textInput()}
                      >
                        <option value="none">None</option>
                        <option value="click">Click</option>
                        <option value="custom_url">Custom URL</option>
                      </select>
                    </div>
                    {config.tool_call_sound === "custom_url" && (
                      <div>
                        <FieldLabel>SOUND URL</FieldLabel>
                        <input
                          value={config.tool_call_sound_url ?? ""}
                          onChange={(e) => updateConfig({ tool_call_sound_url: e.target.value })}
                          className={textInput()}
                        />
                      </div>
                    )}
                  </div>

                  <div>
                    <FieldLabel>AUTHENTICATION</FieldLabel>
                    <select
                      value={config.auth.type}
                      onChange={(e) =>
                        setConfig((c) => ({
                          ...c,
                          auth: { ...c.auth, type: e.target.value as HttpToolConfigV1["auth"]["type"] },
                        }))
                      }
                      className={textInput()}
                    >
                      <option value="none">None</option>
                      <option value="bearer">Bearer token</option>
                      <option value="basic">Basic auth</option>
                      <option value="api_key_header">API key header</option>
                      <option value="connection">Auth connection</option>
                    </select>
                    {config.auth.type === "connection" && (
                      <select
                        value={config.auth.connection_id ?? ""}
                        onChange={(e) =>
                          setConfig((c) => ({
                            ...c,
                            auth: { ...c.auth, connection_id: e.target.value || null },
                          }))
                        }
                        className={`${textInput()} mt-2`}
                      >
                        <option value="">Select connection</option>
                        {authConnections.map((conn) => (
                          <option key={conn.id} value={conn.id}>{conn.label}</option>
                        ))}
                      </select>
                    )}
                    {config.auth.type === "bearer" && (
                      <input
                        value={config.auth.bearer_token ?? ""}
                        onChange={(e) =>
                          setConfig((c) => ({ ...c, auth: { ...c.auth, bearer_token: e.target.value } }))
                        }
                        placeholder="Bearer token or {{VAR}}"
                        className={`${textInput()} mt-2`}
                      />
                    )}
                    {config.auth.type === "api_key_header" && (
                      <div className="mt-2 grid gap-2 sm:grid-cols-2">
                        <input
                          value={config.auth.api_key_header_name ?? "X-Api-Key"}
                          onChange={(e) =>
                            setConfig((c) => ({ ...c, auth: { ...c.auth, api_key_header_name: e.target.value } }))
                          }
                          placeholder="Header name"
                          className={textInput()}
                        />
                        <input
                          value={config.auth.api_key_value ?? ""}
                          onChange={(e) =>
                            setConfig((c) => ({ ...c, auth: { ...c.auth, api_key_value: e.target.value } }))
                          }
                          placeholder="Value or {{VAR}}"
                          className={textInput()}
                        />
                      </div>
                    )}
                    {authConnections.length === 0 && config.auth.type === "connection" && (
                      <p className="mt-1 text-xs text-muted-foreground">
                        No auth connections.{" "}
                        <Link to="/settings" className="text-primary hover:underline">Create in Settings</Link>
                      </p>
                    )}
                  </div>

                  <HeaderEditor headers={config.headers} onChange={setHeaders} />
                  <ParamEditor title="QUERY PARAMETERS" params={config.query_parameters} onChange={setQueryParams} />

                  {config.path_parameters.length > 0 && (
                    <div>
                      <FieldLabel>PATH PARAMETERS (from URL)</FieldLabel>
                      <ul className="mt-2 space-y-1 text-sm text-muted-foreground">
                        {config.path_parameters.map((p) => (
                          <li key={p.name} className="font-mono">
                            {`{${p.name}}`} {p.required ? "(required)" : ""}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  <div>
                    <FieldLabel>LLM PARAMETERS (JSON Schema)</FieldLabel>
                    <textarea
                      rows={8}
                      value={parametersJson}
                      onChange={(e) => setParametersJson(e.target.value)}
                      className={`${textInput()} font-mono text-xs resize-y`}
                    />
                  </div>
                </>
              )}

              <div className="flex flex-wrap gap-2">
                <Button
                  type="submit"
                  size="sm"
                  disabled={submitting || !name.trim() || kind === "custom"}
                  className="w-fit gap-1.5 bg-gradient-primary text-primary-foreground shadow-elegant"
                >
                  {!selectedToolId && <Plus className="w-4 h-4" />}
                  {submitting ? "Saving..." : selectedToolId ? "Save changes" : "Create tool"}
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

      <Dialog open={envDialogOpen} onOpenChange={setEnvDialogOpen}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>Environment variables</DialogTitle>
          </DialogHeader>
          <WorkspaceEnvVarsPanel onClose={() => setEnvDialogOpen(false)} />
        </DialogContent>
      </Dialog>

      <AlertDialog open={!!toolPendingDelete} onOpenChange={(open) => !open && setToolPendingDelete(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete this tool?</AlertDialogTitle>
            <AlertDialogDescription>
              {toolPendingDelete ? `"${toolPendingDelete.name}" will be removed.` : ""}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={deleting}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              onClick={(e) => {
                e.preventDefault();
                void confirmDeleteTool();
              }}
              disabled={deleting}
            >
              {deleting ? "Deleting..." : "Delete"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

function HeaderEditor({
  headers,
  onChange,
}: {
  headers: ToolHeader[];
  onChange: (h: ToolHeader[]) => void;
}) {
  return (
    <div>
      <div className="flex items-center justify-between">
        <FieldLabel>HEADERS</FieldLabel>
        <Button type="button" variant="outline" size="sm" onClick={() => onChange([...headers, { name: "", value: "" }])}>
          Add header
        </Button>
      </div>
      <div className="mt-2 space-y-2">
        {headers.map((h, i) => (
          <div key={i} className="grid gap-2 sm:grid-cols-[1fr_1fr_auto]">
            <input
              value={h.name}
              onChange={(e) => {
                const next = [...headers];
                next[i] = { ...next[i], name: e.target.value };
                onChange(next);
              }}
              placeholder="Header name"
              className={textInput("mt-0")}
            />
            <input
              value={h.value}
              onChange={(e) => {
                const next = [...headers];
                next[i] = { ...next[i], value: e.target.value };
                onChange(next);
              }}
              placeholder="Value or {{VAR}}"
              className={textInput("mt-0")}
            />
            <Button type="button" variant="ghost" size="icon" onClick={() => onChange(headers.filter((_, j) => j !== i))}>
              <Trash2 className="w-4 h-4" />
            </Button>
          </div>
        ))}
      </div>
    </div>
  );
}

function ParamEditor({
  title,
  params,
  onChange,
}: {
  title: string;
  params: ToolParameterDef[];
  onChange: (p: ToolParameterDef[]) => void;
}) {
  return (
    <div>
      <div className="flex items-center justify-between">
        <FieldLabel>{title}</FieldLabel>
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() =>
            onChange([...params, { name: "", description: "", type: "string", required: false }])
          }
        >
          Add parameter
        </Button>
      </div>
      <div className="mt-2 space-y-2">
        {params.map((p, i) => (
          <div key={i} className="grid gap-2 rounded-lg border border-border p-2 sm:grid-cols-4">
            <input
              value={p.name}
              onChange={(e) => {
                const next = [...params];
                next[i] = { ...next[i], name: e.target.value };
                onChange(next);
              }}
              placeholder="name"
              className={textInput("mt-0")}
            />
            <input
              value={p.description}
              onChange={(e) => {
                const next = [...params];
                next[i] = { ...next[i], description: e.target.value };
                onChange(next);
              }}
              placeholder="description"
              className={`${textInput("mt-0")} sm:col-span-2`}
            />
            <label className="flex items-center gap-2 text-xs self-center">
              <input
                type="checkbox"
                checked={p.required}
                onChange={(e) => {
                  const next = [...params];
                  next[i] = { ...next[i], required: e.target.checked };
                  onChange(next);
                }}
              />
              Required
            </label>
          </div>
        ))}
      </div>
    </div>
  );
}
