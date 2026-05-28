import { FormEvent, useEffect, useState } from "react";
import { Plus, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  createAuthConnection,
  deleteAuthConnection,
  listAuthConnections,
  type AuthConnection,
} from "@/lib/api";

export default function AuthConnectionsPanel() {
  const [connections, setConnections] = useState<AuthConnection[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [label, setLabel] = useState("");
  const [type, setType] = useState("api_key_header");
  const [headerName, setHeaderName] = useState("X-Api-Key");
  const [apiKey, setApiKey] = useState("");
  const [bearerToken, setBearerToken] = useState("");
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    listAuthConnections()
      .then(setConnections)
      .catch((err: unknown) => setError(err instanceof Error ? err.message : "Failed to load"))
      .finally(() => setLoading(false));
  }, []);

  async function handleAdd(e: FormEvent) {
    e.preventDefault();
    if (!label.trim()) return;
    setSubmitting(true);
    setError("");
    const config_json: Record<string, unknown> =
      type === "bearer"
        ? { bearer_token: bearerToken }
        : type === "basic"
          ? { username: "", password: "" }
          : { header_name: headerName, api_key: apiKey };
    try {
      const created = await createAuthConnection(label.trim(), type, config_json);
      setConnections((c) => [...c, created]);
      setLabel("");
      setApiKey("");
      setBearerToken("");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to create connection");
    } finally {
      setSubmitting(false);
    }
  }

  async function handleDelete(id: string) {
    try {
      await deleteAuthConnection(id);
      setConnections((c) => c.filter((x) => x.id !== id));
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to delete");
    }
  }

  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">
        Reusable credentials for HTTP tools. Values are stored server-side and masked in the UI.
      </p>
      {error && (
        <div className="rounded-lg border border-destructive/20 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          {error}
        </div>
      )}
      <form onSubmit={handleAdd} className="grid gap-2">
        <input
          value={label}
          onChange={(e) => setLabel(e.target.value)}
          placeholder="Connection label"
          className="rounded-lg border border-border bg-background px-3 py-2 text-sm"
        />
        <select
          value={type}
          onChange={(e) => setType(e.target.value)}
          className="rounded-lg border border-border bg-background px-3 py-2 text-sm"
        >
          <option value="api_key_header">API key header</option>
          <option value="bearer">Bearer token</option>
        </select>
        {type === "api_key_header" && (
          <>
            <input
              value={headerName}
              onChange={(e) => setHeaderName(e.target.value)}
              placeholder="Header name"
              className="rounded-lg border border-border bg-background px-3 py-2 text-sm"
            />
            <input
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="API key or {{VAR}}"
              type="password"
              className="rounded-lg border border-border bg-background px-3 py-2 text-sm"
            />
          </>
        )}
        {type === "bearer" && (
          <input
            value={bearerToken}
            onChange={(e) => setBearerToken(e.target.value)}
            placeholder="Bearer token or {{VAR}}"
            type="password"
            className="rounded-lg border border-border bg-background px-3 py-2 text-sm"
          />
        )}
        <Button type="submit" size="sm" disabled={submitting || !label.trim()} className="w-fit gap-1">
          <Plus className="w-4 h-4" /> Create connection
        </Button>
      </form>
      {loading && <p className="text-xs text-muted-foreground">Loading...</p>}
      <ul className="space-y-2">
        {connections.map((c) => (
          <li
            key={c.id}
            className="flex items-center justify-between rounded-lg border border-border bg-card px-3 py-2 text-sm"
          >
            <span>
              <span className="font-medium">{c.label}</span>
              <span className="ml-2 text-muted-foreground">{c.type}</span>
            </span>
            <button
              type="button"
              onClick={() => void handleDelete(c.id)}
              className="rounded p-1 text-muted-foreground hover:text-destructive"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
