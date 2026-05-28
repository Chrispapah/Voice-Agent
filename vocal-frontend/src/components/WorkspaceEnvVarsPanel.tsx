import { FormEvent, useEffect, useState } from "react";
import { Plus, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  AuthRequiredError,
  createWorkspaceEnvVar,
  deleteWorkspaceEnvVar,
  listWorkspaceEnvVars,
  type WorkspaceEnvVar,
} from "@/lib/api";

export default function WorkspaceEnvVarsPanel({ onClose }: { onClose?: () => void }) {
  const [vars, setVars] = useState<WorkspaceEnvVar[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [name, setName] = useState("");
  const [value, setValue] = useState("");
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    listWorkspaceEnvVars()
      .then(setVars)
      .catch((err: unknown) => {
        if (err instanceof AuthRequiredError) return;
        setError(err instanceof Error ? err.message : "Failed to load variables");
      })
      .finally(() => setLoading(false));
  }, []);

  async function handleAdd(e: FormEvent) {
    e.preventDefault();
    if (!name.trim()) return;
    setSubmitting(true);
    setError("");
    try {
      const created = await createWorkspaceEnvVar(name.trim(), value);
      setVars((current) => [...current, created].sort((a, b) => a.name.localeCompare(b.name)));
      setName("");
      setValue("");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to create variable");
    } finally {
      setSubmitting(false);
    }
  }

  async function handleDelete(id: string) {
    setError("");
    try {
      await deleteWorkspaceEnvVar(id);
      setVars((current) => current.filter((v) => v.id !== id));
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to delete variable");
    }
  }

  return (
    <div className="space-y-4">
      <p className="text-sm text-muted-foreground">
        Use <code className="rounded bg-muted px-1 text-xs">{`{{VAR_NAME}}`}</code> in tool URLs and headers.
      </p>
      {error && (
        <div className="rounded-lg border border-destructive/20 bg-destructive/10 px-3 py-2 text-sm text-destructive">
          {error}
        </div>
      )}
      <form onSubmit={handleAdd} className="grid gap-2 sm:grid-cols-[1fr_1fr_auto]">
        <input
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="VAR_NAME"
          className="rounded-lg border border-border bg-background px-3 py-2 text-sm"
        />
        <input
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Secret value"
          type="password"
          className="rounded-lg border border-border bg-background px-3 py-2 text-sm"
        />
        <Button type="submit" size="sm" disabled={submitting || !name.trim()} className="gap-1">
          <Plus className="w-4 h-4" /> Add
        </Button>
      </form>
      {loading && <p className="text-xs text-muted-foreground">Loading...</p>}
      {!loading && vars.length === 0 && (
        <p className="text-xs text-muted-foreground">No environment variables yet.</p>
      )}
      <ul className="space-y-2">
        {vars.map((v) => (
          <li
            key={v.id}
            className="flex items-center justify-between gap-2 rounded-lg border border-border bg-card px-3 py-2 text-sm"
          >
            <span>
              <span className="font-mono font-medium">{v.name}</span>
              <span className="ml-2 text-muted-foreground">{v.value_masked}</span>
            </span>
            <button
              type="button"
              onClick={() => void handleDelete(v.id)}
              className="rounded p-1 text-muted-foreground hover:bg-destructive/10 hover:text-destructive"
              aria-label={`Delete ${v.name}`}
            >
              <Trash2 className="w-4 h-4" />
            </button>
          </li>
        ))}
      </ul>
      {onClose && (
        <Button type="button" variant="outline" size="sm" onClick={onClose}>
          Close
        </Button>
      )}
    </div>
  );
}
