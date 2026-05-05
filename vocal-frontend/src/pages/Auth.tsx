import { FormEvent, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Bot, KeyRound, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { clearLocalBearerToken, saveLocalBearerToken } from "@/lib/api";
import { assertSupabaseConfigured, isSupabaseConfigured, supabase } from "@/lib/supabase";

type AuthMode = "sign-in" | "sign-up";

export default function AuthPage() {
  const navigate = useNavigate();
  const [mode, setMode] = useState<AuthMode>("sign-in");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [token, setToken] = useState("");
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  async function handleSupabaseAuth(event: FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setMessage("");
    clearLocalBearerToken();
    try {
      assertSupabaseConfigured();
      if (mode === "sign-in") {
        const { error: authError } = await supabase.auth.signInWithPassword({ email, password });
        if (authError) throw authError;
        navigate("/agents");
        return;
      }

      const { data, error: authError } = await supabase.auth.signUp({
        email,
        password,
        options: { data: { display_name: displayName } },
      });
      if (authError) throw authError;
      if (data.session) {
        navigate("/agents");
      } else {
        setMessage("Account created. Check your email to confirm, then sign in.");
        setMode("sign-in");
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Authentication failed");
    } finally {
      setLoading(false);
    }
  }

  function handleTokenSubmit(event: FormEvent) {
    event.preventDefault();
    if (!token.trim()) {
      setError("Paste a bearer token first.");
      return;
    }
    saveLocalBearerToken(token.trim());
    navigate("/agents");
  }

  return (
    <div className="min-h-screen bg-surface-muted/50 px-4 py-10 text-foreground">
      <div className="mx-auto grid max-w-5xl gap-6 md:grid-cols-[1fr_420px]">
        <section className="rounded-2xl border border-border bg-card p-8 shadow-soft">
          <div className="mb-8 flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-primary shadow-elegant">
              <Sparkles className="h-5 w-5 text-primary-foreground" />
            </div>
            <div>
              <h1 className="text-2xl font-semibold tracking-tight">Welcome to Vocal</h1>
              <p className="text-sm text-muted-foreground">Sign in to manage agents and publish flows to your LangChain brain.</p>
            </div>
          </div>

          <div className="grid gap-4 sm:grid-cols-3">
            {[
              ["Build", "Design graph or single-prompt voice agents."],
              ["Publish", "Save specs to the backend bot config."],
              ["Test", "Chat with the same runtime used by the API."],
            ].map(([title, body]) => (
              <div key={title} className="rounded-xl border border-border bg-surface p-4">
                <Bot className="mb-3 h-5 w-5 text-primary" />
                <h2 className="text-sm font-semibold">{title}</h2>
                <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{body}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="rounded-2xl border border-border bg-card p-6 shadow-soft">
          <div className="mb-5 grid grid-cols-2 rounded-lg bg-secondary p-1">
            <button
              type="button"
              onClick={() => setMode("sign-in")}
              className={`rounded-md py-2 text-sm font-medium ${mode === "sign-in" ? "bg-card shadow-sm" : "text-muted-foreground"}`}
            >
              Sign In
            </button>
            <button
              type="button"
              onClick={() => setMode("sign-up")}
              className={`rounded-md py-2 text-sm font-medium ${mode === "sign-up" ? "bg-card shadow-sm" : "text-muted-foreground"}`}
            >
              Sign Up
            </button>
          </div>

          {!isSupabaseConfigured() && (
            <div className="mb-4 rounded-lg border border-warning/30 bg-warning/10 p-3 text-xs text-foreground">
              Add `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY` to `vocal-frontend/.env.local`, or use a bearer token below.
            </div>
          )}

          {error && <div className="mb-4 rounded-lg border border-destructive/20 bg-destructive/10 p-3 text-sm text-destructive">{error}</div>}
          {message && <div className="mb-4 rounded-lg border border-success/20 bg-success/10 p-3 text-sm text-success">{message}</div>}

          <form onSubmit={handleSupabaseAuth} className="space-y-4">
            {mode === "sign-up" && (
              <div>
                <label className="mb-1 block text-xs font-semibold text-muted-foreground">Display name</label>
                <input
                  value={displayName}
                  onChange={(event) => setDisplayName(event.target.value)}
                  className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-ring/40"
                  placeholder="Chris"
                />
              </div>
            )}
            <div>
              <label className="mb-1 block text-xs font-semibold text-muted-foreground">Email</label>
              <input
                type="email"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-ring/40"
                placeholder="you@example.com"
                required
              />
            </div>
            <div>
              <label className="mb-1 block text-xs font-semibold text-muted-foreground">Password</label>
              <input
                type="password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-ring/40"
                placeholder="At least 6 characters"
                required
              />
            </div>
            <Button className="w-full bg-gradient-primary text-primary-foreground hover:opacity-90" type="submit" disabled={loading}>
              {loading ? "Working..." : mode === "sign-in" ? "Sign In" : "Create Account"}
            </Button>
          </form>

          <div className="my-5 flex items-center gap-3 text-xs text-muted-foreground">
            <div className="h-px flex-1 bg-border" />
            local dev
            <div className="h-px flex-1 bg-border" />
          </div>

          <form onSubmit={handleTokenSubmit} className="space-y-3">
            <label className="flex items-center gap-1.5 text-xs font-semibold text-muted-foreground">
              <KeyRound className="h-3.5 w-3.5" /> Bearer token
            </label>
            <textarea
              rows={3}
              value={token}
              onChange={(event) => setToken(event.target.value)}
              className="w-full rounded-lg border border-border bg-background px-3 py-2 text-xs outline-none focus:ring-2 focus:ring-ring/40"
              placeholder="Paste Supabase access token for local testing"
            />
            <Button variant="outline" className="w-full" type="submit">
              Continue With Token
            </Button>
          </form>
        </section>
      </div>
    </div>
  );
}
