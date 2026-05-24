import { FormEvent, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Bot } from "lucide-react";
import { Button } from "@/components/ui/button";
import { assertSupabaseConfigured, isSupabaseConfigured, supabase } from "@/lib/supabase";

type AuthMode = "sign-in" | "sign-up";

export default function AuthPage() {
  const navigate = useNavigate();
  const [mode, setMode] = useState<AuthMode>("sign-in");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  async function handleSupabaseAuth(event: FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setMessage("");
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

  return (
    <div className="min-h-screen bg-surface-muted/50 px-4 py-10 text-foreground">
      <div className="mx-auto grid max-w-5xl gap-6 md:grid-cols-[1fr_420px]">
        <section className="rounded-2xl border border-border bg-card p-8 shadow-soft">
          <div className="mb-8 flex flex-col items-center gap-6 text-center">
            <img
              src="/akoi-logo-no-back.png"
              alt="Akoi"
              className="h-[120px] w-auto max-w-[min(100%,600px)] object-contain"
            />
            <div>
              <h1 className="text-2xl font-semibold tracking-tight">Welcome to Akoi</h1>
              <p className="text-sm text-muted-foreground">Sign in to manage agents and publish flows.</p>
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
              Add `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY` to your `.env.local` to enable authentication.
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
              <div className="mb-1 flex items-center justify-between gap-3">
                <label className="block text-xs font-semibold text-muted-foreground">Password</label>
                {mode === "sign-in" && (
                  <Link to="/forgot-password" className="text-xs font-medium text-primary hover:underline">
                    Forgot password?
                  </Link>
                )}
              </div>
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
        </section>
      </div>
    </div>
  );
}
