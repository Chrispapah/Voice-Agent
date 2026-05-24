import { FormEvent, useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { LockKeyhole } from "lucide-react";
import { Button } from "@/components/ui/button";
import { assertSupabaseConfigured, isSupabaseConfigured, supabase } from "@/lib/supabase";

export default function ResetPasswordPage() {
  const navigate = useNavigate();
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [checkingSession, setCheckingSession] = useState(true);
  const [hasResetSession, setHasResetSession] = useState(false);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    let mounted = true;

    async function checkRecoverySession() {
      if (!isSupabaseConfigured()) {
        if (mounted) setCheckingSession(false);
        return;
      }

      const {
        data: { session },
      } = await supabase.auth.getSession();

      if (!mounted) return;
      setHasResetSession(Boolean(session));
      setCheckingSession(false);
    }

    void checkRecoverySession();

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((event, session) => {
      if (event === "PASSWORD_RECOVERY" || session) {
        setHasResetSession(Boolean(session));
        setCheckingSession(false);
      }
    });

    return () => {
      mounted = false;
      subscription.unsubscribe();
    };
  }, []);

  async function handlePasswordUpdate(event: FormEvent) {
    event.preventDefault();
    setError("");
    setMessage("");

    if (password.length < 6) {
      setError("Password must be at least 6 characters.");
      return;
    }

    if (password !== confirmPassword) {
      setError("Passwords do not match.");
      return;
    }

    setLoading(true);
    try {
      assertSupabaseConfigured();
      const { error: updateError } = await supabase.auth.updateUser({ password });
      if (updateError) throw updateError;
      setMessage("Password updated. Redirecting to sign in...");
      await supabase.auth.signOut();
      window.setTimeout(() => navigate("/auth", { replace: true }), 1200);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Could not update password");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-surface-muted/50 px-4 py-10 text-foreground">
      <section className="mx-auto max-w-md rounded-2xl border border-border bg-card p-6 shadow-soft">
        <div className="mb-6 flex flex-col items-center gap-4 text-center">
          <img src="/akoi-logo-no-back.png" alt="Akoi" className="h-[96px] w-auto max-w-full object-contain" />
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">Choose a new password</h1>
            <p className="mt-1 text-sm text-muted-foreground">Create a new password for your Akoi account.</p>
          </div>
        </div>

        {!isSupabaseConfigured() && (
          <div className="mb-4 rounded-lg border border-warning/30 bg-warning/10 p-3 text-xs text-foreground">
            Add `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY` to your `.env.local` first.
          </div>
        )}

        {checkingSession && <div className="mb-4 rounded-lg border border-border bg-surface p-3 text-sm text-muted-foreground">Checking reset link...</div>}
        {!checkingSession && !hasResetSession && (
          <div className="mb-4 rounded-lg border border-destructive/20 bg-destructive/10 p-3 text-sm text-destructive">
            This reset link is invalid or expired. Request a new password reset email.
          </div>
        )}
        {error && <div className="mb-4 rounded-lg border border-destructive/20 bg-destructive/10 p-3 text-sm text-destructive">{error}</div>}
        {message && <div className="mb-4 rounded-lg border border-success/20 bg-success/10 p-3 text-sm text-success">{message}</div>}

        <form onSubmit={handlePasswordUpdate} className="space-y-4">
          <div>
            <label className="mb-1 block text-xs font-semibold text-muted-foreground">New password</label>
            <input
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-ring/40"
              placeholder="At least 6 characters"
              required
            />
          </div>
          <div>
            <label className="mb-1 block text-xs font-semibold text-muted-foreground">Confirm password</label>
            <input
              type="password"
              value={confirmPassword}
              onChange={(event) => setConfirmPassword(event.target.value)}
              className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-ring/40"
              placeholder="Repeat your new password"
              required
            />
          </div>
          <Button className="w-full bg-gradient-primary text-primary-foreground hover:opacity-90" type="submit" disabled={loading || checkingSession || !hasResetSession}>
            <LockKeyhole className="mr-2 h-4 w-4" />
            {loading ? "Updating..." : "Update Password"}
          </Button>
        </form>

        <Link to="/forgot-password" className="mt-5 block text-center text-sm font-medium text-primary hover:underline">
          Request a new reset link
        </Link>
      </section>
    </div>
  );
}
