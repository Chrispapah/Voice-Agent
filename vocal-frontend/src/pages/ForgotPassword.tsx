import { FormEvent, useState } from "react";
import { Link } from "react-router-dom";
import { Mail } from "lucide-react";
import { Button } from "@/components/ui/button";
import { assertSupabaseConfigured, isSupabaseConfigured, supabase } from "@/lib/supabase";

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  async function handlePasswordReset(event: FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setMessage("");

    try {
      assertSupabaseConfigured();
      const { error: resetError } = await supabase.auth.resetPasswordForEmail(email, {
        redirectTo: `${window.location.origin}/reset-password`,
      });
      if (resetError) throw resetError;
      setMessage("Check your email for a password reset link.");
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Could not send password reset email");
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
            <h1 className="text-2xl font-semibold tracking-tight">Reset your password</h1>
            <p className="mt-1 text-sm text-muted-foreground">Enter your email and we will send you a reset link.</p>
          </div>
        </div>

        {!isSupabaseConfigured() && (
          <div className="mb-4 rounded-lg border border-warning/30 bg-warning/10 p-3 text-xs text-foreground">
            Add `VITE_SUPABASE_URL` and `VITE_SUPABASE_ANON_KEY` to your `.env.local` first.
          </div>
        )}

        {error && <div className="mb-4 rounded-lg border border-destructive/20 bg-destructive/10 p-3 text-sm text-destructive">{error}</div>}
        {message && <div className="mb-4 rounded-lg border border-success/20 bg-success/10 p-3 text-sm text-success">{message}</div>}

        <form onSubmit={handlePasswordReset} className="space-y-4">
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
          <Button className="w-full bg-gradient-primary text-primary-foreground hover:opacity-90" type="submit" disabled={loading}>
            <Mail className="mr-2 h-4 w-4" />
            {loading ? "Sending..." : "Send Reset Link"}
          </Button>
        </form>

        <Link to="/auth" className="mt-5 block text-center text-sm font-medium text-primary hover:underline">
          Back to sign in
        </Link>
      </section>
    </div>
  );
}
