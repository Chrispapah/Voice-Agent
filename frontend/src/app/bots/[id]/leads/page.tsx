"use client";

import { FormEvent, useEffect, useState, use } from "react";
import Link from "next/link";
import { useAuth } from "@/lib/auth";
import { Lead, createLead, getBot, listLeads } from "@/lib/api";
import { ArrowLeft, Plus, User } from "lucide-react";

export default function LeadsPage({ params }: { params: Promise<{ id: string }> }) {
  const { id: botId } = use(params);
  const { user, loading: authLoading } = useAuth();
  const [botName, setBotName] = useState("");
  const [leads, setLeads] = useState<Lead[]>([]);
  const [showForm, setShowForm] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [form, setForm] = useState({
    lead_name: "",
    company: "",
    phone_number: "",
    lead_email: "",
    lead_context: "",
    owner_name: "Sales Team",
    calendar_id: "sales-team",
  });

  useEffect(() => {
    if (authLoading || !user) return;
    Promise.all([getBot(botId), listLeads(botId)])
      .then(([b, loadedLeads]) => {
        setBotName(b.name);
        setLeads(loadedLeads);
      })
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : "Failed to load leads");
      })
      .finally(() => setLoading(false));
  }, [botId, user, authLoading]);

  async function handleCreate(e: FormEvent) {
    e.preventDefault();
    setError("");
    try {
      const lead = await createLead(botId, form);
      setLeads((prev) => [lead, ...prev]);
      setShowForm(false);
      setForm({
        lead_name: "",
        company: "",
        phone_number: "",
        lead_email: "",
        lead_context: "",
        owner_name: "Sales Team",
        calendar_id: "sales-team",
      });
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to create lead");
    }
  }

  return (
    <div className="mx-auto max-w-4xl px-4 py-8">
      <div className="mb-6 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Link href={`/bots/${botId}/settings`} className="rounded p-1 hover:bg-[var(--secondary)]">
            <ArrowLeft size={20} />
          </Link>
          <h1 className="text-xl font-bold">Leads - {botName}</h1>
        </div>
        <button
          onClick={() => setShowForm(!showForm)}
          className="flex items-center gap-2 rounded-md bg-[var(--primary)] px-4 py-2 text-sm font-medium text-[var(--primary-foreground)] hover:opacity-90"
        >
          <Plus size={16} />
          Add Lead
        </button>
      </div>

      {error && (
        <div className="mb-6 rounded-md bg-red-50 p-3 text-sm text-red-700 dark:bg-red-900/30 dark:text-red-300">
          {error}
        </div>
      )}

      {authLoading || loading ? (
        <div className="rounded-lg border border-[var(--border)] p-8 text-center text-[var(--muted-foreground)]">
          Loading...
        </div>
      ) : null}

      {!loading && showForm && (
        <form
          onSubmit={handleCreate}
          className="mb-6 space-y-3 rounded-lg border border-[var(--border)] bg-[var(--card)] p-5"
        >
          <div className="grid gap-3 sm:grid-cols-2">
            <div>
              <label className="mb-1 block text-sm font-medium">Name *</label>
              <input
                required
                value={form.lead_name}
                onChange={(e) => setForm({ ...form, lead_name: e.target.value })}
                className="w-full rounded-md border border-[var(--border)] bg-[var(--background)] px-3 py-2 text-sm"
              />
            </div>
            <div>
              <label className="mb-1 block text-sm font-medium">Company</label>
              <input
                value={form.company}
                onChange={(e) => setForm({ ...form, company: e.target.value })}
                className="w-full rounded-md border border-[var(--border)] bg-[var(--background)] px-3 py-2 text-sm"
              />
            </div>
            <div>
              <label className="mb-1 block text-sm font-medium">Phone *</label>
              <input
                required
                value={form.phone_number}
                onChange={(e) => setForm({ ...form, phone_number: e.target.value })}
                className="w-full rounded-md border border-[var(--border)] bg-[var(--background)] px-3 py-2 text-sm"
                placeholder="+1234567890"
              />
            </div>
            <div>
              <label className="mb-1 block text-sm font-medium">Email</label>
              <input
                type="email"
                value={form.lead_email}
                onChange={(e) => setForm({ ...form, lead_email: e.target.value })}
                className="w-full rounded-md border border-[var(--border)] bg-[var(--background)] px-3 py-2 text-sm"
              />
            </div>
          </div>
          <div>
            <label className="mb-1 block text-sm font-medium">Context / Notes</label>
            <textarea
              rows={2}
              value={form.lead_context}
              onChange={(e) => setForm({ ...form, lead_context: e.target.value })}
              className="w-full rounded-md border border-[var(--border)] bg-[var(--background)] px-3 py-2 text-sm"
              placeholder="Background info the bot should know about this lead..."
            />
          </div>
          <div className="flex gap-2">
            <button
              type="submit"
              className="rounded-md bg-[var(--primary)] px-4 py-2 text-sm font-medium text-[var(--primary-foreground)] hover:opacity-90"
            >
              Create Lead
            </button>
            <button
              type="button"
              onClick={() => setShowForm(false)}
              className="rounded-md border border-[var(--border)] px-4 py-2 text-sm hover:bg-[var(--secondary)]"
            >
              Cancel
            </button>
          </div>
        </form>
      )}

      {!loading && leads.length === 0 ? (
        <div className="rounded-lg border border-dashed border-[var(--border)] p-12 text-center">
          <User size={48} className="mx-auto mb-4 text-[var(--muted-foreground)]" />
          <p className="text-[var(--muted-foreground)]">No leads yet. Add your first lead above.</p>
        </div>
      ) : !loading ? (
        <div className="overflow-hidden rounded-lg border border-[var(--border)]">
          <table className="w-full text-sm">
            <thead className="bg-[var(--secondary)]">
              <tr>
                <th className="px-4 py-3 text-left font-medium">Name</th>
                <th className="px-4 py-3 text-left font-medium">Company</th>
                <th className="px-4 py-3 text-left font-medium">Phone</th>
                <th className="px-4 py-3 text-left font-medium">Email</th>
                <th className="px-4 py-3 text-left font-medium">Stage</th>
              </tr>
            </thead>
            <tbody>
              {leads.map((lead) => (
                <tr key={lead.id} className="border-t border-[var(--border)] hover:bg-[var(--secondary)]">
                  <td className="px-4 py-3 font-medium">{lead.lead_name}</td>
                  <td className="px-4 py-3 text-[var(--muted-foreground)]">{lead.company}</td>
                  <td className="px-4 py-3 text-[var(--muted-foreground)]">{lead.phone_number}</td>
                  <td className="px-4 py-3 text-[var(--muted-foreground)]">{lead.lead_email}</td>
                  <td className="px-4 py-3">
                    <span className="rounded-full bg-[var(--secondary)] px-2 py-0.5 text-xs font-medium">
                      {lead.lifecycle_stage}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </div>
  );
}
