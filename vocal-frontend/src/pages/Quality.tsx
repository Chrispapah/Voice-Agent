import { ShieldCheck, Plus, Settings as SettingsIcon, MoreHorizontal, Calendar } from "lucide-react";
import { Button } from "@/components/ui/button";

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-border p-4">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className="mt-1 text-2xl font-semibold">{value}</div>
      <svg viewBox="0 0 200 50" className="mt-2 w-full h-12 text-primary/60">
        <path d="M0 30 Q 25 10, 50 25 T 100 25 T 150 25 T 200 20" fill="none" stroke="currentColor" strokeWidth="2" />
        <path d="M0 30 Q 25 10, 50 25 T 100 25 T 150 25 T 200 20 L 200 50 L 0 50 Z" fill="currentColor" opacity="0.1" />
      </svg>
    </div>
  );
}

export default function QualityPage() {
  return (
    <div className="flex flex-col flex-1 min-h-0">
      <header className="flex items-center justify-between px-8 py-5 border-b border-border">
        <div className="flex items-center gap-2">
          <ShieldCheck className="w-4 h-4" />
          <h1 className="text-base font-semibold">AI Quality Assurance</h1>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm">Give feedback</Button>
          <Button size="sm" className="bg-foreground text-background gap-1.5"><Plus className="w-4 h-4" /> Create QA</Button>
        </div>
      </header>
      <div className="flex-1 overflow-auto px-8 py-8 space-y-6">
        <div className="max-w-3xl mx-auto text-center">
          <p className="text-sm font-semibold">Analyze AI calls with structured insights across audio, language, and performance.</p>
          <p className="text-xs text-muted-foreground mt-2">
            AI Quality Assurance helps you evaluate calls across key dimensions — from audio quality (overlapping speech, tone, WER), to agent hallucinations, resolution accuracy, and user sentiment.
          </p>
          <div className="mt-3 inline-flex items-center gap-1.5 text-xs text-primary bg-accent/60 px-3 py-1.5 rounded-md">
            ⓘ First 100 minutes of analysis is free
          </div>
        </div>
        <div className="max-w-5xl mx-auto rounded-xl border border-border bg-card shadow-soft p-5 space-y-5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <button className="px-3 py-1.5 rounded-md bg-secondary text-sm font-medium">Call QA Overview</button>
              <button className="px-3 py-1.5 rounded-md text-sm text-muted-foreground">Detailed Calls</button>
            </div>
            <div className="flex items-center gap-2">
              <Button variant="outline" size="sm" className="gap-1.5"><Calendar className="w-4 h-4" /> Date Range</Button>
              <Button variant="outline" size="sm" className="gap-1.5"><SettingsIcon className="w-4 h-4" /> Configure QA Settings</Button>
              <button className="p-1.5 rounded-md hover:bg-secondary"><MoreHorizontal className="w-4 h-4" /></button>
            </div>
          </div>
          <div className="rounded-lg border border-border p-5">
            <div className="flex items-center justify-between text-sm">
              <span className="font-medium">Calls Analysed</span>
              <span className="text-xs text-muted-foreground">09/01/2025 – 10/01/2025</span>
            </div>
            <div className="mt-2 text-2xl font-semibold">Completed: 68 · Total: 240</div>
            <div className="mt-3 h-1.5 rounded-full bg-secondary overflow-hidden">
              <div className="h-full w-[28%] bg-success" />
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <Stat label="Average Score" value="87.00" />
            <Stat label="Call Resolution Rate" value="75.00%" />
            <div className="rounded-lg border border-border p-4">
              <div className="text-xs text-muted-foreground">Transfer Success Rate</div>
              <div className="mt-1 flex items-center gap-3 text-sm flex-wrap">
                <span className="text-2xl font-semibold">70%</span>
                <span className="px-2 py-0.5 rounded bg-secondary text-xs">Total: 100</span>
                <span className="px-2 py-0.5 rounded bg-success/15 text-success text-xs">Success: 70</span>
                <span className="px-2 py-0.5 rounded bg-destructive/15 text-destructive text-xs">Failure: 30</span>
              </div>
            </div>
            <Stat label="Transfer Wait Time" value="4.0s" />
          </div>
          <div className="rounded-lg border border-border">
            <div className="flex items-center justify-between px-4 py-3 border-b border-border">
              <span className="text-sm font-medium">Top Questions from User</span>
              <button className="text-xs text-primary hover:underline">View All</button>
            </div>
            <table className="w-full text-sm">
              <thead className="text-muted-foreground">
                <tr className="text-left">
                  <th className="px-4 py-2 font-medium">No.</th>
                  <th className="px-4 py-2 font-medium">Question</th>
                  <th className="px-4 py-2 font-medium text-right">Resolution Rate</th>
                  <th className="px-4 py-2 font-medium text-right">Resolved / Total</th>
                </tr>
              </thead>
              <tbody>
                {[1,2,3,4,5].map((n) => (
                  <tr key={n} className="border-t border-border">
                    <td className="px-4 py-2.5 text-muted-foreground">{n}</td>
                    <td className="px-4 py-2.5"><div className="h-2 w-64 rounded bg-surface-muted" /></td>
                    <td className="px-4 py-2.5 text-right">67%</td>
                    <td className="px-4 py-2.5 text-right text-muted-foreground">12 / 18</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}