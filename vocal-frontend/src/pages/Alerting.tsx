import { Bell, Plus, Edit2, MoreHorizontal, Activity } from "lucide-react";
import { Button } from "@/components/ui/button";

const alerts = [
  { name: "Payment Failure Rate Spike", desc: "Number of API requests returned error code · Payment Failed, Rate limit" },
  { name: "High Concurrency Spike", desc: "Number of Calls / Chats · Concurrency Exhausted" },
  { name: "LLM Retell Failure Surge", desc: "Number of custom function failures · Payment failed, PaymentProcess" },
  { name: "TTS Provider Error Rate High", desc: "Concurrency used count" },
];

export default function AlertingPage() {
  return (
    <div className="flex flex-col flex-1 min-h-0">
      <header className="flex items-center justify-between px-8 py-5 border-b border-border">
        <div className="flex items-center gap-2">
          <Bell className="w-4 h-4" />
          <h1 className="text-base font-semibold">Alerting</h1>
        </div>
        <Button size="sm" className="bg-foreground text-background hover:opacity-90 gap-1.5">
          <Plus className="w-4 h-4" /> Create Alert
        </Button>
      </header>
      <div className="flex-1 overflow-auto px-8 py-10">
        <div className="max-w-3xl mx-auto text-center mb-8">
          <h2 className="text-base font-semibold">Track key metrics and get notified when something goes wrong.</h2>
          <p className="text-sm text-muted-foreground mt-2">
            Configure alerts on key metrics. We'll evaluate them on a set cadence and notify you via email or webhook when thresholds are met or values change significantly.
          </p>
          <div className="mt-4 flex justify-center gap-2">
            <Button size="sm" className="bg-foreground text-background gap-1.5"><Plus className="w-4 h-4" /> Create Alert</Button>
            <Button size="sm" variant="outline">Read the Docs</Button>
          </div>
        </div>
        <div className="max-w-3xl mx-auto rounded-xl border border-border bg-card shadow-soft">
          <div className="flex items-center gap-6 px-5 pt-4 border-b border-border">
            <button className="text-sm font-semibold border-b-2 border-primary pb-3">Alerting</button>
            <button className="text-sm text-muted-foreground pb-3">Alert history</button>
          </div>
          <ul>
            {alerts.map((a) => (
              <li key={a.name} className="flex items-center gap-3 px-5 py-3.5 border-t border-border first:border-t-0">
                <div className="w-9 h-9 rounded-lg bg-gradient-soft flex items-center justify-center">
                  <Activity className="w-4 h-4 text-primary" />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium">{a.name}</div>
                  <div className="text-xs text-muted-foreground truncate">{a.desc}</div>
                </div>
                <Button size="sm" variant="outline" className="gap-1.5"><Edit2 className="w-3.5 h-3.5" /> Edit</Button>
                <button className="p-1.5 rounded-md hover:bg-secondary"><MoreHorizontal className="w-4 h-4" /></button>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}