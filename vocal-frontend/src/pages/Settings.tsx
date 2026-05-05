import { Settings as SettingsIcon } from "lucide-react";
import { Button } from "@/components/ui/button";

const sections = [
  { title: "Profile", desc: "Update your name, avatar, and contact email." },
  { title: "Workspace", desc: "Manage workspace name, members, and roles." },
  { title: "API Keys", desc: "Create and revoke API keys used by your applications." },
  { title: "Webhooks", desc: "Receive real-time call and chat events on your endpoints." },
  { title: "Integrations", desc: "Connect Twilio, Telnyx, Zapier, Slack and more." },
  { title: "Security", desc: "Two-factor authentication, SSO, audit logs." },
];

export default function SettingsPage() {
  return (
    <div className="flex flex-col flex-1 min-h-0">
      <header className="flex items-center gap-2 px-8 py-5 border-b border-border">
        <SettingsIcon className="w-4 h-4" />
        <h1 className="text-base font-semibold">Settings</h1>
      </header>
      <div className="flex-1 overflow-auto px-8 py-8">
        <div className="max-w-4xl grid grid-cols-1 md:grid-cols-2 gap-4">
          {sections.map((s) => (
            <div key={s.title} className="rounded-xl border border-border bg-card p-5 hover:shadow-soft transition">
              <div className="text-sm font-semibold">{s.title}</div>
              <p className="text-xs text-muted-foreground mt-1">{s.desc}</p>
              <Button variant="outline" size="sm" className="mt-4">Manage</Button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}