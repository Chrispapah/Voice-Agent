import { useState } from "react";
import { Settings as SettingsIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { toast } from "@/components/ui/sonner";

const sections = [
  { title: "Profile", desc: "Update your name, avatar, and contact email." },
  { title: "Workspace", desc: "Manage workspace name, members, and roles." },
  { title: "API Keys", desc: "Create and revoke API keys used by your applications." },
  { title: "Webhooks", desc: "Receive real-time call and chat events on your endpoints." },
  { title: "Integrations", desc: "Connect Twilio, Telnyx, Zapier, Slack and more." },
  { title: "Security", desc: "Two-factor authentication, SSO, audit logs." },
];

export default function SettingsPage() {
  const [apiKeysOpen, setApiKeysOpen] = useState(false);

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
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="mt-4"
                onClick={() => {
                  if (s.title === "API Keys") {
                    setApiKeysOpen(true);
                  } else {
                    toast.info("Coming soon", { description: `${s.title} is not available in this build yet.` });
                  }
                }}
              >
                Manage
              </Button>
            </div>
          ))}
        </div>
      </div>

      <Dialog open={apiKeysOpen} onOpenChange={setApiKeysOpen}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>API keys</DialogTitle>
            <DialogDescription asChild>
              <div className="space-y-3 text-left text-sm text-muted-foreground">
                <p>
                  This UI does not store provider keys. Configure them in your deployment environment (see{" "}
                  <code className="rounded bg-muted px-1 py-0.5 text-xs">.env.example</code> in the repo) or in the
                  admin app under each bot&apos;s settings (API Keys tab).
                </p>
                <p className="font-medium text-foreground">Typical variables</p>
                <ul className="list-inside list-disc space-y-1 text-xs">
                  <li>
                    LLM: <code className="rounded bg-muted px-1">OPENAI_API_KEY</code>,{" "}
                    <code className="rounded bg-muted px-1">ANTHROPIC_API_KEY</code>,{" "}
                    <code className="rounded bg-muted px-1">GROQ_API_KEY</code>
                  </li>
                  <li>
                    Voice: <code className="rounded bg-muted px-1">ELEVENLABS_API_KEY</code>,{" "}
                    <code className="rounded bg-muted px-1">DEEPGRAM_API_KEY</code>
                  </li>
                </ul>
              </div>
            </DialogDescription>
          </DialogHeader>
        </DialogContent>
      </Dialog>
    </div>
  );
}