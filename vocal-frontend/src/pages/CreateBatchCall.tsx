import { Link } from "react-router-dom";
import { ChevronLeft, UploadCloud, Info, Minus, Plus, Download, Clock } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function CreateBatchCallPage() {
  return (
    <div className="flex flex-1 min-h-0">
      <div className="w-[420px] border-r border-border flex flex-col">
        <header className="px-6 py-5 border-b border-border">
          <div className="flex items-center gap-3">
            <Button asChild variant="outline" size="icon" className="h-8 w-8">
              <Link to="/batch-call"><ChevronLeft className="w-4 h-4" /></Link>
            </Button>
            <div>
              <h1 className="text-base font-semibold">Create a batch call</h1>
              <p className="text-xs text-muted-foreground flex items-center gap-1 mt-0.5">
                <Info className="w-3 h-3" /> Batch call cost $0.005 per dial
              </p>
            </div>
          </div>
        </header>
        <div className="flex-1 overflow-auto p-6 space-y-5">
          <Field label="Batch Call Name">
            <input className="w-full h-9 rounded-md border border-border bg-card px-3 text-sm" placeholder="Enter" />
          </Field>
          <Field label="From number">
            <button className="w-full h-9 rounded-md border border-border bg-card px-3 text-sm text-left text-muted-foreground">Select a number</button>
          </Field>
          <Field label="Upload Recipients">
            <button className="text-xs flex items-center gap-1 text-muted-foreground mb-2 hover:text-foreground">
              <Download className="w-3.5 h-3.5" /> Download the template
            </button>
            <div className="rounded-lg border border-dashed border-border bg-surface-muted/40 py-8 px-4 text-center">
              <UploadCloud className="w-6 h-6 mx-auto text-muted-foreground" />
              <p className="mt-2 text-sm font-medium">Choose a csv or drag &amp; drop it here.</p>
              <p className="text-xs text-muted-foreground mt-0.5">Up to 50 MB</p>
            </div>
          </Field>
          <Field label="When to send the calls">
            <div className="grid grid-cols-2 gap-2">
              <label className="flex items-center justify-between gap-2 px-3 h-9 rounded-md border-2 border-primary bg-accent/40 text-sm font-medium cursor-pointer">
                Send Now <span className="w-3 h-3 rounded-full border-2 border-primary bg-primary" />
              </label>
              <label className="flex items-center justify-between gap-2 px-3 h-9 rounded-md border border-border text-sm cursor-pointer">
                Schedule <span className="w-3 h-3 rounded-full border border-muted-foreground" />
              </label>
            </div>
          </Field>
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">When Calls Can Run</span>
            <button className="text-xs text-muted-foreground flex items-center gap-1 hover:text-foreground">
              <Clock className="w-3.5 h-3.5" /> 00:00–23:59, Mon–Sun
            </button>
          </div>
          <Field label="Reserved Concurrency for Other Calls" hint="Number of concurrency reserved for all other calls, such as inbound calls.">
            <div className="flex items-center rounded-md border border-border h-9 overflow-hidden">
              <button className="w-9 h-full grid place-items-center hover:bg-secondary"><Minus className="w-3.5 h-3.5" /></button>
              <input defaultValue={5} className="flex-1 h-full text-center text-sm bg-transparent focus:outline-none" />
              <button className="w-9 h-full grid place-items-center hover:bg-secondary"><Plus className="w-3.5 h-3.5" /></button>
            </div>
          </Field>
          <div className="rounded-md bg-accent/50 border border-accent text-accent-foreground px-3 py-2 text-xs flex items-start gap-2">
            <Info className="w-3.5 h-3.5 mt-0.5" />
            <div>
              Concurrency allocated to batch calling: 15
              <button className="block mt-1 text-primary hover:underline">Purchase more concurrency ↗</button>
            </div>
          </div>
          <p className="text-xs text-muted-foreground">You've read and agree with the Terms of service.</p>
          <div className="flex justify-end gap-2">
            <Button variant="outline" size="sm">Save as draft</Button>
            <Button size="sm" disabled>Send</Button>
          </div>
        </div>
      </div>
      <div className="flex-1 bg-surface-muted/40 flex flex-col">
        <div className="px-8 py-5 border-b border-border">
          <h2 className="text-sm font-semibold">Recipients</h2>
        </div>
        <div className="flex-1 grid place-items-center text-sm text-muted-foreground">
          Please upload recipients first
        </div>
      </div>
    </div>
  );
}

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="text-sm font-medium mb-1.5">{label}</div>
      {hint && <p className="text-xs text-muted-foreground mb-2">{hint}</p>}
      {children}
    </div>
  );
}