import { Link } from "react-router-dom";
import { PhoneCall, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function BatchCallPage() {
  return (
    <div className="flex flex-col flex-1 min-h-0">
      <header className="flex items-center justify-between px-8 py-5 border-b border-border">
        <div className="flex items-center gap-2">
          <PhoneCall className="w-4 h-4" />
          <h1 className="text-base font-semibold">Batch Call</h1>
        </div>
        <Button asChild size="sm" className="bg-foreground text-background hover:opacity-90 gap-1.5">
          <Link to="/batch-call/new"><Plus className="w-4 h-4" /> Create a batch call</Link>
        </Button>
      </header>
      <div className="flex-1 grid place-items-center">
        <div className="text-center">
          <div className="w-12 h-12 mx-auto rounded-xl bg-surface-muted border border-border flex items-center justify-center">
            <PhoneCall className="w-5 h-5 text-muted-foreground" />
          </div>
          <p className="mt-4 text-sm text-muted-foreground">You don't have any batch call</p>
        </div>
      </div>
    </div>
  );
}