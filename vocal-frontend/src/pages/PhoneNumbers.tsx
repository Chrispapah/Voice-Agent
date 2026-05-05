import { Phone, Plus, Search } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function PhoneNumbersPage() {
  return (
    <div className="flex flex-1 min-h-0">
      <aside className="hidden lg:flex flex-col w-72 border-r border-border bg-surface-muted/40">
        <div className="flex items-center justify-between px-4 py-4 border-b border-border">
          <div className="flex items-center gap-2 text-sm font-medium">
            <Phone className="w-4 h-4" /> Phone Numbers
          </div>
          <button className="w-7 h-7 rounded-md bg-foreground text-background flex items-center justify-center hover:opacity-90">
            <Plus className="w-4 h-4" />
          </button>
        </div>
        <div className="p-3 border-b border-border">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-muted-foreground" />
            <input
              placeholder="Search phone numbers"
              className="w-full h-9 rounded-md bg-card border border-border pl-9 pr-3 text-sm focus:outline-none focus:ring-2 focus:ring-ring/40"
            />
          </div>
        </div>
      </aside>
      <div className="flex-1 grid place-items-center">
        <div className="text-center">
          <div className="w-12 h-12 mx-auto rounded-xl bg-surface-muted border border-border flex items-center justify-center">
            <Phone className="w-5 h-5 text-muted-foreground" />
          </div>
          <p className="mt-4 text-sm text-muted-foreground">You don't have any phone numbers</p>
          <Button size="sm" className="mt-4 gap-1.5 bg-gradient-primary text-primary-foreground shadow-elegant">
            <Plus className="w-4 h-4" /> Buy a number
          </Button>
        </div>
      </div>
    </div>
  );
}
