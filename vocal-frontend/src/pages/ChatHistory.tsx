import { MessageSquare, Calendar, Filter, Upload, History } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function ChatHistoryPage() {
  return (
    <div className="flex flex-col flex-1 min-h-0">
      <header className="flex items-center justify-between px-8 py-5 border-b border-border">
        <div className="flex items-center gap-2">
          <MessageSquare className="w-4 h-4" />
          <h1 className="text-base font-semibold">Chat History</h1>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm"><History className="w-4 h-4" /></Button>
          <Button variant="outline" size="sm" className="gap-1.5"><Upload className="w-4 h-4" /> Export</Button>
          <Button variant="outline" size="sm">Customize View</Button>
          <Button variant="outline" size="sm">Custom Attributes</Button>
        </div>
      </header>
      <div className="flex items-center gap-2 px-8 py-3 border-b border-border">
        <Button variant="outline" size="sm" className="gap-1.5"><Calendar className="w-4 h-4" /> Date Range</Button>
        <Button variant="outline" size="sm" className="gap-1.5"><Filter className="w-4 h-4" /> Filter</Button>
      </div>
      <div className="flex-1 overflow-auto">
        <table className="w-full text-sm">
          <thead className="bg-surface-muted/60 text-muted-foreground">
            <tr className="text-left">
              {["Time","Cost","Session ID","Session Status","User Sentiment","From","To"].map(h => (
                <th key={h} className="px-6 py-3 font-medium whitespace-nowrap">{h}</th>
              ))}
            </tr>
          </thead>
        </table>
        <div className="grid place-items-center py-32 text-sm text-muted-foreground">No chat sessions yet</div>
      </div>
      <footer className="px-8 py-3 border-t border-border text-xs text-muted-foreground">
        Page 1 of 1 · Total Session: 0
      </footer>
    </div>
  );
}