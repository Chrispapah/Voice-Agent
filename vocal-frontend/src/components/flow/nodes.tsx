import { Handle, Position, NodeProps } from "@xyflow/react";
import { Hash, PhoneForwarded, PhoneOff, Plus, Play } from "lucide-react";

export function BeginNode() {
  return (
    <div className="flex items-center gap-1.5 rounded-full bg-gradient-primary px-3 py-1.5 text-xs font-medium text-primary-foreground shadow-elegant">
      <Play className="w-3 h-3 fill-current" /> Begin
      <Handle type="source" position={Position.Right} className="!w-2 !h-2 !bg-primary !border-primary-foreground" />
    </div>
  );
}

export function WelcomeNode({ data }: NodeProps) {
  const d = data as { title: string; message: string; transitions: string[] };
  return (
    <div className="w-72 rounded-xl border border-border bg-card shadow-soft overflow-hidden">
      <Handle type="target" position={Position.Left} className="!w-2 !h-2 !bg-primary" />
      <div className="flex items-center gap-1.5 px-3 py-2 border-b border-border bg-gradient-soft">
        <Hash className="w-3.5 h-3.5 text-primary" />
        <span className="text-xs font-semibold">{d.title}</span>
      </div>
      <div className="px-3 py-2.5 text-xs text-foreground/80 leading-relaxed">
        {d.message}
      </div>
      <div className="border-t border-border">
        <div className="flex items-center justify-between px-3 py-2">
          <span className="text-[11px] font-semibold text-muted-foreground tracking-wide">↳ Transition</span>
          <button className="p-0.5 rounded hover:bg-secondary"><Plus className="w-3 h-3" /></button>
        </div>
        {d.transitions.map((t, i) => (
          <div key={i} className="relative px-3 py-2 border-t border-border text-[11px] text-muted-foreground hover:bg-surface-muted/60">
            ↳ {t}
            <Handle
              type="source"
              position={Position.Right}
              id={`t-${i}`}
              style={{ top: "50%" }}
              className="!w-2 !h-2 !bg-primary"
            />
          </div>
        ))}
      </div>
    </div>
  );
}

function ActionNode({ icon: Icon, label, color }: { icon: any; label: string; color: string }) {
  return (
    <div className={`rounded-lg border border-border bg-card shadow-soft px-3 py-2 flex items-center gap-2 text-xs font-medium`}>
      <Handle type="target" position={Position.Left} className="!w-2 !h-2 !bg-primary" />
      <Icon className={`w-3.5 h-3.5 ${color}`} />
      {label}
      <Handle type="source" position={Position.Right} className="!w-2 !h-2 !bg-primary" />
    </div>
  );
}

export const BridgeTransferNode = () => <ActionNode icon={PhoneForwarded} label="Bridge Transfer" color="text-warning" />;
export const CancelTransferNode = () => <ActionNode icon={PhoneOff} label="Cancel Transfer" color="text-destructive" />;
