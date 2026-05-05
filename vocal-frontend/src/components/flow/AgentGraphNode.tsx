import { createContext, memo, useContext } from "react";
import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import { Wrench } from "lucide-react";

export const AgentGraphEntryContext = createContext<string>("");

type AgentGraphNodeData = Record<string, unknown> & {
  label?: string;
  system_prompt?: string;
  tool_ids?: string[];
};

function AgentGraphNodeInner({ id, data, selected }: NodeProps<Node<AgentGraphNodeData>>) {
  const entryNodeId = useContext(AgentGraphEntryContext);
  const isEntry = id === entryNodeId;
  const label = typeof data.label === "string" && data.label.length > 0 ? data.label : id;
  const toolCount = Array.isArray(data.tool_ids) ? data.tool_ids.length : 0;

  return (
    <>
      <Handle type="target" position={Position.Left} className="!h-3 !w-3 !border-2 !border-background !bg-primary" />
      <div className={selected ? "rounded-xl ring-2 ring-ring ring-offset-2 ring-offset-background" : "rounded-xl"}>
        <div
          className={
            isEntry
              ? "min-w-[190px] max-w-[260px] rounded-xl border-2 border-primary bg-accent px-3 py-2 text-left shadow-soft"
              : "min-w-[190px] max-w-[260px] rounded-xl border border-border bg-card px-3 py-2 text-left shadow-soft"
          }
        >
          {isEntry && (
            <span className="mb-1 inline-block rounded border border-primary bg-secondary px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-primary">
              Entry
            </span>
          )}
          <div className="text-sm font-semibold leading-snug text-foreground">{label}</div>
          <div className="mt-1 flex items-center justify-between gap-3">
            <span className="truncate font-mono text-[10px] text-muted-foreground">{id}</span>
            {toolCount > 0 && (
              <span className="inline-flex items-center gap-1 rounded bg-secondary px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground">
                <Wrench className="h-3 w-3" /> {toolCount}
              </span>
            )}
          </div>
        </div>
      </div>
      <Handle type="source" position={Position.Right} className="!h-3 !w-3 !border-2 !border-background !bg-primary" />
    </>
  );
}

export const AgentGraphNode = memo(AgentGraphNodeInner);
