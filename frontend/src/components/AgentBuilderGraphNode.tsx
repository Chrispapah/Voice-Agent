"use client";

import { createContext, memo, useContext } from "react";
import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";

export const AgentGraphEntryContext = createContext<string>("");

type AgentGraphNodeData = Record<string, unknown> & {
  label?: string;
  system_prompt?: string;
};

function AgentBuilderGraphNodeInner({ id, data, selected }: NodeProps<Node<AgentGraphNodeData>>) {
  const entryNodeId = useContext(AgentGraphEntryContext);
  const isEntry = id === entryNodeId;
  const label = typeof data.label === "string" && data.label.length > 0 ? data.label : id;

  return (
    <>
      <Handle
        type="target"
        position={Position.Top}
        className="!h-3 !w-3 !border-2 !border-[var(--background)] !bg-[var(--primary)]"
      />
      <div
        className={
          selected
            ? "rounded-[var(--radius)] ring-2 ring-[var(--ring)] ring-offset-2 ring-offset-[var(--background)]"
            : "rounded-[var(--radius)]"
        }
      >
        <div
          className={
            isEntry
              ? "min-w-[170px] max-w-[240px] rounded-[var(--radius)] border-2 border-[var(--primary)] bg-[color-mix(in_srgb,var(--primary)_14%,var(--card))] px-3 py-2 text-left shadow-md"
              : "min-w-[170px] max-w-[240px] rounded-[var(--radius)] border border-[var(--border)] bg-[var(--muted)] px-3 py-2 text-left shadow-md"
          }
        >
          {isEntry ? (
            <span className="mb-1 inline-block rounded border border-[var(--primary)] bg-[var(--secondary)] px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-[var(--primary)]">
              Entry
            </span>
          ) : null}
          <div className="text-sm font-semibold leading-snug text-[var(--foreground)]">{label}</div>
          <div className="mt-1 truncate font-mono text-[10px] text-[var(--muted-foreground)]">{id}</div>
        </div>
      </div>
      <Handle
        type="source"
        position={Position.Bottom}
        className="!h-3 !w-3 !border-2 !border-[var(--background)] !bg-[var(--primary)]"
      />
    </>
  );
}

export const AgentBuilderGraphNode = memo(AgentBuilderGraphNodeInner);

export const agentBuilderNodeTypes = { agentSpec: AgentBuilderGraphNode };
