"use client";

import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import {
  ReactFlow,
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
  useNodesState,
  useEdgesState,
  type Connection,
  type Edge,
  type Node,
  type NodeChange,
  type EdgeChange,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import type { ConversationSpecV1, SpecEdge, SpecNode } from "@/lib/conversationSpec";
import {
  defaultGraphConversationSpec,
  defaultSingleConversationSpec,
} from "@/lib/conversationSpec";
import { AgentGraphEntryContext, agentBuilderNodeTypes } from "./AgentBuilderGraphNode";

type BuilderMode = "legacy_sdr" | "single" | "graph";

function usePrefersDarkMode(): boolean {
  return useSyncExternalStore(
    (onChange) => {
      const mq = window.matchMedia("(prefers-color-scheme: dark)");
      mq.addEventListener("change", onChange);
      return () => mq.removeEventListener("change", onChange);
    },
    () => window.matchMedia("(prefers-color-scheme: dark)").matches,
    () => false,
  );
}

function inferMode(spec: ConversationSpecV1 | null | undefined): BuilderMode {
  if (spec == null) return "legacy_sdr";
  if (spec.template === "sdr") return "legacy_sdr";
  if (spec.mode === "single") return "single";
  return "graph";
}

function specToNodes(spec: ConversationSpecV1): Node[] {
  if (spec.mode !== "graph" || !spec.nodes?.length) return [];
  return spec.nodes.map((n, i) => ({
    id: n.id,
    type: "agentSpec",
    position: spec.positions?.[n.id] ?? { x: 80 + i * 300, y: 100 },
    data: {
      label: n.label || n.id,
      system_prompt: n.system_prompt,
      static_message: typeof n.static_message === "string" ? n.static_message : "",
      ...(n.loop_min_turns != null ? { loop_min_turns: n.loop_min_turns } : {}),
      ...(n.loop_max_turns != null ? { loop_max_turns: n.loop_max_turns } : {}),
    },
  }));
}

function specToEdges(spec: ConversationSpecV1): Edge[] {
  if (spec.mode !== "graph" || !spec.edges?.length) return [];
  return spec.edges.map((e, i) => ({
    id: `e-${e.from}-${e.to}-${i}`,
    source: e.from,
    target: e.to,
  }));
}

function nodesEdgesToSpec(
  base: ConversationSpecV1,
  nodes: Node[],
  edges: Edge[],
  entryNodeId: string,
): ConversationSpecV1 {
  const specNodes: SpecNode[] = nodes.map((n) => {
    const sn: SpecNode = {
      id: n.id,
      label: typeof n.data?.label === "string" ? n.data.label : null,
      system_prompt: typeof n.data?.system_prompt === "string" ? n.data.system_prompt : "",
    };
    const staticRaw =
      typeof (n.data as { static_message?: unknown })?.static_message === "string"
        ? String((n.data as { static_message: string }).static_message).trim()
        : "";
    if (staticRaw) {
      sn.static_message = staticRaw;
    }
    const lo = (n.data as { loop_min_turns?: unknown })?.loop_min_turns;
    const hi = (n.data as { loop_max_turns?: unknown })?.loop_max_turns;
    if (typeof lo === "number" && Number.isFinite(lo) && lo >= 1) {
      sn.loop_min_turns = Math.floor(lo);
    }
    if (typeof hi === "number" && Number.isFinite(hi) && hi >= 1) {
      sn.loop_max_turns = Math.floor(hi);
    }
    return sn;
  });
  const specEdges: SpecEdge[] = edges.map((e) => ({ from: e.source, to: e.target }));
  const positions: Record<string, { x: number; y: number }> = {};
  for (const n of nodes) {
    positions[n.id] = { x: n.position.x, y: n.position.y };
  }
  return {
    ...base,
    conversation_spec_version: 1,
    mode: "graph",
    template: "custom",
    entry_node_id: entryNodeId,
    nodes: specNodes,
    edges: specEdges,
    positions,
  };
}

function newNodeId(existing: Set<string>): string {
  let i = 1;
  while (existing.has(`step_${i}`)) i += 1;
  return `step_${i}`;
}

export interface AgentBuilderProps {
  botId: string;
  value: ConversationSpecV1 | null | undefined;
  onChange: (next: ConversationSpecV1 | null) => void;
}

export function AgentBuilder({ botId, value, onChange }: AgentBuilderProps) {
  const prefersDark = usePrefersDarkMode();
  const mode = useMemo(() => inferMode(value ?? null), [value]);
  const lastSyncedJson = useRef<string>("");

  const seedGraph = value?.mode === "graph" ? value : defaultGraphConversationSpec();
  const [entryNodeId, setEntryNodeId] = useState<string>(() => seedGraph.entry_node_id || seedGraph.nodes?.[0]?.id || "intro");
  const [nodes, setNodes] = useNodesState(specToNodes(seedGraph));
  const [edges, setEdges] = useEdgesState(specToEdges(seedGraph));
  const nodesRef = useRef(nodes);
  const edgesRef = useRef(edges);
  nodesRef.current = nodes;
  edgesRef.current = edges;

  useEffect(() => {
    lastSyncedJson.current = "";
  }, [botId]);

  useEffect(() => {
    if (inferMode(value) !== "graph" || !value || value.mode !== "graph") return;
    const incoming = JSON.stringify(value);
    if (incoming === lastSyncedJson.current) return;
    lastSyncedJson.current = incoming;
    setNodes(specToNodes(value));
    setEdges(specToEdges(value));
    setEntryNodeId(value.entry_node_id || value.nodes?.[0]?.id || "intro");
  }, [value, setNodes, setEdges]);

  const pushGraphSpec = useCallback(
    (nextNodes: Node[], nextEdges: Edge[], entry: string) => {
      const base = value?.mode === "graph" ? value : defaultGraphConversationSpec();
      const spec = nodesEdgesToSpec(base, nextNodes, nextEdges, entry);
      lastSyncedJson.current = JSON.stringify(spec);
      onChange(spec);
    },
    [value, onChange],
  );

  const selectedId = useMemo(() => {
    const sel = nodes.find((n) => n.selected);
    return sel?.id ?? null;
  }, [nodes]);

  const handleNodesChange = useCallback(
    (changes: NodeChange[]) => {
      setNodes((nds) => {
        const next = applyNodeChanges(changes, nds);
        const skipPush = changes.every((c) => {
          if (c.type === "select" || c.type === "dimensions") return true;
          if (c.type === "position" && "dragging" in c) {
            return Boolean((c as { dragging?: boolean }).dragging);
          }
          return false;
        });
        if (!skipPush) {
          pushGraphSpec(next, edgesRef.current, entryNodeId);
        }
        return next;
      });
    },
    [setNodes, pushGraphSpec, entryNodeId],
  );

  const handleEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      setEdges((eds) => {
        const next = applyEdgeChanges(changes, eds);
        pushGraphSpec(nodesRef.current, next, entryNodeId);
        return next;
      });
    },
    [setEdges, pushGraphSpec, entryNodeId],
  );

  const onConnect = useCallback(
    (c: Connection) => {
      setEdges((eds) => {
        const next = addEdge(c, eds);
        pushGraphSpec(nodesRef.current, next, entryNodeId);
        return next;
      });
    },
    [setEdges, pushGraphSpec, entryNodeId],
  );

  const setMode = (m: BuilderMode) => {
    if (m === "legacy_sdr") {
      lastSyncedJson.current = "";
      onChange(null);
      return;
    }
    if (m === "single") {
      const sp =
        value?.mode === "single" && value.system_prompt
          ? value.system_prompt
          : "You are a helpful voice assistant. Keep replies to one or two short sentences. Context: {lead_name} at {company}.";
      lastSyncedJson.current = "";
      onChange(defaultSingleConversationSpec(sp));
      return;
    }
    const g = defaultGraphConversationSpec();
    lastSyncedJson.current = JSON.stringify(g);
    onChange(g);
  };

  const singlePrompt =
    value?.mode === "single" && value.system_prompt != null
      ? value.system_prompt
      : defaultSingleConversationSpec("").system_prompt ?? "";

  return (
    <div className="space-y-6">
      <div>
        <p className="mb-3 text-sm text-[var(--muted-foreground)]">
          Choose how this conversation is orchestrated. Classic SDR uses the built-in sales graph and stage
          prompts. Single uses one system prompt for every turn. Graph links multiple agents (nodes) with
          directed edges; when a node has multiple outgoing edges, the model picks the next step from the user’s
          latest message.
        </p>
        <div className="flex flex-wrap gap-4">
          <label className="flex cursor-pointer items-center gap-2 text-sm">
            <input
              type="radio"
              name="conv-mode"
              checked={mode === "legacy_sdr"}
              onChange={() => setMode("legacy_sdr")}
            />
            Classic SDR (built-in graph)
          </label>
          <label className="flex cursor-pointer items-center gap-2 text-sm">
            <input type="radio" name="conv-mode" checked={mode === "single"} onChange={() => setMode("single")} />
            Single agent
          </label>
          <label className="flex cursor-pointer items-center gap-2 text-sm">
            <input type="radio" name="conv-mode" checked={mode === "graph"} onChange={() => setMode("graph")} />
            Agent graph
          </label>
        </div>
      </div>

      {mode === "single" && (
        <div>
          <label className="mb-1.5 block text-sm font-medium">System prompt</label>
          <textarea
            rows={12}
            className="input-field font-mono text-xs"
            value={singlePrompt}
            onChange={(e) =>
              onChange({
                conversation_spec_version: 1,
                mode: "single",
                template: "custom",
                system_prompt: e.target.value,
              })
            }
            placeholder="Instructions for the model…"
          />
          <p className="mt-2 text-xs text-[var(--muted-foreground)]">
            Variables: {"{lead_name}"}, {"{company}"}, {"{lead_context}"}, {"{pain_points}"}, {"{sales_rep_name}"}
          </p>
        </div>
      )}

      {mode === "graph" && (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-3">
            <label className="text-sm font-medium">Entry node</label>
            <select
              className="input-field max-w-xs text-sm"
              value={entryNodeId}
              onChange={(e) => {
                const v = e.target.value;
                setEntryNodeId(v);
                pushGraphSpec(nodes, edges, v);
              }}
            >
              {nodes.map((n) => (
                <option key={n.id} value={n.id}>
                  {n.id}
                </option>
              ))}
            </select>
            <button
              type="button"
              className="rounded-md border border-[var(--border)] px-3 py-1.5 text-xs font-medium hover:bg-[var(--secondary)]"
              onClick={() => {
                const ids = new Set(nodes.map((n) => n.id));
                const id = newNodeId(ids);
                const next: Node[] = [
                  ...nodes,
                  {
                    id,
                    type: "agentSpec",
                    position: { x: 120 + nodes.length * 40, y: 180 + nodes.length * 20 },
                    data: {
                      label: id,
                      system_prompt:
                        "Describe this agent’s role in one paragraph. Voice: one or two short sentences per turn.",
                      static_message: "",
                    },
                  },
                ];
                setNodes(next);
                pushGraphSpec(next, edges, entryNodeId || id);
              }}
            >
              Add node
            </button>
            <button
              type="button"
              className="rounded-md border border-[var(--border)] px-3 py-1.5 text-xs font-medium text-red-700 hover:bg-[var(--secondary)] dark:text-red-300"
              onClick={() => {
                if (!selectedId) return;
                const nextNodes = nodes.filter((n) => n.id !== selectedId);
                const nextEdges = edges.filter((e) => e.source !== selectedId && e.target !== selectedId);
                setNodes(nextNodes);
                setEdges(nextEdges);
                const nextEntry =
                  entryNodeId === selectedId ? (nextNodes[0]?.id ?? "intro") : entryNodeId;
                setEntryNodeId(nextEntry);
                pushGraphSpec(nextNodes, nextEdges, nextEntry);
              }}
            >
              Delete selected
            </button>
          </div>

          <div className="h-[min(720px,82vh)] min-h-[480px] overflow-hidden rounded-lg border border-[var(--border)] bg-[color-mix(in_srgb,var(--muted)_35%,transparent)]">
            <AgentGraphEntryContext.Provider value={entryNodeId}>
              <ReactFlow
                className="h-full w-full"
                colorMode={prefersDark ? "dark" : "light"}
                nodeTypes={agentBuilderNodeTypes}
                defaultEdgeOptions={{
                  style: { stroke: "var(--primary)", strokeWidth: 2 },
                }}
                nodes={nodes}
                edges={edges}
                onNodesChange={handleNodesChange}
                onEdgesChange={handleEdgesChange}
                onConnect={onConnect}
                onNodeDragStop={(_, node) => {
                  setNodes((curr) => {
                    const next = curr.map((n) => (n.id === node.id ? { ...n, position: node.position } : n));
                    pushGraphSpec(next, edges, entryNodeId);
                    return next;
                  });
                }}
                fitView
              >
                <Background variant={BackgroundVariant.Dots} gap={18} size={1.25} color="var(--muted-foreground)" />
                <MiniMap
                  className="!m-2 !overflow-hidden !rounded-md !border !border-[var(--border)] !shadow-sm"
                  nodeStrokeWidth={2}
                  nodeColor={(n) => (n.id === entryNodeId ? "var(--primary)" : "var(--muted-foreground)")}
                />
                <Controls className="!m-2 !overflow-hidden !rounded-md !border !border-[var(--border)] !shadow-sm" />
              </ReactFlow>
            </AgentGraphEntryContext.Provider>
          </div>

          {selectedId && (
            <div className="rounded-md border border-[var(--border)] bg-[var(--secondary)]/30 p-4">
              <h3 className="mb-2 text-sm font-semibold">Node: {selectedId}</h3>
              <label className="mb-1 block text-xs font-medium">Display label</label>
              <input
                className="input-field mb-3 text-sm"
                value={String(nodes.find((n) => n.id === selectedId)?.data?.label ?? "")}
                onChange={(e) => {
                  setNodes((curr) => {
                    const next = curr.map((n) =>
                      n.id === selectedId ? { ...n, data: { ...n.data, label: e.target.value } } : n,
                    );
                    pushGraphSpec(next, edges, entryNodeId);
                    return next;
                  });
                }}
              />
              <label className="mb-1 block text-xs font-medium">System prompt</label>
              <textarea
                rows={6}
                className="input-field font-mono text-xs"
                value={String(nodes.find((n) => n.id === selectedId)?.data?.system_prompt ?? "")}
                onChange={(e) => {
                  setNodes((curr) => {
                    const next = curr.map((n) =>
                      n.id === selectedId ? { ...n, data: { ...n.data, system_prompt: e.target.value } } : n,
                    );
                    pushGraphSpec(next, edges, entryNodeId);
                    return next;
                  });
                }}
              />
              <label className="mb-1 mt-3 block text-xs font-medium">Static message (optional)</label>
              <textarea
                rows={4}
                className="input-field font-mono text-xs"
                placeholder="Leave empty to use the LLM with the system prompt above."
                value={String(nodes.find((n) => n.id === selectedId)?.data?.static_message ?? "")}
                onChange={(e) => {
                  setNodes((curr) => {
                    const next = curr.map((n) =>
                      n.id === selectedId ? { ...n, data: { ...n.data, static_message: e.target.value } } : n,
                    );
                    pushGraphSpec(next, edges, entryNodeId);
                    return next;
                  });
                }}
              />
              <p className="mt-1 text-[11px] text-[var(--muted-foreground)]">
                When set, this exact text is spoken on this node after the user speaks (no reply LLM). The opening
                line still uses General → Initial greeting.
              </p>
            </div>
          )}

          <p className="text-xs text-[var(--muted-foreground)]">
            Connect nodes by dragging from one handle to another. The entry node receives the first turn (spoken
            opener from General → Initial greeting). Node ids must start with a letter and use letters, numbers,
            and underscores only.
          </p>
        </div>
      )}
    </div>
  );
}
