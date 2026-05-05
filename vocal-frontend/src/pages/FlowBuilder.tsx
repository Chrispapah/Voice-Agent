import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  BackgroundVariant,
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
  type Connection,
  type Edge,
  type EdgeChange,
  type Node,
  type NodeChange,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import {
  Bot,
  BookOpen,
  ChevronLeft,
  GitBranch,
  MessageSquare,
  Mic,
  Play,
  Plus,
  Save,
  Send,
  Sparkles,
  Trash2,
  Volume2,
  Wrench,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  createLead,
  getBot,
  listLeads,
  sendTestTurn,
  startTestSession,
  stopTestSession,
  updateBot,
  type BotConfig,
  type Lead,
  listTools,
  type ToolDefinition,
  AuthRequiredError,
  listKnowledgeBases,
  listBotKnowledgeBaseIds,
  listNodeKnowledgeBaseAssignments,
  setBotKnowledgeBases,
  setNodeKnowledgeBaseAssignments,
  type KnowledgeBase,
} from "@/lib/api";
import {
  defaultGraphConversationSpec,
  defaultSingleConversationSpec,
  type ConversationSpecV1,
  type SpecEdge,
  type SpecNode,
} from "@/lib/conversationSpec";
import { AgentGraphNode, AgentGraphEntryContext } from "@/components/flow/AgentGraphNode";

type BuilderMode = "single" | "graph";
type ChatMessage = { role: "human" | "agent"; content: string };

const nodeTypes = { agentSpec: AgentGraphNode };

function inferMode(spec: ConversationSpecV1 | null | undefined): BuilderMode {
  return spec?.mode === "single" ? "single" : "graph";
}

function normalizeToolIds(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function specToNodes(spec: ConversationSpecV1): Node[] {
  if (spec.mode !== "graph" || !spec.nodes?.length) return specToNodes(defaultGraphConversationSpec());
  return spec.nodes.map((n, i) => ({
    id: n.id,
    type: "agentSpec",
    position: spec.positions?.[n.id] ?? { x: 160 + i * 320, y: 160 },
    data: { label: n.label || n.id, system_prompt: n.system_prompt, tool_ids: n.tool_ids ?? [] },
  }));
}

function specToEdges(spec: ConversationSpecV1): Edge[] {
  if (spec.mode !== "graph" || !spec.edges?.length) return [];
  return spec.edges.map((e, i) => ({
    id: `e-${e.from}-${e.to}-${i}`,
    source: e.from,
    target: e.to,
    style: { stroke: "hsl(var(--primary))", strokeWidth: 2 },
  }));
}

function nodesEdgesToSpec(
  base: ConversationSpecV1,
  nodes: Node[],
  edges: Edge[],
  entryNodeId: string,
): ConversationSpecV1 {
  const specNodes: SpecNode[] = nodes.map((n) => ({
    id: n.id,
    label: typeof n.data?.label === "string" ? n.data.label : null,
    system_prompt: typeof n.data?.system_prompt === "string" ? n.data.system_prompt : "",
    tool_ids: normalizeToolIds(n.data?.tool_ids),
  }));
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

function withSinglePromptToolIds(systemPrompt: string, toolIds: string[]): ConversationSpecV1 {
  return {
    ...defaultSingleConversationSpec(systemPrompt),
    tool_ids: toolIds,
  };
}

function newNodeId(existing: Set<string>): string {
  let i = 1;
  while (existing.has(`step_${i}`)) i += 1;
  return `step_${i}`;
}

function firstUsableSpec(bot: BotConfig | null): ConversationSpecV1 {
  const spec = bot?.conversation_spec;
  if (spec?.template !== "sdr" && (spec?.mode === "single" || spec?.mode === "graph")) return spec;
  return defaultGraphConversationSpec();
}

function ToolSelector({
  tools,
  selectedToolIds,
  onChange,
}: {
  tools: ToolDefinition[];
  selectedToolIds: string[];
  onChange: (toolIds: string[]) => void;
}) {
  function toggleTool(toolId: string, checked: boolean) {
    if (checked) {
      onChange([...selectedToolIds, toolId]);
      return;
    }
    onChange(selectedToolIds.filter((id) => id !== toolId));
  }

  return (
    <div className="rounded-lg border border-border bg-surface-muted/30 p-3">
      <div className="mb-2 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2 text-sm font-semibold">
          <Wrench className="h-4 w-4" /> Tools
        </div>
        <Link to="/tools" className="text-xs text-primary hover:underline">Define tools</Link>
      </div>
      {tools.length === 0 ? (
        <p className="text-xs text-muted-foreground">
          No tools defined yet. Create reusable tools from the main Tools menu, then attach them here.
        </p>
      ) : (
        <div className="space-y-2">
          {tools.map((tool) => (
            <label key={tool.id} className="flex cursor-pointer items-start gap-2 rounded-md border border-border bg-card px-2 py-2 text-sm">
              <input
                type="checkbox"
                checked={selectedToolIds.includes(tool.id)}
                onChange={(e) => toggleTool(tool.id, e.target.checked)}
                className="mt-1"
              />
              <span className="min-w-0">
                <span className="block font-medium">{tool.name}</span>
                <span className="block truncate text-xs text-muted-foreground">{tool.description || tool.kind}</span>
              </span>
            </label>
          ))}
        </div>
      )}
    </div>
  );
}

function KnowledgeBaseSelector({
  knowledgeBases,
  selectedKnowledgeBaseIds,
  onChange,
}: {
  knowledgeBases: KnowledgeBase[];
  selectedKnowledgeBaseIds: string[];
  onChange: (knowledgeBaseIds: string[]) => void;
}) {
  function toggleKnowledgeBase(knowledgeBaseId: string, checked: boolean) {
    if (checked) {
      onChange([...selectedKnowledgeBaseIds, knowledgeBaseId]);
      return;
    }
    onChange(selectedKnowledgeBaseIds.filter((id) => id !== knowledgeBaseId));
  }

  return (
    <div className="rounded-lg border border-border bg-surface-muted/30 p-3">
      <div className="mb-2 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2 text-sm font-semibold">
          <BookOpen className="h-4 w-4" /> Knowledge Base
        </div>
        <Link to="/knowledge-base" className="text-xs text-primary hover:underline">Manage KBs</Link>
      </div>
      {knowledgeBases.length === 0 ? (
        <p className="text-xs text-muted-foreground">
          No knowledge bases yet. Create one from the Knowledge Base menu, then attach it here.
        </p>
      ) : (
        <div className="space-y-2">
          {knowledgeBases.map((knowledgeBase) => (
            <label key={knowledgeBase.id} className="flex cursor-pointer items-start gap-2 rounded-md border border-border bg-card px-2 py-2 text-sm">
              <input
                type="checkbox"
                checked={selectedKnowledgeBaseIds.includes(knowledgeBase.id)}
                onChange={(e) => toggleKnowledgeBase(knowledgeBase.id, e.target.checked)}
                className="mt-1"
              />
              <span className="min-w-0">
                <span className="block font-medium">{knowledgeBase.name}</span>
                <span className="block truncate text-xs text-muted-foreground">{knowledgeBase.description || "No description"}</span>
              </span>
            </label>
          ))}
        </div>
      )}
    </div>
  );
}

export default function FlowBuilderPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const botId = id || "";
  const [bot, setBot] = useState<BotConfig | null>(null);
  const [draftName, setDraftName] = useState("");
  const [initialGreeting, setInitialGreeting] = useState("");
  const [elevenlabsVoiceId, setElevenlabsVoiceId] = useState("");
  const [elevenlabsModelId, setElevenlabsModelId] = useState("");
  const [deepgramModel, setDeepgramModel] = useState("");
  const [deepgramLanguage, setDeepgramLanguage] = useState("");
  const [spec, setSpec] = useState<ConversationSpecV1>(defaultGraphConversationSpec());
  const [mode, setMode] = useState<BuilderMode>("graph");
  const [entryNodeId, setEntryNodeId] = useState("welcome");
  const [nodes, setNodes] = useState<Node[]>(specToNodes(defaultGraphConversationSpec()));
  const [edges, setEdges] = useState<Edge[]>(specToEdges(defaultGraphConversationSpec()));
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [loading, setLoading] = useState(true);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [leads, setLeads] = useState<Lead[]>([]);
  const [tools, setTools] = useState<ToolDefinition[]>([]);
  const [knowledgeBases, setKnowledgeBases] = useState<KnowledgeBase[]>([]);
  const [selectedAgentKnowledgeBaseIds, setSelectedAgentKnowledgeBaseIds] = useState<string[]>([]);
  const [nodeKnowledgeBaseIds, setNodeKnowledgeBaseIds] = useState<Record<string, string[]>>({});
  const [selectedLead, setSelectedLead] = useState("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [testing, setTesting] = useState(false);
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);
  const nodesRef = useRef(nodes);
  const edgesRef = useRef(edges);
  const sessionIdRef = useRef(sessionId);
  const botIdRef = useRef(botId);
  nodesRef.current = nodes;
  edgesRef.current = edges;
  sessionIdRef.current = sessionId;
  botIdRef.current = botId;

  useEffect(() => {
    return () => {
      const activeSessionId = sessionIdRef.current;
      const activeBotId = botIdRef.current;
      if (activeBotId && activeSessionId) {
        void stopTestSession(activeBotId, activeSessionId, true).catch(() => undefined);
      }
    };
  }, []);

  useEffect(() => {
    if (!botId) return;
    setLoading(true);
    setError("");
    Promise.all([
      getBot(botId),
      listLeads(botId),
      listTools(),
      listKnowledgeBases(),
      listBotKnowledgeBaseIds(botId),
      listNodeKnowledgeBaseAssignments(botId),
    ])
      .then(([loadedBot, loadedLeads, loadedTools, loadedKnowledgeBases, loadedAgentKnowledgeBaseIds, loadedNodeKnowledgeBaseIds]) => {
        const loadedSpec = firstUsableSpec(loadedBot);
        setBot(loadedBot);
        setDraftName(loadedBot.name);
        setInitialGreeting(loadedBot.initial_greeting || "");
        setElevenlabsVoiceId(loadedBot.elevenlabs_voice_id || "");
        setElevenlabsModelId(loadedBot.elevenlabs_model_id || "");
        setDeepgramModel(loadedBot.deepgram_model || "");
        setDeepgramLanguage(loadedBot.deepgram_language || "");
        setSpec(loadedSpec);
        setMode(inferMode(loadedSpec));
        setEntryNodeId(loadedSpec.entry_node_id || loadedSpec.nodes?.[0]?.id || "welcome");
        setNodes(specToNodes(loadedSpec));
        setEdges(specToEdges(loadedSpec));
        setLeads(loadedLeads);
        setTools(loadedTools);
        setKnowledgeBases(loadedKnowledgeBases);
        setSelectedAgentKnowledgeBaseIds(loadedAgentKnowledgeBaseIds);
        setNodeKnowledgeBaseIds(loadedNodeKnowledgeBaseIds);
        setSelectedLead(loadedLeads[0]?.id || "");
      })
      .catch((err: unknown) => {
        if (err instanceof AuthRequiredError) {
          navigate("/auth");
          return;
        }
        setError(err instanceof Error ? err.message : "Failed to load agent");
      })
      .finally(() => setLoading(false));
  }, [botId, navigate]);

  const selectedNode = useMemo(
    () => nodes.find((node) => node.id === selectedNodeId) || null,
    [nodes, selectedNodeId],
  );

  const pushGraphSpec = useCallback(
    (nextNodes: Node[], nextEdges: Edge[], entry: string) => {
      setSpec((current) => nodesEdgesToSpec(current.mode === "graph" ? current : defaultGraphConversationSpec(), nextNodes, nextEdges, entry));
    },
    [],
  );

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => {
      const selected = changes.find((change) => change.type === "select" && "selected" in change && change.selected);
      if (selected && "id" in selected) {
        setSelectedNodeId(String(selected.id));
        setSelectedEdgeId(null);
      }
      setNodes((current) => {
        const next = applyNodeChanges(changes, current);
        const skipPush = changes.every((change) => {
          if (change.type === "select" || change.type === "dimensions") return true;
          if (change.type === "position" && "dragging" in change) return Boolean(change.dragging);
          return false;
        });
        if (!skipPush) pushGraphSpec(next, edgesRef.current, entryNodeId);
        return next;
      });
    },
    [entryNodeId, pushGraphSpec],
  );

  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      const selected = changes.find((change) => change.type === "select" && "selected" in change && change.selected);
      if (selected && "id" in selected) {
        setSelectedEdgeId(String(selected.id));
        setSelectedNodeId(null);
      }
      setEdges((current) => {
        const next = applyEdgeChanges(changes, current);
        setSelectedEdgeId((currentSelectedEdgeId) =>
          currentSelectedEdgeId && next.some((edge) => edge.id === currentSelectedEdgeId) ? currentSelectedEdgeId : null,
        );
        pushGraphSpec(nodesRef.current, next, entryNodeId);
        return next;
      });
    },
    [entryNodeId, pushGraphSpec],
  );

  const onConnect = useCallback(
    (connection: Connection) => {
      setEdges((current) => {
        const next = addEdge(
          { ...connection, style: { stroke: "hsl(var(--primary))", strokeWidth: 2 } },
          current,
        );
        pushGraphSpec(nodesRef.current, next, entryNodeId);
        return next;
      });
    },
    [entryNodeId, pushGraphSpec],
  );

  function changeMode(nextMode: BuilderMode) {
    setMode(nextMode);
    if (nextMode === "single") {
      const systemPrompt = spec.mode === "single" && spec.system_prompt
        ? spec.system_prompt
        : "You are a helpful voice agent. Keep replies to one or two short sentences. Context: {lead_name} at {company}.";
      setSpec(withSinglePromptToolIds(systemPrompt, spec.mode === "single" ? spec.tool_ids ?? [] : []));
      return;
    }
    const graphSpec = spec.mode === "graph" ? spec : defaultGraphConversationSpec();
    setSpec(graphSpec);
    setEntryNodeId(graphSpec.entry_node_id || graphSpec.nodes?.[0]?.id || "welcome");
    setNodes(specToNodes(graphSpec));
    setEdges(specToEdges(graphSpec));
  }

  function addNode() {
    const id = newNodeId(new Set(nodes.map((node) => node.id)));
    const nextNodes = [
      ...nodes,
      {
        id,
        type: "agentSpec",
        position: { x: 220 + nodes.length * 80, y: 180 + nodes.length * 40 },
        data: {
          label: id,
          system_prompt: "Describe this agent's role. Keep voice replies short and natural.",
          tool_ids: [],
        },
      },
    ];
    setNodes(nextNodes);
    setSelectedNodeId(id);
    pushGraphSpec(nextNodes, edges, entryNodeId || id);
  }

  function deleteSelectedNode() {
    if (!selectedNodeId) return;
    const nextNodes = nodes.filter((node) => node.id !== selectedNodeId);
    const nextEdges = edges.filter((edge) => edge.source !== selectedNodeId && edge.target !== selectedNodeId);
    const nextEntry = entryNodeId === selectedNodeId ? nextNodes[0]?.id || "welcome" : entryNodeId;
    setNodes(nextNodes);
    setEdges(nextEdges);
    setSelectedNodeId(null);
    setEntryNodeId(nextEntry);
    setNodeKnowledgeBaseIds((current) => {
      const next = { ...current };
      delete next[selectedNodeId];
      return next;
    });
    pushGraphSpec(nextNodes, nextEdges, nextEntry);
  }

  function deleteSelectedEdge() {
    if (!selectedEdgeId) return;
    const nextEdges = edges.filter((edge) => edge.id !== selectedEdgeId);
    setEdges(nextEdges);
    setSelectedEdgeId(null);
    pushGraphSpec(nodes, nextEdges, entryNodeId);
  }

  function deleteSelectedGraphItem() {
    if (selectedEdgeId) {
      deleteSelectedEdge();
      return;
    }
    deleteSelectedNode();
  }

  function updateSelectedNode(field: "label" | "system_prompt", value: string) {
    if (!selectedNodeId) return;
    setNodes((current) => {
      const next = current.map((node) =>
        node.id === selectedNodeId ? { ...node, data: { ...node.data, [field]: value } } : node,
      );
      pushGraphSpec(next, edgesRef.current, entryNodeId);
      return next;
    });
  }

  function updateSelectedNodeTools(toolIds: string[]) {
    if (!selectedNodeId) return;
    setNodes((current) => {
      const next = current.map((node) =>
        node.id === selectedNodeId ? { ...node, data: { ...node.data, tool_ids: toolIds } } : node,
      );
      pushGraphSpec(next, edgesRef.current, entryNodeId);
      return next;
    });
  }

  function updateSelectedNodeKnowledgeBases(knowledgeBaseIds: string[]) {
    if (!selectedNodeId) return;
    setNodeKnowledgeBaseIds((current) => ({
      ...current,
      [selectedNodeId]: knowledgeBaseIds,
    }));
  }

  function updateSinglePrompt(value: string) {
    setSpec((current) => withSinglePromptToolIds(value, current.mode === "single" ? current.tool_ids ?? [] : []));
  }

  function updateSingleAgentTools(toolIds: string[]) {
    setSpec((current) => {
      const systemPrompt = current.mode === "single" ? current.system_prompt || "" : "";
      return withSinglePromptToolIds(systemPrompt, toolIds);
    });
  }

  async function handleSave() {
    if (!bot) return;
    setSaving(true);
    setStatus("");
    setError("");
    try {
      const activeNodeIds = new Set(nodes.map((node) => node.id));
      const activeNodeAssignments = Object.fromEntries(
        Object.entries(nodeKnowledgeBaseIds).filter(([nodeId]) => activeNodeIds.has(nodeId)),
      );
      const saved = await updateBot(bot.id, {
        name: draftName,
        initial_greeting: initialGreeting,
        elevenlabs_voice_id: elevenlabsVoiceId || null,
        elevenlabs_model_id: elevenlabsModelId || "eleven_turbo_v2",
        deepgram_model: deepgramModel || "nova-2",
        deepgram_language: deepgramLanguage || "en-US",
        conversation_spec: spec,
      });
      await Promise.all([
        setBotKnowledgeBases(bot.id, selectedAgentKnowledgeBaseIds),
        setNodeKnowledgeBaseAssignments(bot.id, activeNodeAssignments),
      ]);
      setBot(saved);
      setNodeKnowledgeBaseIds(activeNodeAssignments);
      setStatus("Published to LangChain brain");
      setTimeout(() => setStatus(""), 2500);
    } catch (err: unknown) {
      if (err instanceof AuthRequiredError) {
        navigate("/auth");
        return;
      }
      setError(err instanceof Error ? err.message : "Failed to publish agent");
    } finally {
      setSaving(false);
    }
  }

  async function ensureLead(): Promise<string> {
    if (selectedLead) return selectedLead;
    const created = await createLead(botId, {
      lead_name: "Test Lead",
      company: "Demo Company",
      phone_number: "+15551234567",
      lead_email: "test@example.com",
      lead_context: "Created from the vocal frontend test console.",
      timezone: "UTC",
      owner_name: "Sales Team",
      calendar_id: "sales-team",
    });
    setLeads((current) => [created, ...current]);
    setSelectedLead(created.id);
    return created.id;
  }

  async function handleStartTest() {
    setTesting(true);
    setError("");
    try {
      if (sessionId) {
        await stopTestSession(botId, sessionId).catch(() => undefined);
      }
      await handleSave();
      const leadId = await ensureLead();
      const response = await startTestSession(botId, leadId);
      setSessionId(response.conversation_id);
      setMessages([{ role: "agent", content: response.agent_response }]);
    } catch (err: unknown) {
      if (err instanceof AuthRequiredError) {
        navigate("/auth");
        return;
      }
      setError(err instanceof Error ? err.message : "Failed to start test session");
    } finally {
      setTesting(false);
    }
  }

  async function handleStopTest() {
    if (!sessionId) return;
    const activeSessionId = sessionId;
    setTesting(true);
    setError("");
    try {
      await stopTestSession(botId, activeSessionId);
      setSessionId(null);
      setMessages([]);
      setInput("");
      setStatus("Test stopped");
      setTimeout(() => setStatus(""), 2000);
    } catch (err: unknown) {
      if (err instanceof AuthRequiredError) {
        navigate("/auth");
        return;
      }
      setError(err instanceof Error ? err.message : "Failed to stop test session");
    } finally {
      setTesting(false);
    }
  }

  async function handleSend(e: FormEvent) {
    e.preventDefault();
    if (!sessionId || !input.trim()) return;
    const text = input.trim();
    setInput("");
    setMessages((current) => [...current, { role: "human", content: text }]);
    setTesting(true);
    try {
      const response = await sendTestTurn(botId, sessionId, text);
      setMessages((current) => [...current, { role: "agent", content: response.agent_response }]);
    } catch (err: unknown) {
      if (err instanceof AuthRequiredError) {
        navigate("/auth");
        return;
      }
      setError(err instanceof Error ? err.message : "Failed to get agent response");
    } finally {
      setTesting(false);
    }
  }

  if (loading) {
    return <div className="flex h-screen items-center justify-center text-muted-foreground">Loading agent...</div>;
  }

  return (
    <div className="flex flex-col h-screen bg-background">
      <header className="flex items-center justify-between px-5 h-14 border-b border-border bg-card/60 backdrop-blur">
        <div className="flex items-center gap-3 min-w-0">
          <Link to="/agents" className="p-1.5 rounded-md hover:bg-secondary"><ChevronLeft className="w-4 h-4" /></Link>
          <div className="flex flex-col min-w-0">
            <input
              value={draftName}
              onChange={(e) => setDraftName(e.target.value)}
              className="bg-transparent text-sm font-semibold outline-none"
            />
            <div className="flex items-center gap-3 text-[10px] text-muted-foreground">
              <span>Agent ID: {botId.slice(0, 8)}...</span>
              <span>{mode === "graph" ? "Conversation Flow" : "Single Prompt"}</span>
              <span>LangChain runtime</span>
            </div>
          </div>
        </div>
        <nav className="absolute left-1/2 -translate-x-1/2 flex items-center gap-1 text-sm">
          <button
            onClick={() => changeMode("graph")}
            className={`px-3 py-1.5 rounded-md ${mode === "graph" ? "font-medium border-b-2 border-primary text-foreground" : "text-muted-foreground"}`}
          >
            Flow
          </button>
          <button
            onClick={() => changeMode("single")}
            className={`px-3 py-1.5 rounded-md ${mode === "single" ? "font-medium border-b-2 border-primary text-foreground" : "text-muted-foreground"}`}
          >
            Single Prompt
          </button>
        </nav>
        <div className="flex items-center gap-2">
          {status && <span className="text-xs text-success">{status}</span>}
          <Button variant="outline" size="sm" onClick={handleStartTest} disabled={testing || saving}>
            <Play className="w-3.5 h-3.5 mr-1" /> Test
          </Button>
          <Button size="sm" className="bg-gradient-primary text-primary-foreground hover:opacity-90 shadow-elegant" onClick={handleSave} disabled={saving}>
            <Save className="w-3.5 h-3.5 mr-1" /> {saving ? "Publishing..." : "Publish"}
          </Button>
        </div>
      </header>

      {error && (
        <div className="border-b border-destructive/20 bg-destructive/10 px-5 py-2 text-sm text-destructive">
          {error}
        </div>
      )}

      <div className="flex-1 flex min-h-0">
        <aside className="w-72 border-r border-border bg-surface flex flex-col">
          <div className="p-4 space-y-4 border-b border-border">
            <div>
              <label className="text-[11px] font-semibold text-muted-foreground tracking-wide">INITIAL GREETING</label>
              <textarea
                rows={5}
                value={initialGreeting}
                onChange={(e) => setInitialGreeting(e.target.value)}
                className="mt-2 w-full text-sm border border-border rounded-lg p-3 bg-surface-muted/40 focus:outline-none focus:ring-2 focus:ring-ring/40 resize-none"
              />
            </div>
            <div className="rounded-lg border border-border bg-card p-3">
              <div className="mb-3 text-[11px] font-semibold text-muted-foreground tracking-wide">VOICE I/O PLACEHOLDERS</div>
              <div className="space-y-3">
                <div>
                  <label className="flex items-center gap-1.5 text-xs font-medium">
                    <Volume2 className="h-3.5 w-3.5" /> Text to Speech
                  </label>
                  <div className="mt-2 grid grid-cols-2 gap-2">
                    <input
                      value={elevenlabsVoiceId}
                      onChange={(e) => setElevenlabsVoiceId(e.target.value)}
                      placeholder="Voice ID"
                      className="min-w-0 rounded-lg border border-border bg-background px-2 py-1.5 text-xs"
                    />
                    <input
                      value={elevenlabsModelId}
                      onChange={(e) => setElevenlabsModelId(e.target.value)}
                      placeholder="eleven_turbo_v2"
                      className="min-w-0 rounded-lg border border-border bg-background px-2 py-1.5 text-xs"
                    />
                  </div>
                </div>
                <div>
                  <label className="flex items-center gap-1.5 text-xs font-medium">
                    <Mic className="h-3.5 w-3.5" /> Speech to Text
                  </label>
                  <div className="mt-2 grid grid-cols-2 gap-2">
                    <input
                      value={deepgramModel}
                      onChange={(e) => setDeepgramModel(e.target.value)}
                      placeholder="nova-2"
                      className="min-w-0 rounded-lg border border-border bg-background px-2 py-1.5 text-xs"
                    />
                    <input
                      value={deepgramLanguage}
                      onChange={(e) => setDeepgramLanguage(e.target.value)}
                      placeholder="en-US"
                      className="min-w-0 rounded-lg border border-border bg-background px-2 py-1.5 text-xs"
                    />
                  </div>
                </div>
                <p className="text-[11px] leading-relaxed text-muted-foreground">
                  Provider keys stay in Settings; this placeholder captures the agent-level voice defaults.
                </p>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <button
                onClick={() => changeMode("graph")}
                className={`rounded-lg border px-3 py-2 text-sm ${mode === "graph" ? "border-primary bg-accent" : "border-border hover:bg-secondary"}`}
              >
                <GitBranch className="mx-auto mb-1 h-4 w-4" /> Flow
              </button>
              <button
                onClick={() => changeMode("single")}
                className={`rounded-lg border px-3 py-2 text-sm ${mode === "single" ? "border-primary bg-accent" : "border-border hover:bg-secondary"}`}
              >
                <Bot className="mx-auto mb-1 h-4 w-4" /> Single
              </button>
            </div>
          </div>

          {mode === "graph" ? (
            <div className="p-4 space-y-3">
              <Button variant="outline" size="sm" className="w-full gap-1.5" onClick={addNode}>
                <Plus className="h-4 w-4" /> Add Agent Node
              </Button>
              <Button variant="outline" size="sm" className="w-full gap-1.5" onClick={deleteSelectedGraphItem} disabled={!selectedNodeId && !selectedEdgeId}>
                <Trash2 className="h-4 w-4" /> {selectedEdgeId ? "Delete Selected Connection" : "Delete Selected"}
              </Button>
              <div>
                <label className="text-[11px] font-semibold text-muted-foreground tracking-wide">ENTRY NODE</label>
                <select
                  value={entryNodeId}
                  onChange={(e) => {
                    setEntryNodeId(e.target.value);
                    pushGraphSpec(nodes, edges, e.target.value);
                  }}
                  className="mt-2 w-full rounded-lg border border-border bg-background px-3 py-2 text-sm"
                >
                  {nodes.map((node) => (
                    <option key={node.id} value={node.id}>{String(node.data?.label || node.id)}</option>
                  ))}
                </select>
              </div>
              <KnowledgeBaseSelector
                knowledgeBases={knowledgeBases}
                selectedKnowledgeBaseIds={selectedAgentKnowledgeBaseIds}
                onChange={setSelectedAgentKnowledgeBaseIds}
              />
            </div>
          ) : (
            <div className="p-4 space-y-4">
              <label className="text-[11px] font-semibold text-muted-foreground tracking-wide">SYSTEM PROMPT</label>
              <textarea
                rows={14}
                value={spec.mode === "single" ? spec.system_prompt || "" : ""}
                onChange={(e) => updateSinglePrompt(e.target.value)}
                className="mt-2 w-full text-xs font-mono border border-border rounded-lg p-3 bg-surface-muted/40 focus:outline-none focus:ring-2 focus:ring-ring/40 resize-none"
              />
              <ToolSelector
                tools={tools}
                selectedToolIds={spec.mode === "single" ? spec.tool_ids ?? [] : []}
                onChange={updateSingleAgentTools}
              />
              <KnowledgeBaseSelector
                knowledgeBases={knowledgeBases}
                selectedKnowledgeBaseIds={selectedAgentKnowledgeBaseIds}
                onChange={setSelectedAgentKnowledgeBaseIds}
              />
            </div>
          )}
        </aside>

        <div className="flex-1 relative bg-surface-muted/40">
          {mode === "graph" ? (
            <AgentGraphEntryContext.Provider value={entryNodeId}>
              <ReactFlow
                nodes={nodes}
                edges={edges}
                onNodesChange={onNodesChange}
                onEdgesChange={onEdgesChange}
                onConnect={onConnect}
                onNodeClick={(_, node) => {
                  setSelectedNodeId(node.id);
                  setSelectedEdgeId(null);
                }}
                onEdgeClick={(_, edge) => {
                  setSelectedEdgeId(edge.id);
                  setSelectedNodeId(null);
                }}
                onPaneClick={() => {
                  setSelectedNodeId(null);
                  setSelectedEdgeId(null);
                }}
                onNodeDragStop={(_, node) => {
                  setNodes((current) => {
                    const next = current.map((n) => (n.id === node.id ? { ...n, position: node.position } : n));
                    pushGraphSpec(next, edges, entryNodeId);
                    return next;
                  });
                }}
                nodeTypes={nodeTypes}
                fitView
                proOptions={{ hideAttribution: true }}
              >
                <Background variant={BackgroundVariant.Dots} gap={18} size={1.2} color="hsl(var(--border))" />
                <Controls className="!shadow-soft !border !border-border !rounded-lg overflow-hidden" />
                <MiniMap pannable zoomable className="!bg-card !border !border-border !rounded-lg" maskColor="hsl(var(--surface-muted))" />
              </ReactFlow>
            </AgentGraphEntryContext.Provider>
          ) : (
            <div className="flex h-full items-center justify-center px-8 text-center text-muted-foreground">
              <div>
                <Sparkles className="mx-auto mb-3 h-8 w-8 text-primary" />
                <p className="max-w-md text-sm">
                  Single Prompt mode sends every turn to one LangChain agent using the prompt on the left.
                </p>
              </div>
            </div>
          )}
        </div>

        <aside className="w-96 border-l border-border bg-card flex flex-col">
          <div className="flex border-b border-border">
            <button className="flex-1 px-4 py-3 text-sm font-medium border-b-2 border-primary">Inspector</button>
            <button className="flex-1 px-4 py-3 text-sm text-muted-foreground">Test Agent</button>
          </div>

          <div className="flex-1 min-h-0 overflow-y-auto p-4 space-y-4">
            {mode === "graph" && selectedNode ? (
              <div className="space-y-3">
                <div>
                  <label className="text-[11px] font-semibold text-muted-foreground tracking-wide">NODE LABEL</label>
                  <input
                    value={String(selectedNode.data?.label || "")}
                    onChange={(e) => updateSelectedNode("label", e.target.value)}
                    className="mt-2 w-full rounded-lg border border-border bg-background px-3 py-2 text-sm"
                  />
                </div>
                <div>
                  <label className="text-[11px] font-semibold text-muted-foreground tracking-wide">SYSTEM PROMPT</label>
                  <textarea
                    rows={8}
                    value={String(selectedNode.data?.system_prompt || "")}
                    onChange={(e) => updateSelectedNode("system_prompt", e.target.value)}
                    className="mt-2 w-full text-xs font-mono border border-border rounded-lg p-3 bg-surface-muted/40 focus:outline-none focus:ring-2 focus:ring-ring/40 resize-none"
                  />
                </div>
                <ToolSelector
                  tools={tools}
                  selectedToolIds={normalizeToolIds(selectedNode.data?.tool_ids)}
                  onChange={updateSelectedNodeTools}
                />
                <KnowledgeBaseSelector
                  knowledgeBases={knowledgeBases}
                  selectedKnowledgeBaseIds={nodeKnowledgeBaseIds[selectedNode.id] ?? []}
                  onChange={updateSelectedNodeKnowledgeBases}
                />
              </div>
            ) : (
              <p className="rounded-lg border border-border bg-surface-muted/40 p-3 text-sm text-muted-foreground">
                {mode === "graph" ? "Select a node to edit its LangChain system prompt." : "Edit the single-agent prompt in the left panel."}
              </p>
            )}

            <div className="border-t border-border pt-4">
              <div className="mb-3 flex items-center justify-between">
                <h2 className="flex items-center gap-2 text-sm font-semibold"><MessageSquare className="h-4 w-4" /> Test Console</h2>
                <div className="flex gap-2">
                  {sessionId && (
                    <Button variant="outline" size="sm" onClick={handleStopTest} disabled={testing}>
                      Stop
                    </Button>
                  )}
                  <Button variant="outline" size="sm" onClick={handleStartTest} disabled={testing}>
                    {sessionId ? "Restart" : "Start"}
                  </Button>
                </div>
              </div>
              <select
                value={selectedLead}
                onChange={(e) => setSelectedLead(e.target.value)}
                className="mb-3 w-full rounded-lg border border-border bg-background px-3 py-2 text-sm"
              >
                <option value="">Auto-create demo lead</option>
                {leads.map((lead) => (
                  <option key={lead.id} value={lead.id}>{lead.lead_name} - {lead.company}</option>
                ))}
              </select>
              <div className="h-64 space-y-2 overflow-y-auto rounded-lg border border-border bg-surface-muted/40 p-3">
                {messages.length === 0 ? (
                  <p className="text-sm text-muted-foreground">Start a test to talk to the saved LangChain brain.</p>
                ) : (
                  messages.map((message, index) => (
                    <div key={index} className={`flex ${message.role === "human" ? "justify-end" : "justify-start"}`}>
                      <div className={`max-w-[82%] rounded-lg px-3 py-2 text-xs ${message.role === "human" ? "bg-primary text-primary-foreground" : "bg-card border border-border"}`}>
                        {message.content}
                      </div>
                    </div>
                  ))
                )}
              </div>
              <form onSubmit={handleSend} className="mt-3 flex gap-2">
                <input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder={sessionId ? "Type a reply..." : "Start test first"}
                  disabled={!sessionId || testing}
                  className="min-w-0 flex-1 rounded-lg border border-border bg-background px-3 py-2 text-sm"
                />
                <Button type="submit" size="sm" disabled={!sessionId || !input.trim() || testing}>
                  <Send className="h-4 w-4" />
                </Button>
              </form>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}
