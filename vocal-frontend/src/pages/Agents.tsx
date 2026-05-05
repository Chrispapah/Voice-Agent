import { FormEvent, useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Link, useNavigate } from "react-router-dom";
import { Search, Plus, Upload, MoreHorizontal, Folder, Bot, ChevronLeft } from "lucide-react";
import { Button } from "@/components/ui/button";
import { botExecutionLabel } from "@/lib/conversationSpec";
import {
  AuthRequiredError,
  createAgentFolder,
  createBot,
  listAgentsPageData,
  type AgentListItem,
  type AgentsPageData,
} from "@/lib/api";

type AgentFilter = "all" | "templates" | "transfer" | "folder";
const AGENTS_PAGE_QUERY_KEY = ["agents-page"] as const;

const typeStyles: Record<string, string> = {
  "Conversation Flow": "bg-accent text-accent-foreground",
  "Single Prompt": "bg-secondary text-secondary-foreground",
  "Classic SDR": "bg-secondary text-secondary-foreground",
};

const filterTitles: Record<AgentFilter, string> = {
  all: "All Agents",
  templates: "Template Agents",
  transfer: "Transfer Screening Agents",
  folder: "Folder",
};

function hasTransferSignal(agent: AgentListItem): boolean {
  const searchableText = [
    agent.name,
    agent.conversation_spec ? JSON.stringify(agent.conversation_spec) : "",
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();

  return searchableText.includes("transfer");
}

function matchesFilter(agent: AgentListItem, filter: AgentFilter, folderId: string | null): boolean {
  if (filter === "all") return true;
  if (filter === "templates") {
    return agent.conversation_spec == null || agent.conversation_spec.template === "sdr";
  }
  if (filter === "folder") return agent.folder_id === folderId;
  return hasTransferSignal(agent);
}

function matchesSearch(agent: AgentListItem, query: string): boolean {
  const normalizedQuery = query.trim().toLowerCase();
  if (!normalizedQuery) return true;
  return [agent.name, botExecutionLabel(agent.conversation_spec), agent.elevenlabs_voice_id, agent.twilio_phone_number]
    .filter(Boolean)
    .some((value) => String(value).toLowerCase().includes(normalizedQuery));
}

export default function AgentsPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const {
    data,
    error: loadError,
    isLoading: loading,
  } = useQuery({
    queryKey: AGENTS_PAGE_QUERY_KEY,
    queryFn: listAgentsPageData,
    staleTime: 30_000,
    gcTime: 5 * 60_000,
  });
  const [creating, setCreating] = useState(false);
  const [creatingFolder, setCreatingFolder] = useState(false);
  const [error, setError] = useState("");
  const [showFolderForm, setShowFolderForm] = useState(false);
  const [newFolderName, setNewFolderName] = useState("");
  const [activeFilter, setActiveFilter] = useState<AgentFilter>("all");
  const [activeFolderId, setActiveFolderId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(false);
  const agents = data?.agents ?? [];
  const folders = data?.folders ?? [];

  const activeFolder = folders.find((folder) => folder.id === activeFolderId) ?? null;
  const pageTitle = activeFilter === "folder" ? activeFolder?.name ?? "Folder" : filterTitles[activeFilter];
  const templateCount = agents.filter((agent) => matchesFilter(agent, "templates", null)).length;
  const transferCount = agents.filter((agent) => matchesFilter(agent, "transfer", null)).length;
  const filteredAgents = agents.filter(
    (agent) => matchesFilter(agent, activeFilter, activeFolderId) && matchesSearch(agent, searchQuery),
  );

  useEffect(() => {
    if (!loadError) return;
    if (loadError instanceof AuthRequiredError) {
      navigate("/auth");
      return;
    }
    setError(loadError instanceof Error ? loadError.message : "Failed to load agents");
  }, [loadError, navigate]);

  function selectFilter(filter: Exclude<AgentFilter, "folder">) {
    setActiveFilter(filter);
    setActiveFolderId(null);
  }

  function selectFolder(folderId: string) {
    setActiveFilter("folder");
    setActiveFolderId(folderId);
  }

  async function handleCreate() {
    setCreating(true);
    setError("");
    try {
      const bot = await createBot("New Vocal Agent", activeFilter === "folder" ? activeFolderId : null);
      navigate(`/agents/${bot.id}`);
    } catch (err: unknown) {
      if (err instanceof AuthRequiredError) {
        navigate("/auth");
        return;
      }
      setError(err instanceof Error ? err.message : "Failed to create agent");
    } finally {
      setCreating(false);
    }
  }

  async function handleCreateFolder(e: FormEvent) {
    e.preventDefault();
    const name = newFolderName.trim();
    if (!name) return;
    setCreatingFolder(true);
    setError("");
    try {
      const folder = await createAgentFolder(name);
      queryClient.setQueryData<AgentsPageData>(AGENTS_PAGE_QUERY_KEY, (current) => {
        if (!current) return { agents: [], folders: [folder] };
        return { ...current, folders: [...current.folders, folder] };
      });
      setNewFolderName("");
      setShowFolderForm(false);
      selectFolder(folder.id);
    } catch (err: unknown) {
      if (err instanceof AuthRequiredError) {
        navigate("/auth");
        return;
      }
      setError(err instanceof Error ? err.message : "Failed to create folder");
    } finally {
      setCreatingFolder(false);
    }
  }

  return (
    <div className="flex flex-1 min-h-0">
      {/* Folders rail */}
      <aside className={`${isSidebarCollapsed ? "hidden" : "hidden lg:flex"} flex-col w-64 border-r border-border bg-surface-muted/40 p-4 gap-3`}>
        <div className="flex items-center justify-between">
          <button
            type="button"
            onClick={() => selectFilter("all")}
            className={`text-sm font-medium flex items-center gap-1.5 px-2 py-1.5 rounded-md ${
              activeFilter === "all" ? "bg-secondary" : "hover:bg-secondary"
            }`}
          >
            <Bot className="w-4 h-4" /> All Agents
            <span className="ml-1 text-xs text-muted-foreground">{agents.length}</span>
          </button>
          <button
            type="button"
            onClick={() => setIsSidebarCollapsed(true)}
            className="p-1.5 rounded-md hover:bg-secondary"
            aria-label="Collapse agents sidebar"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
        </div>
        <div className="mt-2">
          <div className="flex items-center justify-between px-2 mb-1">
            <span className="text-[11px] tracking-wider font-semibold text-muted-foreground">FOLDERS</span>
            <button
              type="button"
              onClick={() => setShowFolderForm((visible) => !visible)}
              className="p-1 rounded hover:bg-secondary"
              aria-label="Create agent folder"
              disabled={creatingFolder}
            >
              <Plus className="w-3.5 h-3.5" />
            </button>
          </div>
          {showFolderForm && (
            <form onSubmit={handleCreateFolder} className="mb-2 rounded-md border border-border bg-card p-2">
              <input
                autoFocus
                value={newFolderName}
                onChange={(e) => setNewFolderName(e.target.value)}
                placeholder="Folder name"
                className="w-full rounded-md border border-border bg-background px-2 py-1.5 text-sm"
              />
              <div className="mt-2 flex gap-2">
                <Button type="submit" size="sm" className="h-7 px-2 text-xs" disabled={creatingFolder || !newFolderName.trim()}>
                  {creatingFolder ? "Creating..." : "Create"}
                </Button>
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  className="h-7 px-2 text-xs"
                  disabled={creatingFolder}
                  onClick={() => {
                    setShowFolderForm(false);
                    setNewFolderName("");
                  }}
                >
                  Cancel
                </Button>
              </div>
            </form>
          )}
          <button
            type="button"
            onClick={() => selectFilter("templates")}
            className={`w-full flex items-center gap-2 px-2 py-1.5 text-sm rounded-md ${
              activeFilter === "templates" ? "bg-secondary" : "hover:bg-secondary"
            }`}
          >
            <Folder className="w-4 h-4 text-muted-foreground" /> Template Agents
            <span className="ml-auto text-xs text-muted-foreground">{templateCount}</span>
          </button>
          {folders.map((folder) => {
            const folderCount = agents.filter((agent) => agent.folder_id === folder.id).length;
            return (
              <button
                key={folder.id}
                type="button"
                onClick={() => selectFolder(folder.id)}
                className={`mt-1 w-full flex items-center gap-2 px-2 py-1.5 text-sm rounded-md ${
                  activeFilter === "folder" && activeFolderId === folder.id ? "bg-secondary" : "hover:bg-secondary"
                }`}
              >
                <Folder className="w-4 h-4 text-muted-foreground" /> {folder.name}
                <span className="ml-auto text-xs text-muted-foreground">{folderCount}</span>
              </button>
            );
          })}
        </div>
        <div className="mt-2">
          <div className="px-2 mb-1 text-[11px] tracking-wider font-semibold text-muted-foreground">TRANSFER AGENTS</div>
          <button
            type="button"
            onClick={() => selectFilter("transfer")}
            className={`w-full flex items-center gap-2 px-2 py-1.5 text-sm rounded-md ${
              activeFilter === "transfer" ? "bg-secondary" : "hover:bg-secondary"
            }`}
          >
            <Folder className="w-4 h-4 text-muted-foreground" /> Transfer Screening Agents
            <span className="ml-auto text-xs text-muted-foreground">{transferCount}</span>
          </button>
        </div>
      </aside>

      <div className="flex-1 min-w-0 flex flex-col">
        {/* Header */}
        <header className="flex items-center justify-between gap-4 px-8 py-5 border-b border-border">
          <div className="flex items-center gap-2">
            {isSidebarCollapsed && (
              <button
                type="button"
                onClick={() => setIsSidebarCollapsed(false)}
                className="hidden lg:inline-flex p-1.5 rounded-md hover:bg-secondary"
                aria-label="Expand agents sidebar"
              >
                <ChevronLeft className="w-4 h-4 rotate-180" />
              </button>
            )}
            <h1 className="text-xl font-semibold tracking-tight">{pageTitle}</h1>
          </div>
          <div className="flex items-center gap-3">
            <div className="relative w-72 hidden md:block">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <input
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search…"
                className="w-full h-9 rounded-full bg-surface-muted border border-border pl-9 pr-3 text-sm focus:outline-none focus:ring-2 focus:ring-ring/40"
              />
            </div>
            <Button variant="outline" size="sm" className="gap-1.5">
              <Upload className="w-4 h-4" /> Import
            </Button>
            <Button
              size="sm"
              className="gap-1.5 bg-gradient-primary hover:opacity-90 shadow-elegant text-primary-foreground"
              onClick={handleCreate}
              disabled={creating}
            >
              <Plus className="w-4 h-4" /> {creating ? "Creating..." : "Create an Agent"}
            </Button>
          </div>
        </header>

        {/* Table */}
        <div className="flex-1 overflow-auto px-8 py-6">
          {error && (
            <div className="mb-4 rounded-lg border border-destructive/20 bg-destructive/10 px-4 py-3 text-sm text-destructive">
              {error}
            </div>
          )}
          <div className="rounded-xl border border-border overflow-hidden bg-card shadow-soft">
            <table className="w-full text-sm">
              <thead className="bg-surface-muted/60 text-muted-foreground">
                <tr className="text-left">
                  <th className="px-5 py-3 font-medium">Agent Name</th>
                  <th className="px-5 py-3 font-medium">Agent Type</th>
                  <th className="px-5 py-3 font-medium">Voice</th>
                  <th className="px-5 py-3 font-medium">Phone</th>
                  <th className="px-5 py-3 font-medium">Edited by</th>
                  <th className="w-10" />
                </tr>
              </thead>
              <tbody>
                {loading && (
                  <tr>
                    <td className="px-5 py-8 text-center text-muted-foreground" colSpan={6}>
                      Loading agents...
                    </td>
                  </tr>
                )}
                {!loading && agents.length === 0 && (
                  <tr>
                    <td className="px-5 py-8 text-center text-muted-foreground" colSpan={6}>
                      No agents yet. Create one to start building a LangChain-backed voice flow.
                    </td>
                  </tr>
                )}
                {!loading && agents.length > 0 && filteredAgents.length === 0 && (
                  <tr>
                    <td className="px-5 py-8 text-center text-muted-foreground" colSpan={6}>
                      No agents match this view.
                    </td>
                  </tr>
                )}
                {!loading && filteredAgents.map((a) => {
                  const type = botExecutionLabel(a.conversation_spec);
                  return (
                  <tr key={a.id} className="border-t border-border hover:bg-surface-muted/40 transition">
                    <td className="px-5 py-3.5">
                      <Link to={`/agents/${a.id}`} className="flex items-center gap-2.5 font-medium hover:text-primary">
                        <div className="w-7 h-7 rounded-md bg-gradient-soft flex items-center justify-center">
                          <Bot className="w-3.5 h-3.5 text-primary" />
                        </div>
                        {a.name}
                      </Link>
                    </td>
                    <td className="px-5 py-3.5">
                      <span className={`inline-flex items-center px-2 py-0.5 rounded-md text-xs font-medium ${typeStyles[type]}`}>
                        {type}
                      </span>
                    </td>
                    <td className="px-5 py-3.5">
                      <span className="inline-flex items-center gap-1.5">
                        <span className="w-5 h-5 rounded-full bg-gradient-primary" />
                        {a.elevenlabs_voice_id || "Cimo"}
                      </span>
                    </td>
                    <td className="px-5 py-3.5 text-muted-foreground">{a.twilio_phone_number || "-"}</td>
                    <td className="px-5 py-3.5 text-muted-foreground">
                      {a.updated_at ? new Date(a.updated_at).toLocaleString() : "-"}
                    </td>
                    <td className="px-3"><button className="p-1.5 rounded-md hover:bg-secondary"><MoreHorizontal className="w-4 h-4" /></button></td>
                  </tr>
                )})}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
