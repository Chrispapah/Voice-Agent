export const SINGLE_AGENT_NODE_ID = "__single__" as const;

export type ConversationSpecMode = "single" | "graph";
export type ConversationSpecTemplate = "custom" | "sdr";

export type ReplyTurnMode = "static" | "llm";

export interface SpecNode {
  id: string;
  label?: string | null;
  system_prompt: string;
  tool_ids?: string[];
  /**
   * Fixed text when this node's reply mode is "static". Same placeholders as system_prompt.
   */
  static_message?: string | null;
  /**
   * Ordered modes per agent utterance at this node (0 = opener before the user speaks).
   * If omitted with static_message set: after the user speaks, replies use static_message (legacy).
   */
  reply_turn_modes?: ReplyTurnMode[] | null;
  /** Minimum completed self-loop turns before classifier may route to a different node. */
  loop_min_turns?: number | null;
  /** After this many completed stays, force exit to a non-self neighbor. */
  loop_max_turns?: number | null;
  /** Shown only to the edge-router LLM when this node fans out (not in TTS prompts). */
  classify_hint?: string | null;
}

export interface SpecEdge {
  from: string;
  to: string;
}

export interface ConversationSpecV1 {
  conversation_spec_version: 1;
  mode: ConversationSpecMode;
  template?: ConversationSpecTemplate;
  system_prompt?: string | null;
  tool_ids?: string[];
  entry_node_id?: string | null;
  nodes?: SpecNode[];
  edges?: SpecEdge[];
  variables_hint?: string | null;
  positions?: Record<string, { x: number; y: number }>;
  /** Single mode: fixed text when a turn uses "static" in single_reply_turn_modes. */
  single_static_message?: string | null;
  /** Single mode: same semantics as per-node reply_turn_modes. */
  single_reply_turn_modes?: ReplyTurnMode[] | null;
}

export function defaultGraphConversationSpec(): ConversationSpecV1 {
  return {
    conversation_spec_version: 1,
    mode: "graph",
    template: "custom",
    entry_node_id: "welcome",
    nodes: [
      {
        id: "welcome",
        label: "Welcome",
        tool_ids: [],
        system_prompt:
          "You are a warm voice agent opening the call. Keep the reply to one or two short spoken sentences. Context: {lead_name}, {company}, {lead_context}.",
      },
      {
        id: "qualify",
        label: "Qualify",
        tool_ids: [],
        system_prompt:
          "Ask one concise qualifying question based on the caller's latest answer. Keep it natural for a phone conversation.",
      },
    ],
    edges: [{ from: "welcome", to: "qualify" }],
    positions: {
      welcome: { x: 220, y: 130 },
      qualify: { x: 560, y: 130 },
    },
  };
}

export function defaultSingleConversationSpec(systemPrompt: string): ConversationSpecV1 {
  return {
    conversation_spec_version: 1,
    mode: "single",
    template: "custom",
    system_prompt: systemPrompt,
    tool_ids: [],
  };
}

export function formatReplyTurnModes(modes: ReplyTurnMode[] | null | undefined): string {
  return modes?.length ? modes.join(", ") : "";
}

/** Parse comma/space-separated static|llm tokens; returns undefined if empty or invalid. */
export function parseReplyTurnModes(raw: string): ReplyTurnMode[] | undefined {
  const parts = raw
    .split(/[,;\s]+/)
    .map((s) => s.trim().toLowerCase())
    .filter(Boolean);
  if (!parts.length) return undefined;
  const out: ReplyTurnMode[] = [];
  for (const p of parts) {
    if (p === "static" || p === "llm") out.push(p);
    else return undefined;
  }
  return out;
}

export function botExecutionLabel(spec?: ConversationSpecV1 | null): string {
  if (spec == null || spec.template === "sdr") return "Classic SDR";
  if (spec.mode === "single") return "Single Prompt";
  return "Conversation Flow";
}
