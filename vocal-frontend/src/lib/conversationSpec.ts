export const SINGLE_AGENT_NODE_ID = "__single__" as const;

export type ConversationSpecMode = "single" | "graph";
export type ConversationSpecTemplate = "custom" | "sdr";

export interface SpecNode {
  id: string;
  label?: string | null;
  system_prompt: string;
  tool_ids?: string[];
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

export function botExecutionLabel(spec?: ConversationSpecV1 | null): string {
  if (spec == null || spec.template === "sdr") return "Classic SDR";
  if (spec.mode === "single") return "Single Prompt";
  return "Conversation Flow";
}
