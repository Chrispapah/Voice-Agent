/**
 * Mirrors `ai_sdr_agent.graph.spec` (v1). Router rules are documented in that module.
 */
export const SINGLE_AGENT_NODE_ID = "__single__" as const;

export type ConversationSpecMode = "single" | "graph";
export type ConversationSpecTemplate = "custom" | "sdr";

export interface SpecNode {
  id: string;
  label?: string | null;
  system_prompt: string;
  /**
   * When non-empty, overrides the reply LLM for turns after the user speaks.
   * Opening line still uses bot Initial greeting. Same template variables as system_prompt.
   */
  static_message?: string | null;
  loop_min_turns?: number | null;
  loop_max_turns?: number | null;
  /** Router LLM hint when choosing among outbound edges (not spoken). */
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
  entry_node_id?: string | null;
  nodes?: SpecNode[];
  edges?: SpecEdge[];
  variables_hint?: string | null;
  /** Optional React Flow positions keyed by node id */
  positions?: Record<string, { x: number; y: number }>;
}

export function defaultSdrConversationSpec(): ConversationSpecV1 {
  return {
    conversation_spec_version: 1,
    mode: "graph",
    template: "sdr",
    nodes: [],
    edges: [],
  };
}

export function defaultSingleConversationSpec(systemPrompt: string): ConversationSpecV1 {
  return {
    conversation_spec_version: 1,
    mode: "single",
    template: "custom",
    system_prompt: systemPrompt,
  };
}

export function botExecutionLabel(spec?: ConversationSpecV1 | null): string {
  if (spec == null || spec.template === "sdr") return "Classic SDR";
  if (spec.mode === "single") return "Single agent";
  return "Agent graph";
}

/** Built-in SDR funnel + stage prompts apply only in this mode (matches Agent builder "Classic SDR"). */
export function isClassicSdrBuiltInGraph(spec: ConversationSpecV1 | null | undefined): boolean {
  return spec == null || spec.template === "sdr";
}

export function defaultGraphConversationSpec(): ConversationSpecV1 {
  const a = "intro";
  const b = "main";
  return {
    conversation_spec_version: 1,
    mode: "graph",
    template: "custom",
    entry_node_id: a,
    nodes: [
      {
        id: a,
        label: "Introduction",
        system_prompt:
          "You are a voice agent introducing yourself briefly. Variables: {lead_name}, {company}, {lead_context}.",
      },
      {
        id: b,
        label: "Main",
        system_prompt:
          "You continue the conversation helpfully in one or two short sentences. Variables: {lead_name}, {company}.",
      },
    ],
    edges: [{ from: a, to: b }],
    positions: {
      [a]: { x: 80, y: 120 },
      [b]: { x: 380, y: 120 },
    },
  };
}
