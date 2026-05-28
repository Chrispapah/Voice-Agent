export type HttpMethod = "GET" | "POST" | "PUT" | "PATCH" | "DELETE";
export type PreToolSpeech = "auto" | "force" | "disabled";
export type ExecutionMode = "default" | "blocking";
export type ToolCallSound = "none" | "click" | "custom_url";
export type AuthType = "none" | "bearer" | "basic" | "api_key_header" | "connection";

export interface ToolHeader {
  name: string;
  value: string;
}

export interface ToolParameterDef {
  name: string;
  description: string;
  type: "string" | "number" | "integer" | "boolean";
  required: boolean;
}

export interface ToolAuthConfig {
  type: AuthType;
  bearer_token?: string | null;
  basic_username?: string | null;
  basic_password?: string | null;
  api_key_header_name?: string | null;
  api_key_value?: string | null;
  connection_id?: string | null;
}

export interface HttpToolConfigV1 {
  schema_version: 1;
  method: HttpMethod;
  url: string;
  response_timeout_seconds: number;
  disable_interruptions: boolean;
  pre_tool_speech: PreToolSpeech;
  pre_tool_speech_text?: string | null;
  execution_mode: ExecutionMode;
  tool_call_sound: ToolCallSound;
  tool_call_sound_url?: string | null;
  auth: ToolAuthConfig;
  headers: ToolHeader[];
  path_parameters: ToolParameterDef[];
  query_parameters: ToolParameterDef[];
  parameters?: Record<string, unknown> | null;
}

export function defaultHttpToolConfig(): HttpToolConfigV1 {
  return {
    schema_version: 1,
    method: "POST",
    url: "",
    response_timeout_seconds: 20,
    disable_interruptions: false,
    pre_tool_speech: "auto",
    pre_tool_speech_text: null,
    execution_mode: "default",
    tool_call_sound: "none",
    tool_call_sound_url: null,
    auth: { type: "none" },
    headers: [],
    path_parameters: [],
    query_parameters: [],
    parameters: {
      type: "object",
      properties: {},
      required: [],
      additionalProperties: false,
    },
  };
}

export function parseToolConfig(raw: Record<string, unknown> | null | undefined): HttpToolConfigV1 {
  if (!raw || Object.keys(raw).length === 0) {
    return defaultHttpToolConfig();
  }
  if (raw.schema_version === 1) {
    return { ...defaultHttpToolConfig(), ...(raw as unknown as HttpToolConfigV1), schema_version: 1 };
  }
  const url = typeof raw.url === "string" ? raw.url : typeof raw.endpoint_url === "string" ? raw.endpoint_url : "";
  const method = (typeof raw.method === "string" ? raw.method : "POST").toUpperCase() as HttpMethod;
  return {
    ...defaultHttpToolConfig(),
    method: ["GET", "POST", "PUT", "PATCH", "DELETE"].includes(method) ? method : "POST",
    url,
  };
}

export function configToJson(config: HttpToolConfigV1): Record<string, unknown> {
  return { ...config };
}

export function extractPathParamNames(url: string): string[] {
  const matches = url.matchAll(/\{([a-zA-Z_][a-zA-Z0-9_]*)\}/g);
  const names: string[] = [];
  for (const m of matches) {
    if (!names.includes(m[1])) names.push(m[1]);
  }
  return names;
}

export function syncPathParameters(url: string, existing: ToolParameterDef[]): ToolParameterDef[] {
  const names = extractPathParamNames(url);
  const byName = new Map(existing.map((p) => [p.name, p]));
  return names.map(
    (name) =>
      byName.get(name) ?? {
        name,
        description: "",
        type: "string",
        required: true,
      },
  );
}

export function validateParametersJson(text: string): string | null {
  try {
    const parsed = JSON.parse(text) as Record<string, unknown>;
    if (parsed.type !== "object") {
      return "JSON Schema must have type: object";
    }
    return null;
  } catch {
    return "Invalid JSON";
  }
}
