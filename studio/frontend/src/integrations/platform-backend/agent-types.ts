export type PlatformAgentDsl = Record<string, unknown>;

export interface PlatformAgent {
  id: string;
  title?: string | null;
  description?: string | null;
  permission?: string;
  user_id?: string;
  tenant_id?: string;
  nickname?: string;
  tags?: string[];
  canvas_type?: string | null;
  canvas_category?: string;
  release?: boolean;
  dsl?: PlatformAgentDsl;
  create_time?: number | null;
  update_time?: number | null;
  release_time?: number | null;
  last_publish_time?: number | null;
}

export interface PlatformAgentListResult {
  items: PlatformAgent[];
  total: number;
}

export interface PlatformAgentVersion {
  id: string;
  user_canvas_id?: string;
  title?: string | null;
  description?: string | null;
  dsl?: PlatformAgentDsl;
  release?: boolean;
  create_time?: number | null;
  update_time?: number | null;
}

export interface PlatformAgentSession {
  id: string;
  dialog_id?: string;
  name?: string;
  message?: unknown[];
  reference?: unknown[];
  user_id?: string;
  create_time?: number | null;
  update_time?: number | null;
}

export interface PlatformAgentComponent {
  name: string;
  category?: string;
  inputs?: Record<string, unknown>;
  outputs?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface PlatformAgentUpload {
  id?: string;
  name?: string;
  filename?: string;
  url?: string;
  [key: string]: unknown;
}

export interface PlatformAgentCompletionRequest {
  agentId: string;
  query: string;
  sessionId?: string;
  inputs?: Record<string, unknown>;
  files?: unknown[][];
  returnTrace?: boolean;
}

export interface PlatformAgentRunRequest {
  agentId: string;
  userInput: string;
  sessionId?: string;
  version?: string;
}

export type PlatformAgentStreamEvent =
  | {
      type: "event";
      event: string;
      data: unknown;
      messageId: string | null;
      sessionId: string | null;
    }
  | { type: "done"; messageId: string | null; sessionId: string | null }
  | { type: "error"; message: string; code?: number | string };

export interface PlatformMcpServer {
  id: string;
  name: string;
  tenant_id?: string;
  url: string;
  server_type: string;
  description?: string | null;
  variables?: Record<string, unknown>;
  headers?: Record<string, unknown>;
  create_time?: number | null;
  update_time?: number | null;
  create_date?: string | null;
  update_date?: string | null;
}

export interface PlatformMcpServerInput {
  name: string;
  url: string;
  server_type: string;
  description?: string;
  variables?: Record<string, unknown>;
  headers?: Record<string, unknown>;
  timeout?: number;
}

export interface PlatformPluginToolParameter {
  type?: string;
  description?: string;
  displayDescription?: string;
  required?: boolean;
  [key: string]: unknown;
}

export interface PlatformPluginTool {
  name: string;
  displayName?: string;
  description?: string;
  displayDescription?: string;
  parameters?: Record<string, PlatformPluginToolParameter>;
  [key: string]: unknown;
}

export const EMPTY_PLATFORM_AGENT_DSL: PlatformAgentDsl = {
  components: {
    begin: {
      obj: { component_name: "Begin", params: {} },
      downstream: ["message"],
      upstream: [],
    },
    message: {
      obj: {
        component_name: "Message",
        params: { content: ["{sys.query}"] },
      },
      downstream: [],
      upstream: ["begin"],
    },
  },
  history: [],
  retrieval: [],
  path: [],
  globals: {
    "sys.query": "",
    "sys.user_id": "",
    "sys.conversation_turns": 0,
    "sys.files": [],
  },
  variables: {},
};

const SECRET_KEY =
  /password|secret|token|authorization|api[_-]?key|credential|headers?/i;

/** Removes credential-bearing values before they can reach diagnostics or UI output. */
export function redactAgentSecrets(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(redactAgentSecrets);
  if (!value || typeof value !== "object") return value;
  return Object.fromEntries(
    Object.entries(value as Record<string, unknown>).map(([key, item]) => [
      key,
      SECRET_KEY.test(key) ? "<redacted>" : redactAgentSecrets(item),
    ]),
  );
}
