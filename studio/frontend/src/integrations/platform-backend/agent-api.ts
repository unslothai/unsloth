import { platformRequest } from "./client";
import type {
  PlatformAgent,
  PlatformAgentComponent,
  PlatformAgentDsl,
  PlatformAgentListResult,
  PlatformAgentSession,
  PlatformAgentUpload,
  PlatformAgentVersion,
  PlatformMcpServer,
  PlatformMcpServerInput,
  PlatformPluginTool,
} from "./agent-types";

const AGENT_TIMEOUT_MS = 180_000;
export const PLATFORM_AGENT_FILE_MAX_BYTES = 64 * 1024 * 1024;

const enc = encodeURIComponent;
const array = <T>(value: unknown): T[] =>
  Array.isArray(value) ? (value as T[]) : [];

export async function listAgents(
  options: { keywords?: string; page?: number; pageSize?: number } = {},
  signal?: AbortSignal,
): Promise<PlatformAgentListResult> {
  const data = await platformRequest<{ canvas?: unknown; total?: unknown }>(
    "/agents",
    {
      query: {
        page: options.page ?? 1,
        page_size: options.pageSize ?? 100,
        orderby: "update_time",
        desc: true,
        ...(options.keywords?.trim()
          ? { keywords: options.keywords.trim() }
          : {}),
      },
      signal,
    },
  );
  const items = array<PlatformAgent>(data?.canvas);
  const parsedTotal = Number(data?.total);
  return {
    items,
    total: Number.isFinite(parsedTotal) ? parsedTotal : items.length,
  };
}

export function createAgent(
  payload: { title: string; description?: string; dsl: PlatformAgentDsl },
  signal?: AbortSignal,
): Promise<PlatformAgent> {
  return platformRequest("/agents", { method: "POST", json: payload, signal });
}

export function getAgent(
  id: string,
  signal?: AbortSignal,
): Promise<PlatformAgent> {
  return platformRequest(`/agents/${enc(id)}`, { signal });
}

export function updateAgent(
  id: string,
  patch: Partial<
    Pick<PlatformAgent, "title" | "description" | "permission" | "dsl">
  >,
  signal?: AbortSignal,
): Promise<unknown> {
  return platformRequest(`/agents/${enc(id)}`, {
    method: "PUT",
    json: patch,
    signal,
  });
}

export function deleteAgent(
  id: string,
  signal?: AbortSignal,
): Promise<unknown> {
  return platformRequest(`/agents/${enc(id)}`, { method: "DELETE", signal });
}

export function updateAgentTags(
  id: string,
  tags: string[],
  signal?: AbortSignal,
) {
  return platformRequest(`/agents/${enc(id)}/tags`, {
    method: "PUT",
    json: { tags },
    signal,
  });
}

export function publishAgent(
  id: string,
  payload: { title?: string; description?: string; dsl?: PlatformAgentDsl },
  signal?: AbortSignal,
): Promise<PlatformAgentVersion> {
  return platformRequest(`/agents/${enc(id)}/publish`, {
    method: "POST",
    json: payload,
    signal,
  });
}

export function resetAgent(
  id: string,
  signal?: AbortSignal,
): Promise<PlatformAgentDsl> {
  return platformRequest(`/agents/${enc(id)}/reset`, {
    method: "POST",
    signal,
  });
}

/** Active v0.26.4 contract; current source replaces this with session-scoped task cancellation. */
export function cancelAgentRun(
  id: string,
  signal?: AbortSignal,
): Promise<unknown> {
  return platformRequest(`/agents/${enc(id)}/run`, {
    method: "DELETE",
    signal,
  });
}

export function cancelAgentSession(
  sessionId: string,
  signal?: AbortSignal,
): Promise<unknown> {
  return platformRequest(`/tasks/${enc(sessionId)}/cancel`, {
    method: "POST",
    signal,
  });
}

export function getAgentComponentInputForm(
  agentId: string,
  componentId: string,
  signal?: AbortSignal,
): Promise<Record<string, unknown>> {
  return platformRequest(
    `/agents/${enc(agentId)}/components/${enc(componentId)}/input-form`,
    { signal },
  );
}

export function debugAgentComponent(
  agentId: string,
  componentId: string,
  params: Record<string, { value: unknown }>,
  signal?: AbortSignal,
): Promise<Record<string, unknown>> {
  return platformRequest(
    `/agents/${enc(agentId)}/components/${enc(componentId)}/debug`,
    {
      method: "POST",
      json: { params },
      signal,
    },
  );
}

export async function listAgentSessions(
  agentId: string,
  signal?: AbortSignal,
): Promise<PlatformAgentSession[]> {
  const value = await platformRequest<unknown>(
    `/agents/${enc(agentId)}/sessions`,
    { signal },
  );
  if (Array.isArray(value)) return value as PlatformAgentSession[];
  if (value && typeof value === "object" && "sessions" in value) {
    return array<PlatformAgentSession>(
      (value as { sessions?: unknown }).sessions,
    );
  }
  return [];
}

export function createAgentSession(
  agentId: string,
  name: string,
  signal?: AbortSignal,
): Promise<PlatformAgentSession> {
  return platformRequest(`/agents/${enc(agentId)}/sessions`, {
    method: "POST",
    json: { name },
    signal,
  });
}

export function getAgentSession(
  agentId: string,
  sessionId: string,
  signal?: AbortSignal,
) {
  return platformRequest<PlatformAgentSession>(
    `/agents/${enc(agentId)}/sessions/${enc(sessionId)}`,
    { signal },
  );
}

export function deleteAgentSession(
  agentId: string,
  sessionId: string,
  signal?: AbortSignal,
) {
  return platformRequest(`/agents/${enc(agentId)}/sessions/${enc(sessionId)}`, {
    method: "DELETE",
    signal,
  });
}

export function deleteAgentSessions(
  agentId: string,
  options: { ids?: string[]; deleteAll?: boolean },
  signal?: AbortSignal,
) {
  return platformRequest(`/agents/${enc(agentId)}/sessions`, {
    method: "DELETE",
    query: {
      ...(options.ids?.length ? { ids: options.ids.join(",") } : {}),
      ...(options.deleteAll ? { delete_all: true } : {}),
    },
    signal,
  });
}

export async function listAgentVersions(id: string, signal?: AbortSignal) {
  const value = await platformRequest<unknown>(`/agents/${enc(id)}/versions`, {
    signal,
  });
  return array<PlatformAgentVersion>(value);
}

export function getAgentVersion(
  id: string,
  versionId: string,
  signal?: AbortSignal,
) {
  return platformRequest<PlatformAgentVersion>(
    `/agents/${enc(id)}/versions/${enc(versionId)}`,
    { signal },
  );
}

export function deleteAgentVersion(
  id: string,
  versionId: string,
  signal?: AbortSignal,
) {
  return platformRequest(`/agents/${enc(id)}/versions/${enc(versionId)}`, {
    method: "DELETE",
    signal,
  });
}

export function getAgentLogs(
  id: string,
  messageId: string,
  signal?: AbortSignal,
) {
  return platformRequest<unknown>(`/agents/${enc(id)}/logs/${enc(messageId)}`, {
    signal,
  });
}

export function getAgentWebhookLogs(id: string, signal?: AbortSignal) {
  return platformRequest<unknown>(`/agents/${enc(id)}/webhook/logs`, {
    signal,
  });
}

export function testAgentWebhook(
  id: string,
  payload: unknown,
  method: "GET" | "POST" | "PUT" | "PATCH" | "DELETE" | "HEAD" = "POST",
  signal?: AbortSignal,
) {
  return platformRequest<unknown>(`/agents/${enc(id)}/webhook/test`, {
    method,
    ...(["GET", "HEAD"].includes(method) ? {} : { json: payload }),
    signal,
    timeoutMs: AGENT_TIMEOUT_MS,
  });
}

export function testAgentDatabaseConnection(
  payload: {
    db_type: string;
    database: string;
    username: string;
    host: string;
    port: number;
    password: string;
  },
  signal?: AbortSignal,
) {
  return platformRequest<unknown>("/agents/test_db_connection", {
    method: "POST",
    json: payload,
    signal,
    timeoutMs: AGENT_TIMEOUT_MS,
  });
}

export function rerunAgentDocument(
  payload: { id: string; dsl: PlatformAgentDsl; component_id: string },
  signal?: AbortSignal,
) {
  return platformRequest<unknown>("/agents/rerun", {
    method: "POST",
    json: payload,
    signal,
    timeoutMs: AGENT_TIMEOUT_MS,
  });
}

export async function listAgentTemplates(signal?: AbortSignal) {
  const value = await platformRequest<unknown>("/agents/templates", { signal });
  return array<Record<string, unknown>>(value);
}

export function getAgentPrompts(signal?: AbortSignal) {
  return platformRequest<Record<string, string>>("/agents/prompts", {
    signal,
  });
}

export async function listAvailableAgentTags(signal?: AbortSignal) {
  const value = await platformRequest<unknown>("/agents/tags", { signal });
  return array<{ tag: string; count: number }>(value);
}

export async function listAgentComponents(signal?: AbortSignal) {
  const value = await platformRequest<unknown>("/components", { signal });
  return array<PlatformAgentComponent>(value);
}

export async function uploadAgentFiles(
  agentId: string,
  files: File[],
  signal?: AbortSignal,
) {
  const form = new FormData();
  files.forEach((file) => form.append("file", file, file.name));
  const value = await platformRequest<unknown>(
    `/agents/${enc(agentId)}/upload`,
    {
      method: "POST",
      body: form,
      signal,
      timeoutMs: AGENT_TIMEOUT_MS,
    },
  );
  return Array.isArray(value)
    ? (value as PlatformAgentUpload[])
    : [value as PlatformAgentUpload];
}

export function downloadAgentFile(
  id: string,
  signal?: AbortSignal,
): Promise<Blob> {
  return platformRequest("/agents/download", {
    query: { id },
    responseType: "blob",
    signal,
  });
}

export function previewAgentAttachment(
  attachmentId: string,
  options: { ext: string; mimeType: string; filename: string },
  signal?: AbortSignal,
): Promise<Blob> {
  return platformRequest(`/agents/attachments/${enc(attachmentId)}/preview`, {
    query: {
      ext: options.ext,
      mime_type: options.mimeType,
      filename: options.filename,
    },
    responseType: "blob",
    signal,
  });
}

export function downloadAgentAttachment(
  attachmentId: string,
  options: { ext: string; mimeType: string; filename: string },
  signal?: AbortSignal,
): Promise<Blob> {
  return platformRequest(`/agents/attachments/${enc(attachmentId)}/download`, {
    query: {
      ext: options.ext,
      mime_type: options.mimeType,
      filename: options.filename,
    },
    responseType: "blob",
    signal,
  });
}

export async function listMcpServers(signal?: AbortSignal) {
  const data = await platformRequest<{
    mcp_servers?: unknown;
    total?: unknown;
  }>("/mcp/servers", {
    query: { page: 1, page_size: 100, orderby: "create_time", desc: true },
    signal,
  });
  return {
    items: array<PlatformMcpServer>(data?.mcp_servers),
    total: Number(data?.total) || 0,
  };
}

export function createMcpServer(
  input: PlatformMcpServerInput,
  signal?: AbortSignal,
) {
  return platformRequest<PlatformMcpServer>("/mcp/servers", {
    method: "POST",
    json: input,
    signal,
  });
}

export function getMcpServer(id: string, signal?: AbortSignal) {
  return platformRequest<PlatformMcpServer>(`/mcp/servers/${enc(id)}`, {
    signal,
  });
}

export function updateMcpServer(
  id: string,
  input: Partial<PlatformMcpServerInput>,
  signal?: AbortSignal,
) {
  return platformRequest<PlatformMcpServer>(`/mcp/servers/${enc(id)}`, {
    method: "PUT",
    json: input,
    signal,
  });
}

export function deleteMcpServer(id: string, signal?: AbortSignal) {
  return platformRequest(`/mcp/servers/${enc(id)}`, {
    method: "DELETE",
    signal,
  });
}

export function importMcpServers(
  mcpServers: Record<string, Record<string, unknown>>,
  timeout = 10,
  signal?: AbortSignal,
) {
  return platformRequest<{ results?: unknown }>("/mcp/servers/import", {
    method: "POST",
    json: { mcpServers, timeout },
    signal,
    timeoutMs: AGENT_TIMEOUT_MS,
  });
}

export function testMcpServer(
  id: string,
  input: PlatformMcpServerInput,
  signal?: AbortSignal,
) {
  return platformRequest<unknown>(`/mcp/servers/${enc(id)}/test`, {
    method: "POST",
    json: input,
    signal,
    timeoutMs: AGENT_TIMEOUT_MS,
  });
}

export async function listPluginTools(signal?: AbortSignal) {
  const value = await platformRequest<unknown>("/plugin/tools", { signal });
  return array<PlatformPluginTool>(value);
}
