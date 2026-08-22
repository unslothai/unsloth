// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { formatFastApiDetail } from "@/lib/format-fastapi-error";

export interface McpServerConfig {
  id: string;
  display_name: string;
  url: string;
  headers: Record<string, string>;
  is_enabled: boolean;
  use_oauth: boolean;
  created_at: string;
  updated_at: string;
}

export interface McpServerProbeResult {
  ok: boolean;
  tool_count: number;
  error: string | null;
}

export interface McpServerImportResult {
  created: McpServerConfig[];
  skipped: string[];
  errors: string[];
}

function parseErrorText(status: number, body: unknown): string {
  if (body && typeof body === "object") {
    const { detail, message } = body as { detail?: unknown; message?: unknown };
    const formatted = formatFastApiDetail(detail);
    if (formatted) return formatted;
    if (typeof message === "string" && message) return message;
  }
  return `Request failed (${status})`;
}

async function mcpRequest<T>(
  path: string,
  init?: { method?: string; body?: object },
): Promise<T> {
  const response = await authFetch(`/api/mcp/servers${path}`, {
    method: init?.method,
    headers: init?.body ? { "Content-Type": "application/json" } : undefined,
    body: init?.body ? JSON.stringify(init.body) : undefined,
  });
  // 204 No Content (DELETE) has no body — calling .json() would throw.
  if (response.status === 204) return undefined as T;
  const json = await response.json().catch(() => null);
  if (!response.ok) throw new Error(parseErrorText(response.status, json));
  return json as T;
}

export function listMcpServers(): Promise<McpServerConfig[]> {
  return mcpRequest("/");
}

export function createMcpServer(payload: {
  displayName: string;
  url: string;
  headers?: Record<string, string>;
  isEnabled?: boolean;
  useOauth?: boolean;
}): Promise<McpServerConfig> {
  return mcpRequest("/", {
    method: "POST",
    body: {
      display_name: payload.displayName,
      url: payload.url,
      headers: payload.headers ?? null,
      is_enabled: payload.isEnabled ?? true,
      use_oauth: payload.useOauth ?? false,
    },
  });
}

export function updateMcpServer(
  serverId: string,
  payload: {
    displayName?: string;
    url?: string;
    /** null = drop stored headers; omit to leave as-is */
    headers?: Record<string, string> | null;
    isEnabled?: boolean;
    useOauth?: boolean;
  },
): Promise<McpServerConfig> {
  const body: Record<string, unknown> = {};
  if (payload.displayName !== undefined) body.display_name = payload.displayName;
  if (payload.url !== undefined) body.url = payload.url;
  if (payload.headers !== undefined) body.headers = payload.headers;
  if (payload.isEnabled !== undefined) body.is_enabled = payload.isEnabled;
  if (payload.useOauth !== undefined) body.use_oauth = payload.useOauth;
  return mcpRequest(`/${serverId}`, { method: "PUT", body });
}

export function deleteMcpServer(serverId: string): Promise<void> {
  return mcpRequest(`/${serverId}`, { method: "DELETE" });
}

export function refreshMcpServerTools(
  serverId: string,
): Promise<McpServerProbeResult> {
  return mcpRequest(`/${serverId}/refresh`, { method: "POST" });
}

export function testMcpServer(payload: {
  url: string;
  headers?: Record<string, string>;
  useOauth?: boolean;
}): Promise<McpServerProbeResult> {
  return mcpRequest("/test", {
    method: "POST",
    body: {
      url: payload.url,
      headers: payload.headers ?? null,
      use_oauth: payload.useOauth ?? false,
    },
  });
}

// Bulk-import servers from a standard mcpServers JSON config (Claude Desktop,
// Cursor, Cline, VS Code). The backend skips duplicates and reports per-entry
// errors instead of failing the whole batch.
export function importMcpServers(
  config: unknown,
): Promise<McpServerImportResult> {
  return mcpRequest("/import", { method: "POST", body: { config } });
}

/** A ui:// template and the sandbox settings it declared in _meta.ui. */
export interface McpUiResource {
  uri: string;
  mime_type: string;
  text: string;
  ui: {
    csp?: {
      connectDomains?: string[];
      resourceDomains?: string[];
      frameDomains?: string[];
      baseUriDomains?: string[];
    };
    prefersBorder?: boolean;
    domain?: string;
  };
}

export interface McpUiToolCallResult {
  content: { type?: string; text?: string; [key: string]: unknown }[];
  structured_content: Record<string, unknown> | null;
  is_error: boolean;
  meta: Record<string, unknown> | null;
}

/** Fetch the widget template a tool result points at. */
export function readMcpUiResource(
  serverId: string,
  uri: string,
  scope?: { threadId?: string; sessionId?: string },
): Promise<McpUiResource> {
  const query = new URLSearchParams({ uri });
  if (scope?.threadId) query.set("thread_id", scope.threadId);
  if (scope?.sessionId) query.set("session_id", scope.sessionId);
  return mcpRequest(`/${serverId}/ui-resource?${query.toString()}`);
}

/**
 * Relay a tool call a widget asked for. `serverId` comes from the tool part that
 * drew the frame, never from the widget's own message.
 */
export function callMcpUiTool(
  serverId: string,
  payload: {
    toolName: string;
    arguments?: Record<string, unknown>;
    threadId?: string;
    sessionId?: string;
  },
): Promise<McpUiToolCallResult> {
  return mcpRequest(`/${serverId}/ui-tool-call`, {
    method: "POST",
    body: {
      tool_name: payload.toolName,
      arguments: payload.arguments ?? {},
      thread_id: payload.threadId ?? null,
      session_id: payload.sessionId ?? null,
    },
  });
}
