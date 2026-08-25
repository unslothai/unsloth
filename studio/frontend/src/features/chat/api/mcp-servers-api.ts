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
