// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { formatFastApiDetail } from "@/lib/format-fastapi-error";

import {
  getMcpServerMutationEpoch,
  readAfterPendingMcpServerMutations,
  readMcpServerMutationSnapshot,
  trackMcpServerMutation,
} from "./mcp-server-mutation-tracker";

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

export interface McpStdioCommand {
  command: string;
  arguments: string[];
}

let mcpServerListRequest: Promise<McpServerConfig[]> | null = null;
let mcpServerSettlementListRequest: {
  minimumEpoch: number;
  promise: Promise<McpServerConfig[]>;
} | null = null;

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

export function listMcpServers({
  waitForPendingMutations = true,
  minimumMutationEpoch,
}: {
  waitForPendingMutations?: boolean;
  minimumMutationEpoch?: number;
} = {}): Promise<McpServerConfig[]> {
  if (!waitForPendingMutations) {
    const requestedEpoch = minimumMutationEpoch ?? getMcpServerMutationEpoch();
    if (
      mcpServerSettlementListRequest &&
      mcpServerSettlementListRequest.minimumEpoch >= requestedEpoch
    ) {
      return mcpServerSettlementListRequest.promise;
    }
    const request = readMcpServerMutationSnapshot(() =>
      mcpRequest<McpServerConfig[]>("/"),
    );
    const slot = { minimumEpoch: requestedEpoch, promise: request };
    mcpServerSettlementListRequest = slot;
    void request.then(
      () => {
        if (mcpServerSettlementListRequest === slot) {
          mcpServerSettlementListRequest = null;
        }
      },
      () => {
        if (mcpServerSettlementListRequest === slot) {
          mcpServerSettlementListRequest = null;
        }
      },
    );
    return request;
  }

  if (mcpServerListRequest) return mcpServerListRequest;

  const request = readAfterPendingMcpServerMutations(() =>
    mcpRequest<McpServerConfig[]>("/"),
  );
  mcpServerListRequest = request;
  void request.then(
    () => {
      if (mcpServerListRequest === request) mcpServerListRequest = null;
    },
    () => {
      if (mcpServerListRequest === request) mcpServerListRequest = null;
    },
  );
  return request;
}

export function createMcpServer(payload: {
  displayName: string;
  url: string;
  headers?: Record<string, string>;
  isEnabled?: boolean;
  useOauth?: boolean;
}): Promise<McpServerConfig> {
  return trackMcpServerMutation(
    mcpRequest("/", {
      method: "POST",
      body: {
        display_name: payload.displayName,
        url: payload.url,
        headers: payload.headers ?? null,
        is_enabled: payload.isEnabled ?? true,
        use_oauth: payload.useOauth ?? false,
      },
    }),
  );
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
  if (payload.displayName !== undefined)
    body.display_name = payload.displayName;
  if (payload.url !== undefined) body.url = payload.url;
  if (payload.headers !== undefined) body.headers = payload.headers;
  if (payload.isEnabled !== undefined) body.is_enabled = payload.isEnabled;
  if (payload.useOauth !== undefined) body.use_oauth = payload.useOauth;
  return trackMcpServerMutation(
    mcpRequest(`/${serverId}`, { method: "PUT", body }),
  );
}

export function deleteMcpServer(serverId: string): Promise<void> {
  return trackMcpServerMutation(
    mcpRequest(`/${serverId}`, { method: "DELETE" }),
  );
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

export function decodeMcpStdioCommand(url: string): Promise<McpStdioCommand> {
  return mcpRequest("/stdio/decode", {
    method: "POST",
    body: { url },
  });
}

export function encodeMcpStdioCommand(payload: McpStdioCommand): Promise<{
  url: string;
}> {
  return mcpRequest("/stdio/encode", {
    method: "POST",
    body: payload,
  });
}

// Bulk-import servers from a standard mcpServers JSON config (Claude Desktop, Cursor, Cline, VS
// Code). The backend skips duplicates and reports per-entry errors instead of failing the batch.
export function importMcpServers(
  config: unknown,
): Promise<McpServerImportResult> {
  return trackMcpServerMutation(
    mcpRequest("/import", { method: "POST", body: { config } }),
  );
}
