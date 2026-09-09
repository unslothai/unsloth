// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const MCP_TOOL_PREFIX = "mcp__";

/** The server display name stamped into tool-call provenance by the backend. */
export function mcpServerFromProvenance(provenance: unknown): string | undefined {
  if (!provenance || typeof provenance !== "object") return undefined;
  const value = (provenance as { mcp_server?: unknown }).mcp_server;
  return typeof value === "string" && value ? value : undefined;
}

/** "GitHub · create_issue" for mcp__<serverId>__<tool>, else null. Falls back to the raw id. */
export function formatMcpToolName(
  toolName: string,
  mcpServer?: string,
): string | null {
  if (!toolName.startsWith(MCP_TOOL_PREFIX)) return null;
  const rest = toolName.slice(MCP_TOOL_PREFIX.length);
  const sep = rest.indexOf("__");
  if (sep <= 0) return null;
  return `${mcpServer || rest.slice(0, sep)} · ${rest.slice(sep + 2)}`;
}
