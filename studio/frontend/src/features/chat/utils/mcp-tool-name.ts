// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const MCP_TOOL_PREFIX = "mcp__";

export function isMcpToolName(toolName: string | undefined): boolean {
  return typeof toolName === "string" && toolName.startsWith(MCP_TOOL_PREFIX);
}

function provenanceString(
  provenance: unknown,
  key: "mcp_server" | "mcp_tool",
): string | undefined {
  if (!provenance || typeof provenance !== "object") return undefined;
  const value = (provenance as Record<string, unknown>)[key];
  return typeof value === "string" && value ? value : undefined;
}

export function mcpServerFromProvenance(provenance: unknown): string | undefined {
  return provenanceString(provenance, "mcp_server");
}

/** The raw MCP tool name; an aliased tool's composed name no longer spells it. */
export function mcpToolFromProvenance(provenance: unknown): string | undefined {
  return provenanceString(provenance, "mcp_tool");
}

export function formatMcpToolName(
  toolName: string,
  mcpServer?: string,
  mcpTool?: string,
): string | null {
  const parts = splitMcpToolName(toolName);
  return parts
    ? `${mcpServer || parts.serverId} · ${mcpTool || parts.tool}`
    : null;
}

export function splitMcpToolName(
  toolName: string,
): { serverId: string; tool: string } | null {
  if (!toolName.startsWith(MCP_TOOL_PREFIX)) return null;
  const rest = toolName.slice(MCP_TOOL_PREFIX.length);
  const sep = rest.indexOf("__");
  return sep > 0
    ? { serverId: rest.slice(0, sep), tool: rest.slice(sep + 2) }
    : null;
}
