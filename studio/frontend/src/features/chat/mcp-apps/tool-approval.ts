// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Matches the chat adapter's keys so "Always allow" covers model and widget calls alike.

export function mcpAppApprovalScope(
  sessionId: string | undefined,
  threadId: string | undefined,
): string {
  const session = sessionId || "_default";
  return threadId ? `${session}:${threadId}` : session;
}

export function mcpAppToolKey(serverId: string, toolName: string): string {
  return `mcp__${serverId}__${toolName}`;
}

export const MCP_APP_TOOL_DECLINED = "The user declined to run this tool call.";

const ARGS_PREVIEW_CHARS = 600;

export function mcpAppArgsPreview(args: Record<string, unknown>): string {
  if (Object.keys(args).length === 0) return "";
  let text: string;
  try {
    text = JSON.stringify(args, null, 2);
  } catch {
    return "";
  }
  return text.length > ARGS_PREVIEW_CHARS
    ? `${text.slice(0, ARGS_PREVIEW_CHARS)}…`
    : text;
}
