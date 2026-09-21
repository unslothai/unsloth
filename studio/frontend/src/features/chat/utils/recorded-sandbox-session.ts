// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  SANDBOX_FILE_TOOLS,
  isSandboxToolResult,
} from "@/components/assistant-ui/sandbox-files";

import type { MessageRecord } from "../types";

/** Every sandbox a chat's tool calls actually wrote to, read back from its history, newest first.
 *  Moving a chat between projects rewrites only its `projectId` while the files stay put, so a
 *  scope from current membership can name a folder this chat never used; the id recorded on a tool
 *  result is the one its files are under. A chat that ran tools on both sides of such a move names
 *  both, which is why this returns a list. Gated on the tool name and the full wrapper shape, as
 *  the adapter is: a custom or MCP tool with its own `sessionId` would name an unrelated folder. */
export function allRecordedSandboxSessionIds(
  messages: MessageRecord[],
): string[] {
  const found: string[] = [];
  const seen = new Set<string>();
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const content = messages[index]?.content;
    if (!Array.isArray(content)) continue;
    for (let part = content.length - 1; part >= 0; part -= 1) {
      const entry = content[part] as {
        type?: unknown;
        toolName?: unknown;
        result?: unknown;
      } | null;
      if (entry?.type !== "tool-call") continue;
      if (
        typeof entry.toolName !== "string" ||
        !SANDBOX_FILE_TOOLS.has(entry.toolName)
      ) {
        continue;
      }
      const result: unknown = entry.result;
      if (!isSandboxToolResult(result)) continue;
      if (result.sessionId.length === 0) continue;
      if (seen.has(result.sessionId)) continue;
      seen.add(result.sessionId);
      found.push(result.sessionId);
    }
  }
  return found;
}
