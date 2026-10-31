// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  SANDBOX_FILE_TOOLS,
  isSandboxToolResult,
} from "@/components/assistant-ui/sandbox-files";

import type { MessageRecord } from "../types";

/** Moving a chat changes only `projectId`, so use the session ids recorded on tool results. */
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
