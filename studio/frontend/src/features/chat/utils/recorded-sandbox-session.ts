// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { MessageRecord } from "../types";

/**
 * The sandbox a chat's tool calls actually wrote to, read back from its own
 * history.
 *
 * Moving a chat into or out of a project rewrites only its `projectId`, while
 * the files stay where they were written, so a scope derived from current
 * membership can name a folder this chat never used. The session id recorded on
 * a tool result is the one its files are under.
 */
export function recordedSandboxSessionId(
  messages: MessageRecord[],
): string | undefined {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const content = messages[index]?.content;
    if (!Array.isArray(content)) continue;
    for (let part = content.length - 1; part >= 0; part -= 1) {
      const result = (content[part] as { result?: unknown } | null)?.result;
      if (!result || typeof result !== "object") continue;
      const sessionId = (result as { sessionId?: unknown }).sessionId;
      if (typeof sessionId === "string" && sessionId.length > 0) {
        return sessionId;
      }
    }
  }
  return undefined;
}
