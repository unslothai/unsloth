// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Search the logical conversation, not the subset mounted by the virtualizer.
 * Only user-visible text parts are indexed; non-text/tool HTML needs separate
 * semantic parity work before releasing the experimental mode.
 */
export type AgentFindMatch = Readonly<{ row: number; offset: number }>;
export type AgentFindResults = Readonly<{ matches: AgentFindMatch[]; capped: boolean }>;

function messageText(message: unknown): string {
  if (!message || typeof message !== "object") return "";
  const content = (message as { content?: unknown }).content;
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .filter((part: unknown) =>
      part !== null && typeof part === "object" &&
      (part as { type?: unknown }).type === "text" &&
      typeof (part as { text?: unknown }).text === "string")
    .map((part: unknown) => (part as { text: string }).text)
    .join("\n");
}

export function findAgentTextMatches(
  messages: readonly unknown[],
  query: string,
  limit = 5000,
): AgentFindResults {
  if (!query || !Number.isSafeInteger(limit) || limit < 1) return { matches: [], capped: false };
  const needle = query.toLocaleLowerCase();
  const matches: AgentFindMatch[] = [];
  for (let row = 0; row < messages.length; row++) {
    const haystack = messageText(messages[row]).toLocaleLowerCase();
    let pos = 0;
    while (pos < haystack.length) {
      const at = haystack.indexOf(needle, pos);
      if (at < 0) break;
      if (matches.length === limit) return { matches, capped: true };
      matches.push({ row, offset: at });
      pos = at + Math.max(1, needle.length);
    }
  }
  return { matches, capped: false };
}
