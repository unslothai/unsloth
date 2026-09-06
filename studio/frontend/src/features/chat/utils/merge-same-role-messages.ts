// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  OpenAIChatMessage,
  OpenAIMessageContent,
} from "../types/api";

/**
 * Merge consecutive messages that share the same role into one.
 *
 * Canceling a run keeps its user message without an assistant reply, so
 * re-sending the same prompt produced [user, user] payloads that strict
 * Jinja chat templates reject ("conversation roles must alternate").
 *
 * Assistant turns carrying tool_calls are never merged: the matching
 * role="tool" results that follow key off each assistant turn's call ids,
 * and collapsing the calls would orphan them. role="tool" messages pass
 * through untouched for the same reason. Text parts join with a blank
 * line; non-text parts are preserved in order.
 */
export function mergeConsecutiveSameRoleMessages(
  messages: OpenAIChatMessage[],
): OpenAIChatMessage[] {
  const merged: OpenAIChatMessage[] = [];
  const carriesToolCalls = (m: OpenAIChatMessage): boolean =>
    m.role === "assistant" && Boolean(m.tool_calls?.length);
  for (const message of messages) {
    const previous = merged[merged.length - 1];
    const mergeable =
      previous &&
      previous.role === message.role &&
      message.role !== "tool" &&
      !carriesToolCalls(previous) &&
      !carriesToolCalls(message);
    if (!mergeable) {
      merged.push({ ...message });
      continue;
    }
    const contentParts = (
      content: OpenAIMessageContent | null | undefined,
    ): Extract<OpenAIMessageContent, unknown[]>[number][] =>
      typeof content === "string"
        ? content
          ? [{ type: "text", text: content }]
          : []
        : [...(content ?? [])];
    // Merge newest-first so a long same-role run collapses in one pass.
    const parts = [
      ...contentParts(previous.content),
      ...contentParts(message.content),
    ];
    for (let i = parts.length - 1; i > 0; i -= 1) {
      const part = parts[i];
      const last = parts[i - 1];
      if (part.type === "text" && last.type === "text") {
        const separator = last.text.trim() && part.text.trim() ? "\n\n" : "";
        last.text = `${last.text}${separator}${part.text}`;
        parts.splice(i, 1);
      }
    }
    previous.content =
      parts.length === 0
        ? ""
        : parts.length === 1 && parts[0].type === "text"
          ? parts[0].text
          : parts;
  }
  return merged;
}
