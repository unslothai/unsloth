// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** jsonl body with every record newline-terminated for clean concatenation. */
export function ndjsonBody(records: readonly string[]): string {
  return records.length > 0 ? `${records.join("\n")}\n` : "";
}

export type ConversationJsonlLayout = "training" | "messages";

export function canMergeConversationExport(format: string): boolean {
  return format !== "jsonl-messages";
}

export function exportFormatIncludesSiblings(format: string): boolean {
  return format !== "jsonl-raw" && format !== "jsonl-messages";
}

const OPENAI_MESSAGE_ROLES = new Set([
  "system",
  "developer",
  "user",
  "assistant",
  "tool",
]);

export function isOpenAIMessageRecord(
  record: unknown,
): record is Record<string, unknown> {
  if (typeof record !== "object" || record === null || Array.isArray(record)) {
    return false;
  }
  const message = record as Record<string, unknown>;
  return (
    typeof message.role === "string" &&
    OPENAI_MESSAGE_ROLES.has(message.role) &&
    ("content" in message || Array.isArray(message.tool_calls))
  );
}

export function messageJsonlConversationRecord(
  records: readonly unknown[],
): { messages: Record<string, unknown>[] } | null {
  if (records.length === 0 || !records.every(isOpenAIMessageRecord)) {
    return null;
  }
  return { messages: [...records] };
}

export function conversationJsonlBody(
  messages: readonly unknown[],
  layout: ConversationJsonlLayout,
): string {
  const records = layout === "training" ? [{ messages }] : messages;
  return records.map((record) => JSON.stringify(record)).join("\n");
}
