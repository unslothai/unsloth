// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

type ToolHistoryPart = {
  type?: unknown;
  provenance?: unknown;
  [key: string]: unknown;
};

export type ToolHistoryMessage = {
  content?: readonly ToolHistoryPart[];
  [key: string]: unknown;
};

export type StudioToolHistoryOptions = {
  /** When set, only these tool-call parts count. Use this to ignore provider builtins that OpenAI
   *  serialization drops, so ownership matches the payload. */
  toolCallSurvives?: (part: ToolHistoryPart) => boolean;
};

export function hasOnlyStudioOwnedToolHistory(
  messages: readonly ToolHistoryMessage[],
  options?: StudioToolHistoryOptions,
): boolean {
  const survives = options?.toolCallSurvives;
  let sawToolCall = false;
  for (const message of messages) {
    for (const part of message.content ?? []) {
      if (part.type !== "tool-call") continue;
      if (survives && !survives(part)) continue;
      sawToolCall = true;
      const provenance = part.provenance;
      if (
        !provenance ||
        typeof provenance !== "object" ||
        Array.isArray(provenance) ||
        (provenance as { source?: unknown }).source !== "local"
      ) {
        return false;
      }
    }
  }
  return sawToolCall;
}

export function studioToolHistoryRequestFields(
  messages: readonly ToolHistoryMessage[],
  options?: StudioToolHistoryOptions,
): { studio_tool_history?: true } {
  return hasOnlyStudioOwnedToolHistory(messages, options)
    ? { studio_tool_history: true }
    : {};
}
