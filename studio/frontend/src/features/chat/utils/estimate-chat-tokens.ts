// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { MessageRecord } from "../types";
import { orderBySelectedBranch } from "./message-order.ts";

const RAG_SOURCES_TAIL = "\n__RAG_SOURCES__:";

function modelVisibleLength(resultText: string): number {
  const tail = resultText.lastIndexOf(RAG_SOURCES_TAIL);
  return tail < 0 ? resultText.length : tail;
}

function partChars(part: unknown): number {
  const value = part as {
    type?: unknown;
    text?: unknown;
    argsText?: unknown;
    result?: unknown;
    provenance?: { source?: unknown };
  } | null;
  if (typeof value?.text === "string") return value.text.length;
  if (value?.type !== "tool-call" || value.provenance?.source !== "local") return 0;
  const result = value.result as { text?: unknown } | string | null | undefined;
  const resultText =
    typeof result === "string" ? result : typeof result?.text === "string" ? result.text : "";
  return (
    (typeof value.argsText === "string" ? value.argsText.length : 0) +
    modelVisibleLength(resultText)
  );
}

export function estimateMessagesTokenCount(
  records: readonly MessageRecord[] | null | undefined,
): number | null {
  if (!records || records.length === 0) return null;
  let chars = 0;
  for (const record of orderBySelectedBranch(records.slice())) {
    const content = record.content as unknown;
    if (typeof content === "string") {
      chars += content.length;
      continue;
    }
    if (!Array.isArray(content)) continue;
    for (const part of content) chars += partChars(part);
  }
  return chars > 0 ? Math.max(1, Math.round(chars / 4)) : null;
}

// Text and Studio-run tool payloads, about 4 characters a token: shown until a loaded model counts the chat exactly.
export function estimateContextUsage(
  records: readonly MessageRecord[] | null | undefined,
) {
  const tokens = estimateMessagesTokenCount(records);
  if (tokens === null) return null;
  return {
    promptTokens: tokens,
    completionTokens: 0,
    totalTokens: tokens,
    cachedTokens: 0,
    estimated: true,
  };
}
