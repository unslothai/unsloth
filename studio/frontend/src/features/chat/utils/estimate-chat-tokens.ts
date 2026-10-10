// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { MessageRecord } from "../types";
import { orderBySelectedBranch } from "./message-order.ts";

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
    for (const part of content) {
      const text = (part as { text?: unknown } | null)?.text;
      if (typeof text === "string") chars += text.length;
    }
  }
  return chars > 0 ? Math.max(1, Math.round(chars / 4)) : null;
}

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
