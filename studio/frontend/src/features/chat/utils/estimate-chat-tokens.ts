// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ThreadMessage } from "@assistant-ui/react";
import type { MessageRecord } from "../types";
import { orderBySelectedBranch } from "./order-selected-branch.ts";

/**
 * Extracts raw text content from message records or ThreadMessages across the active branch.
 */
export function extractMessageText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  let text = "";
  for (const part of content) {
    if (!part || typeof part !== "object") continue;
    const p = part as { type?: string; text?: unknown };
    if (typeof p.text === "string") {
      text += p.text;
    }
  }
  return text;
}

export { orderBySelectedBranch as orderMessagesByBranch };

/**
 * Estimate token count from a list of stored message records or ThreadMessages (char length / 4).
 * Returns null if no text is found.
 */
export function estimateMessagesTokenCount(
  messages: readonly (MessageRecord | ThreadMessage)[] | null | undefined,
): number | null {
  if (!messages || messages.length === 0) return null;

  const branch = orderBySelectedBranch(
    messages as readonly {
      id: string;
      createdAt: number | Date;
      role: string;
      parentId?: string | null;
      content: MessageRecord["content"];
    }[],
  );

  let totalChars = 0;
  for (const msg of branch) {
    const text = extractMessageText(msg.content);
    totalChars += text.length;
  }

  if (totalChars === 0) return null;
  return Math.max(1, Math.round(totalChars / 4));
}
