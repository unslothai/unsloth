// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function normaliseChatMemory(text: string): string {
  return text.replace(/\r\n/g, "\n").trim();
}

/** UTF-8 size as the backend saves it, so the editor refuses what the save would. */
export function chatMemoryBytes(text: string): number {
  const body = normaliseChatMemory(text);
  return body ? new TextEncoder().encode(`${body}\n`).length : 0;
}
