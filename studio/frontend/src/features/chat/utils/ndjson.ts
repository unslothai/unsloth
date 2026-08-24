// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type ConversationJsonlLayout = "training" | "messages";

export function conversationJsonlBody(
  messages: readonly unknown[],
  layout: ConversationJsonlLayout,
): string {
  const records = layout === "training" ? [{ messages }] : messages;
  return records.map((record) => JSON.stringify(record)).join("\n");
}
