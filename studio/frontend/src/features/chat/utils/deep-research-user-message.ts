// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { MessageRecord } from "../types";

export type DeepResearchUserMessageInput = {
  readonly id: string;
  readonly threadId: string;
  readonly parentId: string | null;
  readonly content: MessageRecord["content"];
  readonly attachments?: MessageRecord["attachments"];
  readonly createdAt: number;
  /** Persisted row metadata: authoritative, including explicit null clears. */
  readonly storedMetadata?: Record<string, unknown>;
  /** Live `message.metadata.custom` fallback when the row has no value yet. */
  readonly liveMetadata?: Record<string, unknown>;
};

/**
 * Builds the user-message row Deep Research rewrites before starting a run.
 * The backend replaces metadata_json wholesale, so this must carry the full
 * stored snapshot while filling fields the row has not received yet.
 */
export function deepResearchUserMessageRecord(
  input: DeepResearchUserMessageInput,
): MessageRecord {
  const metadata: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(input.liveMetadata ?? {})) {
    if (value !== undefined) {
      metadata[key] = value;
    }
  }
  for (const [key, value] of Object.entries(input.storedMetadata ?? {})) {
    // An explicit null is a deliberate clear and outranks any live fallback.
    metadata[key] = value;
  }

  return {
    id: input.id,
    threadId: input.threadId,
    parentId: input.parentId,
    role: "user",
    content: input.content,
    ...(input.attachments && input.attachments.length > 0
      ? { attachments: input.attachments }
      : {}),
    ...(Object.keys(metadata).length > 0 ? { metadata } : {}),
    createdAt: input.createdAt,
  };
}
