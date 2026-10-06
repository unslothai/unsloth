// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ProviderCompactionContentPart } from "../types/api";

export function providerCompactionPart(
  value: unknown,
): ProviderCompactionContentPart | null {
  if (!value || typeof value !== "object") return null;
  const { content, encrypted_content } = value as Record<string, unknown>;
  const compaction: ProviderCompactionContentPart = { type: "compaction" };
  if (typeof content === "string" && content) {
    compaction.content = content;
  }
  if (typeof encrypted_content === "string" && encrypted_content) {
    compaction.encrypted_content = encrypted_content;
  }
  return compaction.content || compaction.encrypted_content ? compaction : null;
}
