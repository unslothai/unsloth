// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { SIDEBAR_ORGANIZATION_STORAGE_KEY } from "../stores/sidebar-organization-keys.ts";
import { isValidPinId } from "./pinned-chats-sync.ts";

function ids(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return [...new Set(value.filter(isValidPinId))].slice(0, 10_000);
}

/** Read the old browser-only pins once the server confirms it has no pin state. */
export function readLegacyPinnedChatIds(
  storage: Pick<Storage, "getItem">,
): string[] {
  try {
    const saved = JSON.parse(storage.getItem("unsloth_pinned_chats") ?? "null");
    const pinned = ids(saved?.state?.pinnedIds);
    let order: string[] = [];
    try {
      const organization = JSON.parse(
        storage.getItem(SIDEBAR_ORGANIZATION_STORAGE_KEY) ?? "null",
      );
      order = ids(organization?.state?.manualOrder?.pinned);
    } catch {
      // A damaged sidebar preference must not discard otherwise valid pins.
    }
    const rank = new Map(order.map((id, index) => [id, index]));
    // Match the old sidebar: newly pinned (unranked) chats lead the dragged rows.
    return pinned.sort((a, b) => (rank.get(a) ?? -1) - (rank.get(b) ?? -1));
  } catch {
    return [];
  }
}
