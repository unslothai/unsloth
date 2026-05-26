// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { notifyModelDeleted } from "@/features/chat";

export type DeletedInventoryEntry =
  | { kind: "model"; id: string; ggufVariant?: string | null }
  | { kind: "dataset"; id: string };

export function notifyInventoryEntryDeleted(entry: DeletedInventoryEntry): void {
  if (entry.kind === "model") {
    notifyModelDeleted(entry.id, entry.ggufVariant, { notifyInventory: false });
  }
}
