// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { listCachedGguf, listGgufVariants } from "@/features/chat/api/chat-api";
import {
  pinKey,
  pinnedQuantEntries,
  usePinnedModelsStore,
} from "./pinned-models";

export async function reconcileGgufPinsAfterDelete(
  repoId: string,
  hfToken?: string,
): Promise<void> {
  try {
    const copies = await listCachedGguf();
    const present = copies.some((copy) => copy.repo_id === repoId);
    const variants = present
      ? (
          await listGgufVariants(repoId, hfToken, {
            preferLocalCache: true,
          })
        ).variants
      : [];
    const state = usePinnedModelsStore.getState();
    if (!present && state.pinned.includes(pinKey(repoId)))
      state.togglePinned(repoId);
    for (const pin of pinnedQuantEntries(state.pinned)) {
      if (
        pin.repoId === repoId &&
        !variants.some(
          (v) => v.quant === pin.quant && v.downloaded && !v.partial,
        )
      ) {
        state.togglePinned(repoId, pin.quant);
      }
    }
  } catch {
    // An unavailable scan is not evidence that the last downloaded copy is gone.
  }
}
