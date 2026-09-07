// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { listCachedGguf, listGgufVariants } from "@/features/chat/api/chat-api";
import {
  pinKey,
  pinnedQuantEntries,
  usePinnedModelsStore,
} from "./pinned-models";
import { missingPinnedQuants } from "./pinned-quant-sources";

export async function reconcileGgufPinsAfterDelete(
  repoId: string,
  quant?: string,
  hfToken?: string,
): Promise<void> {
  const pinned = usePinnedModelsStore.getState().pinned;
  const barePinned = pinned.includes(pinKey(repoId));
  const pins = pinnedQuantEntries(pinned).filter(
    (pin) => pin.repoId === repoId && (!quant || pin.quant === quant),
  );
  if (!pins.length && !barePinned) return;
  try {
    const copies = await listCachedGguf();
    const missing = await missingPinnedQuants(pins, copies, async (copy) => {
      const response = await listGgufVariants(copy.repo_id, hfToken, {
        preferLocalCache: true,
        localPath: copy.cache_path || copy.load_id || undefined,
      });
      return response.variants;
    });
    if (barePinned && !copies.some((copy) => copy.repo_id === repoId)) {
      const state = usePinnedModelsStore.getState();
      if (state.pinned.includes(pinKey(repoId))) state.togglePinned(repoId);
    }
    for (const pin of missing) {
      const state = usePinnedModelsStore.getState();
      if (state.pinned.includes(pinKey(pin.repoId, pin.quant))) {
        state.togglePinned(pin.repoId, pin.quant);
      }
    }
  } catch {
    // Preserve the user's pins until all remaining copies can be checked.
  }
}
