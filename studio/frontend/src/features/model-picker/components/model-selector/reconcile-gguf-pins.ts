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
  const pins = pinnedQuantEntries(
    usePinnedModelsStore.getState().pinned,
  ).filter((pin) => pin.repoId === repoId && (!quant || pin.quant === quant));
  if (!pins.length) return;
  try {
    const missing = await missingPinnedQuants(
      pins,
      await listCachedGguf(),
      async (copy) => {
        const response = await listGgufVariants(copy.repo_id, hfToken, {
          preferLocalCache: true,
          localPath: copy.cache_path || copy.load_id || undefined,
        });
        return response.variants;
      },
    );
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
