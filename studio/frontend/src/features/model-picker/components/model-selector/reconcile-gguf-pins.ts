// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { listGgufVariants } from "@/features/chat/api/chat-api";
import { fetchCachedGgufInventory } from "@/features/hub/inventory/api";
import { pinnedQuantEntries, usePinnedModelsStore } from "./pinned-models";
import {
  ggufVariantsMatchForPicker,
  modelIdsMatchForPicker,
} from "./row-identity";

export function isChatGgufTask(task: string | null | undefined): boolean {
  return !task || task === "text-generation" || task === "image-text-to-text";
}

export async function reconcileGgufPinsAfterDelete(
  repoId: string,
  hfToken?: string,
): Promise<void> {
  try {
    const inventory = await fetchCachedGgufInventory(hfToken);
    if (inventory.scan_confirmed === false) return;
    const copies = inventory.cached.filter((copy) =>
      modelIdsMatchForPicker(copy.repo_id, repoId),
    );
    const variants =
      copies.length > 0
        ? (
            await listGgufVariants(repoId, hfToken, {
              preferLocalCache: true,
            })
          ).variants
        : [];
    // An anonymous Hub listing cannot disprove the complete copy found on disk.
    if (
      !hfToken &&
      copies.some((copy) => !copy.partial) &&
      !variants.some((variant) => variant.downloaded || variant.partial)
    )
      return;
    const state = usePinnedModelsStore.getState();
    if (!variants.some((v) => v.downloaded && !v.partial)) {
      for (const pin of state.pinned) {
        if (modelIdsMatchForPicker(pin, repoId)) state.togglePinned(pin);
      }
    }
    for (const pin of pinnedQuantEntries(state.pinned)) {
      if (
        modelIdsMatchForPicker(pin.repoId, repoId) &&
        !variants.some(
          (v) =>
            ggufVariantsMatchForPicker(v.quant, pin.quant) &&
            v.downloaded &&
            !v.partial,
        )
      ) {
        state.togglePinned(pin.repoId, pin.quant);
      }
    }
  } catch {
    // An unavailable scan is not evidence that the last downloaded copy is gone.
  }
}
