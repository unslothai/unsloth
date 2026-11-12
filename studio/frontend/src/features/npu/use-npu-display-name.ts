// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect } from "react";
import { isNpuModelId } from "./api";
import { npuDisplayNameOf } from "./display-name";
import { refreshNpuModels, useNpuCatalogStore } from "./npu-catalog-store";
import { useNpuStatus } from "./use-npu-status";

let listed = false;

/** A lookup from NPU model id or path to its display name (null when unlisted); `npuInUse` lists the catalog once. */
export function useNpuDisplayNames(
  npuInUse: boolean,
): (idOrPath: string | null | undefined) => string | null {
  const models = useNpuCatalogStore((state) => state.models);
  const wantsListing = npuInUse && models === null && !listed;
  const [status] = useNpuStatus(wantsListing);
  useEffect(() => {
    // A model loaded on an earlier visit is named before any picker lists the catalog.
    // Listing starts lemond, so only on a validated runtime.
    if (!wantsListing || status?.ready !== true) return;
    listed = true;
    void refreshNpuModels().then((ok) => {
      // A failed listing may retry on a later render, spaced out so a down server is not hammered.
      if (!ok) setTimeout(() => (listed = false), 30_000);
    });
  }, [wantsListing, status?.ready]);
  return useCallback(
    (idOrPath: string | null | undefined) => npuDisplayNameOf(models, idOrPath),
    [models],
  );
}

export function useNpuDisplayName(
  idOrPath: string | null | undefined,
): string | null {
  return useNpuDisplayNames(isNpuModelId(idOrPath))(idOrPath);
}

/** Non-hook form for toasts and callbacks. */
export function npuDisplayNameNow(
  idOrPath: string | null | undefined,
): string | null {
  return npuDisplayNameOf(useNpuCatalogStore.getState().models, idOrPath);
}
