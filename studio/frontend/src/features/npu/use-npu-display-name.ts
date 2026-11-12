// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect } from "react";
import { isNpuModelId } from "./api";
import { npuDisplayNameOf } from "./display-name";
import { refreshNpuModels, useNpuCatalogStore } from "./npu-catalog-store";

let listed = false;

/** A lookup from NPU model id or path to its display name (null when unlisted); `npuInUse` lists the catalog once. */
export function useNpuDisplayNames(
  npuInUse: boolean,
): (idOrPath: string | null | undefined) => string | null {
  const models = useNpuCatalogStore((state) => state.models);
  useEffect(() => {
    // A model loaded on an earlier visit is named before any picker has listed the catalog.
    if (!npuInUse || models !== null || listed) return;
    listed = true;
    void refreshNpuModels();
  }, [npuInUse, models]);
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
