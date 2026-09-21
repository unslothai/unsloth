// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useSyncExternalStore } from "react";
import {
  getGgufVariantsCacheVersion,
  subscribeGgufVariantsCache,
} from "./gguf-variants-cache-events";

export function useGgufVariantsCacheVersion(repoId?: string | null): string {
  return useSyncExternalStore(
    subscribeGgufVariantsCache,
    () => getGgufVariantsCacheVersion(repoId),
    () => getGgufVariantsCacheVersion(repoId),
  );
}

/** One snapshot over several repos. Invalidation is per repo, so a caller
 *  watching a list has to read every repo's version, not just the global one. */
export function useGgufVariantsCacheVersions(
  repoIds: readonly string[],
): string {
  const getSnapshot = useCallback(
    () => repoIds.map((id) => getGgufVariantsCacheVersion(id)).join(","),
    [repoIds],
  );
  return useSyncExternalStore(
    subscribeGgufVariantsCache,
    getSnapshot,
    getSnapshot,
  );
}
