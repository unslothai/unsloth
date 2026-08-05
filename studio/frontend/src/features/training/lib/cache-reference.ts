// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { normalizeModelIdentity } from "@/features/hub/lib/model-identity.ts";

export function cacheLocalPathMatchesSelection(
  currentLocalPath: string | null | undefined,
  expectedLocalPath: string | null | undefined,
): boolean {
  if (expectedLocalPath === undefined) {
    return true;
  }
  if (currentLocalPath == null || expectedLocalPath == null) {
    return currentLocalPath == null && expectedLocalPath == null;
  }
  return (
    normalizeModelIdentity(currentLocalPath) ===
    normalizeModelIdentity(expectedLocalPath)
  );
}

export function cachedInventoryPathMatchesSelection(
  cachePath: string | null | undefined,
  selectedLocalPath: string | null,
): boolean {
  if (!selectedLocalPath?.trim()) {
    return true;
  }
  if (!cachePath?.trim()) {
    return false;
  }
  return cacheLocalPathMatchesSelection(cachePath, selectedLocalPath);
}

export function cacheReferenceMatchesSelection({
  currentId,
  expectedId,
  knownCached,
  currentLocalPath,
  expectedLocalPath,
}: {
  currentId: string | null;
  expectedId: string;
  knownCached: boolean;
  currentLocalPath: string | null;
  expectedLocalPath?: string | null;
}): boolean {
  if (currentId !== expectedId || !knownCached) {
    return false;
  }
  return cacheLocalPathMatchesSelection(currentLocalPath, expectedLocalPath);
}
