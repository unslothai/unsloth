// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { normalizeModelIdentity } from "../../../lib/model-identity.ts";

function normalizeCachePath(path: string): string {
  const normalized = normalizeModelIdentity(path.trim());
  const minLength = normalized.startsWith("//")
    ? 2
    : /^[A-Za-z]:\//.test(normalized)
      ? 3
      : 1;
  let end = normalized.length;
  while (end > minLength && normalized.charCodeAt(end - 1) === 47) {
    end -= 1;
  }
  return end === normalized.length ? normalized : normalized.slice(0, end);
}

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
    normalizeCachePath(currentLocalPath) ===
    normalizeCachePath(expectedLocalPath)
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
