// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function isMissingLocalDatasetCacheError(error: unknown): boolean {
  if (!error || typeof error !== "object") {
    return false;
  }
  const candidate = error as { errorCode?: unknown; code?: unknown };
  const errorCode = candidate.errorCode ?? candidate.code;
  return errorCode === "dataset_local_cache_miss";
}
