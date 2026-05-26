// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function isMissingLocalDatasetCacheError(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error ?? "");
  const lower = message.toLowerCase();
  return (
    lower.includes("not available in the local cache") ||
    (lower.includes("local cache") && lower.includes("not found"))
  );
}
