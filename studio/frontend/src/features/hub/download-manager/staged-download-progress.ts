// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pure logic only (no "@/" imports) so node:test can load it directly.

/** Bytes of an entry's missing files transferred so far. `entryBytes` is the plan's missing-file total. A fresh job
 *  reports the same scope (the backend takes the already cached files off both of its counters), but a job resuming
 *  an earlier attempt on these files counts its whole file set, cached files included, against a scope-wide
 *  `expectedBytes`. The difference is the cached part, so it comes off the reading: counting it showed 100% while the
 *  missing files were still downloading. */
export function stagedEntryTransferredBytes(
  entryBytes: number,
  progress:
    | { expectedBytes: number; downloadedBytes: number }
    | null
    | undefined,
): number {
  const missing = Math.max(0, entryBytes);
  if (!progress) {
    return 0;
  }
  const cached = Math.max(0, progress.expectedBytes - missing);
  return Math.min(missing, Math.max(0, progress.downloadedBytes - cached));
}
