// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Deletes each run in turn; ``failed`` is the first error message, or null when all went. */
export async function deleteEach(
  ids: string[],
  remove: (id: string) => Promise<unknown>,
): Promise<{ deleted: Set<string>; failed: string | null }> {
  const deleted = new Set<string>();
  let failed: string | null = null;
  for (const id of ids) {
    try {
      await remove(id);
      deleted.add(id);
    } catch (err) {
      failed ??= err instanceof Error ? err.message : String(err);
    }
  }
  return { deleted, failed };
}
