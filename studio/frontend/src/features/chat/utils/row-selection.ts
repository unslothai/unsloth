// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Multi-select for sidebar rows: cmd or ctrl click toggles one row, shift click takes the block
 *  between the anchor and the clicked row. */

/** Rows from `anchorId` to `targetId` inclusive, in list order, either way round. */
export function rangeBetween(
  ids: string[],
  anchorId: string,
  targetId: string,
): string[] {
  const from = ids.indexOf(anchorId);
  const to = ids.indexOf(targetId);
  // A missing anchor means the list changed under the selection, so the click stands on its own
  // rather than selecting an arbitrary block.
  if (from === -1 || to === -1) return to === -1 ? [] : [targetId];
  return from <= to ? ids.slice(from, to + 1) : ids.slice(to, from + 1);
}

/** The set with `id` added, or removed when it was already there. */
export function toggleSelected(
  selected: ReadonlySet<string>,
  id: string,
): Set<string> {
  const next = new Set(selected);
  if (!next.delete(id)) next.add(id);
  return next;
}
