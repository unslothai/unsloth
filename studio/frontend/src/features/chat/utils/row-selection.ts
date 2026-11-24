// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function rangeBetween(
  ids: string[],
  anchorId: string,
  targetId: string,
): string[] {
  const from = ids.indexOf(anchorId);
  const to = ids.indexOf(targetId);
  // A missing anchor means the list changed, so the click stands on its own.
  if (from === -1 || to === -1) return to === -1 ? [] : [targetId];
  return from <= to ? ids.slice(from, to + 1) : ids.slice(to, from + 1);
}

export function toggleSelected(
  selected: ReadonlySet<string>,
  id: string,
): Set<string> {
  const next = new Set(selected);
  if (!next.delete(id)) next.add(id);
  return next;
}
