// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The dialog holds one list height for a whole open: compact only when the history is known to
// have no rows, given up as soon as rows arrive, so filtering never resizes it. `hasRows` must
// therefore answer for a history not yet indexed (see chatSearchIndexHasRows): reading that as
// "no rows" sizes a populated dialog compact and then grows it mid-open.
export function isCompactChatSearchList(
  wasCompact: boolean,
  hasRows: boolean | null,
): boolean {
  // null is "unknown", which only a completed build settles. Compact is wrong for a history that
  // turns out to have rows, so unknown takes the fixed height.
  return wasCompact && hasRows === false;
}
