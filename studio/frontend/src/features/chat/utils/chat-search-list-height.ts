// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The search dialog holds one list height for a whole open. It starts compact only when the
// history is known to have no rows, and gives the compact height up as soon as rows arrive,
// so a query that narrows or empties the list never resizes the dialog. `hasRows` therefore
// has to answer for a history the current index has not been built for yet (see
// chatSearchIndexHasRows): treating that as "no rows" would size a populated dialog compact
// and then grow it mid-open, which is the stutter this is here to remove.
export function isCompactChatSearchList(
  wasCompact: boolean,
  hasRows: boolean | null,
): boolean {
  // null is "history unknown", which only a completed build can settle. Compact is right
  // for a history known to be empty and wrong for one that turns out to have rows, so an
  // unknown history takes the fixed height and never resizes mid-open.
  return wasCompact && hasRows === false;
}
