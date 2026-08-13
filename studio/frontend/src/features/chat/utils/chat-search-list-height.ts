// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The search dialog holds one list height for a whole open. It starts compact unless the
// index is already known to have rows, and gives the compact height up as soon as rows
// arrive, so a query that narrows or empties the list never resizes the dialog. Staying
// compact until then keeps an empty or not-yet-built index off a height it cannot fill.
export function isCompactChatSearchList(
  wasCompact: boolean,
  itemCount: number,
): boolean {
  return wasCompact && itemCount === 0;
}
