// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Height is fixed per open; only known-empty history is compact, so filtering never resizes.
export function isCompactChatSearchList(
  wasCompact: boolean,
  hasRows: boolean | null,
): boolean {
  return wasCompact && hasRows === false;
}
