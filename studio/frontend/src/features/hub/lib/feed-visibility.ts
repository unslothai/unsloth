// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { resolveOwnerProviderLogo } from "./provider-logos.ts";

// Iconless models (no provider logo, e.g. Ornith, Inkling) show once they clear this many likes.
export const MIN_ICONLESS_MODEL_LIKES = 30;

export interface FeedGateRow {
  owner: string;
  repo: string;
  likes?: number | null;
}

/**
 * Feed visibility for a model row: logo'd providers, iconless ones above the
 * likes threshold, and every row published by the list's own owner. A
 * createdAt-sorted owner list ("Latest Unsloth Models") must show that owner's
 * newest releases, which are usually under the threshold (#9456).
 */
export function passesFeedIconlessGate(
  row: FeedGateRow,
  listOwner: string | null | undefined,
): boolean {
  if (listOwner && row.owner.toLowerCase() === listOwner.toLowerCase()) {
    return true;
  }
  return (
    resolveOwnerProviderLogo(row.owner, row.repo) !== null ||
    (row.likes ?? 0) >= MIN_ICONLESS_MODEL_LIKES
  );
}
