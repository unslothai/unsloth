// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { HfModelResult } from "../hooks/use-hub-model-search";
import { ownerOf, repoOf } from "./format";
import {
  hasRecommendedCheckpointLikes,
  matchesRecommended,
} from "./format-filters";
import { resolveOwnerProviderLogo } from "./provider-logos";

// Under Recommended, iconless models (no provider logo, e.g. Ornith, Inkling) show once they clear this many likes.
const MIN_ICONLESS_MODEL_LIKES = 30;

export function matchesRecommendedBrowse(
  result: HfModelResult,
  quantFormats: readonly string[],
  browsing = true,
): boolean {
  return (
    (!browsing ||
      ((resolveOwnerProviderLogo(ownerOf(result.id), repoOf(result.id)) !==
        null ||
        (result.likes ?? 0) >= MIN_ICONLESS_MODEL_LIKES) &&
        hasRecommendedCheckpointLikes(result))) &&
    matchesRecommended(result, quantFormats)
  );
}
