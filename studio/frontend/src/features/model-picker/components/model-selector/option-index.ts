// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Each key's first position, as `indexOf` reports it, built in one pass so rows can look up in O(1). */
export function firstIndexByKey(keys: readonly string[]): Map<string, number> {
  const indexByKey = new Map<string, number>();
  keys.forEach((key, index) => {
    if (!indexByKey.has(key)) indexByKey.set(key, index);
  });
  return indexByKey;
}
