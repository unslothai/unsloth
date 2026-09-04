// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { MessageTree } from "../../../i18n/types";

function isTree(value: unknown): value is MessageTree {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Deep-merge two message trees. Overlay keys win; nested objects are merged.
 * Used to keep fork-only Unforgettable strings out of upstream locale catalogs.
 */
export function mergeMessageTrees<A extends MessageTree, B extends MessageTree>(
  base: A,
  overlay: B,
): A & B {
  const result: Record<string, string | MessageTree> = { ...base };
  for (const key of Object.keys(overlay)) {
    const overlayValue = overlay[key];
    const baseValue = result[key];
    if (isTree(baseValue) && isTree(overlayValue)) {
      result[key] = mergeMessageTrees(baseValue, overlayValue);
    } else {
      result[key] = overlayValue;
    }
  }
  return result as A & B;
}
