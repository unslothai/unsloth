// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Which quantizations an expander lists. No React/DOM deps so it stays easy
// to test.

/** On Device with "Show all quantizations" off lists what the repo holds on
 *  disk: complete quants, plus torn ones, which still occupy space and need a
 *  resume. Quants that were never downloaded stay hidden. Browse lists
 *  (Recommended and the rest) always show every quant. */
export function visibleGgufVariants<
  T extends { downloaded?: boolean; partial?: boolean },
>(
  variants: readonly T[],
  { onDevice, showAll }: { onDevice: boolean; showAll: boolean },
): readonly T[] {
  if (showAll || !onDevice) return variants;
  return variants.filter((v) => v.downloaded === true || v.partial === true);
}
