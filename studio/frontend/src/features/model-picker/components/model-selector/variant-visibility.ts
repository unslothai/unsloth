// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What a quantization expander shows, and when it opens. No React/DOM deps
// so it stays easy to test.

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

/** Whether a row's expander should mount. Auto-expansion waits for the
 *  sole-quant probe: without the wait, every On Device row opens an expander,
 *  and its remote listing, moments before collapsing into a single row. A
 *  row the user opened is not held back. */
export function shouldMountVariantExpander({
  expanded,
  autoExpand,
  soleQuantsPending,
}: {
  expanded: boolean;
  autoExpand: boolean;
  soleQuantsPending: boolean;
}): boolean {
  return expanded && !(autoExpand && soleQuantsPending);
}
