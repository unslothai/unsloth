// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What a quantization expander shows, and when it opens. No React/DOM deps so it stays easy to test.

/** On Device with "Show all quantizations" off lists what the repo holds on disk: complete quants,
 *  plus torn ones, which still occupy space and need a resume. Never-downloaded quants stay
 *  hidden. Browse lists always show every quant. */
export function visibleGgufVariants<
  T extends { downloaded?: boolean; partial?: boolean },
>(
  variants: readonly T[],
  { onDevice, showAll }: { onDevice: boolean; showAll: boolean },
): readonly T[] {
  if (showAll || !onDevice) return variants;
  return variants.filter((v) => v.downloaded === true || v.partial === true);
}

/** Whether a row's expander should mount. Auto-expansion waits for the sole-quant probe: without
 *  the wait, every On Device row opens an expander, and its remote listing, moments before
 *  collapsing into a single row. A row the user opened is not held back. */
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

/** Next collapse/reopen sets after a click on an auto-expanded row. `showing` is what the row
 *  renders, not what the collapse set says: a row held back by its sole-quant probe shows
 *  nothing, so a click opens it. A reopened row stops following the auto-expand preference until
 *  it is collapsed again. */
export function toggleAutoExpandedRow(
  state: { collapsed: ReadonlySet<string>; reopened: ReadonlySet<string> },
  { repoId, showing }: { repoId: string; showing: boolean },
): { collapsed: Set<string>; reopened: Set<string> } {
  const collapsed = new Set(state.collapsed);
  const reopened = new Set(state.reopened);
  if (showing) {
    collapsed.add(repoId);
    reopened.delete(repoId);
  } else {
    collapsed.delete(repoId);
    reopened.add(repoId);
  }
  return { collapsed, reopened };
}
