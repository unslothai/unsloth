// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** On Device with "All quantizations" off: complete and torn quants only. Browse shows all. */
export function visibleGgufVariants<
  T extends { downloaded?: boolean; partial?: boolean },
>(
  variants: readonly T[],
  { onDevice, showAll }: { onDevice: boolean; showAll: boolean },
): readonly T[] {
  if (showAll || !onDevice) return variants;
  return variants.filter((v) => v.downloaded === true || v.partial === true);
}

/** Auto-expansion waits for the sole-quant probe, else every row opens and fetches before collapsing. */
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

/** `showing` is what the row renders, not the collapse set: a probe-held row shows nothing. */
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
