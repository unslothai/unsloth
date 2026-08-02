// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Bookkeeping for the On Device sole-quant probe: what is known, what is
// still being read, and what needs reading. Per repo, so one repo's
// invalidation leaves the rest of the list alone. No React/DOM deps so it
// stays easy to test.

/** A repo to probe, tagged with the cache version and location it reads at. */
export interface SoleQuantTarget {
  repoId: string;
  localSource: string | null;
  key: string;
}

/** A probe result. A null quant means the repo has no single complete quant,
 *  including when the read failed: either way the row stays expandable. */
export interface SoleQuantEntry<T> {
  key: string;
  quant: T | null;
}

/** Identity of one repo's probe. Moves when that repo's variants cache is
 *  invalidated or the row points at another directory. */
export function soleQuantKey(
  version: string | undefined,
  localSource: string | null,
): string {
  return `${version ?? ""}::${localSource ?? ""}`;
}

/** Split the listed repos into resolved rows, repos still being read, and
 *  repos needing a read. A repo holds its result until its own key moves. */
export function partitionSoleQuants<T>(
  targets: readonly SoleQuantTarget[],
  entries: ReadonlyMap<string, SoleQuantEntry<T>>,
  { enabled }: { enabled: boolean },
): {
  quants: ReadonlyMap<string, T>;
  pending: ReadonlySet<string>;
  stale: SoleQuantTarget[];
} {
  const quants = new Map<string, T>();
  const pending = new Set<string>();
  const stale: SoleQuantTarget[] = [];
  if (!enabled) return { quants, pending, stale };
  for (const target of targets) {
    const entry = entries.get(target.repoId);
    if (!entry || entry.key !== target.key) {
      pending.add(target.repoId);
      stale.push(target);
      continue;
    }
    if (entry.quant) quants.set(target.repoId, entry.quant);
  }
  return { quants, pending, stale };
}
