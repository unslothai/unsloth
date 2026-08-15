// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { InventoryHint } from "./types";

export type PendingInventoryHints = Record<
  InventoryHint["kind"],
  Map<string, InventoryHint>
>;

export type InventoryHintRow = {
  repo_id: string;
  size_bytes: number;
  partial?: boolean;
  optimistic?: boolean;
};

export type InventoryHintReconciliation = {
  rows: InventoryHintRow[];
  pending: PendingInventoryHints;
  staleCompletedHints: InventoryHint[];
  observedKeys: Set<string>;
};

type InventoryHintCommit = {
  pending: PendingInventoryHints;
  completedHintsToSuppress: InventoryHint[];
};

export const INVENTORY_HINT_TTL_MS = 5 * 60_000;

function rowRepoId(row: InventoryHintRow): string {
  return row.repo_id;
}

function rowSizeBytes(row: InventoryHintRow): number {
  return row.size_bytes;
}

function optimisticRow(hint: InventoryHint): InventoryHintRow {
  return {
    repo_id: hint.repoId,
    size_bytes: hint.bytes ?? 0,
    partial: false,
    optimistic: true,
  };
}

function suppressibleInventoryHint(hint: InventoryHint): InventoryHint {
  return {
    kind: hint.kind,
    repoId: hint.repoId,
    ...(hint.bytes ? { bytes: hint.bytes } : {}),
  };
}

export function createPendingInventoryHints(): PendingInventoryHints {
  return {
    model: new Map(),
    gguf: new Map(),
    dataset: new Map(),
  };
}

export function clonePendingInventoryHints(
  source: PendingInventoryHints,
): PendingInventoryHints {
  return {
    model: new Map(source.model),
    gguf: new Map(source.gguf),
    dataset: new Map(source.dataset),
  };
}

function clonePendingInventoryHintsForKind(
  source: PendingInventoryHints,
  kind: InventoryHint["kind"],
): PendingInventoryHints {
  return {
    ...source,
    [kind]: new Map(source[kind]),
  };
}

export function repoKey(repoId: string): string {
  return repoId.trim().toLowerCase();
}

export function inventoryHintKey(
  kind: InventoryHint["kind"],
  repoId: string,
): string {
  return `${kind}:${repoKey(repoId)}`;
}

function mergeInventoryHint(
  rows: InventoryHintRow[],
  hint: InventoryHint,
): InventoryHintRow[] {
  const lower = repoKey(hint.repoId);
  const seed = optimisticRow(hint);
  const idx = rows.findIndex((row) => repoKey(rowRepoId(row)) === lower);
  if (idx === -1) {
    return [...rows, seed];
  }
  const serverRow = rows[idx];
  const merged = {
    ...serverRow,
    // A completed hint may arrive before a partial server scan catches up. In
    // that case keep the synthetic row non-runnable. A complete server row is
    // already authoritative even when its runnable-weight size is smaller than
    // the hint's full-snapshot byte count, so do not mark that merge optimistic.
    ...(serverRow.partial ? seed : { optimistic: false }),
    size_bytes: Math.max(rowSizeBytes(rows[idx]), rowSizeBytes(seed)),
  };
  return [...rows.slice(0, idx), merged, ...rows.slice(idx + 1)];
}

export function rememberInventoryHint(
  pending: PendingInventoryHints,
  hint: InventoryHint,
): PendingInventoryHints {
  const next = clonePendingInventoryHints(pending);
  const key = repoKey(hint.repoId);
  const existing = next[hint.kind].get(key);
  const bytes = Math.max(existing?.bytes ?? 0, hint.bytes ?? 0);
  const createdAt = existing?.createdAt ?? hint.createdAt ?? Date.now();
  next[hint.kind].set(key, {
    kind: hint.kind,
    repoId: hint.repoId,
    ...(bytes > 0 ? { bytes } : {}),
    createdAt,
  });
  return next;
}

function inventoryHintExpired(hint: InventoryHint, now: number): boolean {
  return (hint.createdAt ?? now) + INVENTORY_HINT_TTL_MS <= now;
}

export function nextInventoryHintExpiryDelay(
  pending: PendingInventoryHints,
  now = Date.now(),
): number | null {
  let nextExpiry = Number.POSITIVE_INFINITY;
  for (const hints of Object.values(pending)) {
    for (const hint of hints.values()) {
      if (typeof hint.createdAt !== "number") {
        continue;
      }
      nextExpiry = Math.min(nextExpiry, hint.createdAt + INVENTORY_HINT_TTL_MS);
    }
  }
  return Number.isFinite(nextExpiry) ? Math.max(0, nextExpiry - now) : null;
}

export function pruneExpiredInventoryHints(
  pending: PendingInventoryHints,
  now = Date.now(),
): PendingInventoryHints {
  let next = pending;
  for (const kind of Object.keys(pending) as InventoryHint["kind"][]) {
    for (const [key, hint] of next[kind]) {
      if (!inventoryHintExpired(hint, now)) {
        continue;
      }
      if (next === pending) {
        next = clonePendingInventoryHints(pending);
      }
      next[kind].delete(key);
    }
  }
  return next;
}

function serverRowSatisfiesHint(
  row: InventoryHintRow,
  hint: InventoryHint,
): boolean {
  if (row.partial) {
    return false;
  }
  return !hint.bytes || rowSizeBytes(row) >= hint.bytes;
}

export function reconcileInventoryHints<T extends InventoryHintRow>({
  pending,
  kind,
  rows,
  previouslyObserved,
}: {
  pending: PendingInventoryHints;
  kind: InventoryHint["kind"];
  rows: T[];
  previouslyObserved: ReadonlySet<string>;
}): InventoryHintReconciliation {
  const pruned = pruneExpiredInventoryHints(pending);
  const serverRows = new Map(rows.map((row) => [repoKey(rowRepoId(row)), row]));
  const observedKeys = new Set(serverRows.keys());
  if (pruned[kind].size === 0) {
    return {
      rows,
      pending: pruned,
      staleCompletedHints: [],
      observedKeys,
    };
  }

  const nextPending = clonePendingInventoryHintsForKind(pruned, kind);
  const hints = nextPending[kind];
  const staleCompletedHints: InventoryHint[] = [];

  for (const [key, hint] of hints) {
    const serverRow = serverRows.get(key);
    if (serverRow && serverRowSatisfiesHint(serverRow, hint)) {
      hints.delete(key);
    } else if (!serverRow && previouslyObserved.has(key)) {
      hints.delete(key);
      staleCompletedHints.push(hint);
    }
  }

  let mergedRows: InventoryHintRow[] = rows;
  for (const hint of hints.values()) {
    mergedRows = mergeInventoryHint(mergedRows, hint);
  }

  return {
    rows: mergedRows,
    pending: nextPending,
    staleCompletedHints,
    observedKeys,
  };
}

export function commitInventoryHintReconciliation({
  current,
  kind,
  baseline,
  next,
  staleCompletedHints,
}: {
  current: PendingInventoryHints;
  kind: InventoryHint["kind"];
  baseline: PendingInventoryHints;
  next: PendingInventoryHints;
  staleCompletedHints: InventoryHint[];
}): InventoryHintCommit {
  const committed = clonePendingInventoryHints(
    pruneExpiredInventoryHints(current),
  );
  const sharedHints = committed[kind];
  const staleKeys = new Set(
    staleCompletedHints.map((hint) => repoKey(hint.repoId)),
  );
  const completedHintsToSuppress: InventoryHint[] = [];

  for (const [key, baselineHint] of baseline[kind]) {
    const nextHint = next[kind].get(key);
    const currentHint = sharedHints.get(key);
    const currentIsNewer =
      currentHint && (currentHint.bytes ?? 0) > (baselineHint.bytes ?? 0);

    if (nextHint) {
      if (!currentIsNewer) {
        sharedHints.set(key, nextHint);
      }
      continue;
    }
    if (currentIsNewer) {
      continue;
    }
    sharedHints.delete(key);
    if (staleKeys.has(key)) {
      completedHintsToSuppress.push(suppressibleInventoryHint(baselineHint));
    }
  }

  return { pending: committed, completedHintsToSuppress };
}
