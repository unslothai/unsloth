// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { INVENTORY_HINT_KINDS } from "./constants.ts";
import {
  type InventoryHintReconciliation,
  type PendingInventoryHints,
  clonePendingInventoryHints,
  commitInventoryHintReconciliation,
  createPendingInventoryHints,
  pruneExpiredInventoryHints,
  reconcileInventoryHints,
  rememberInventoryHint,
  repoKey,
} from "./inventory-hints.ts";
import type { InventoryHint } from "./types.ts";
import { evictOldestUnprotected } from "../lib/lru-map.ts";

export const MAX_OBSERVED_INVENTORY_KEYS_PER_KIND = 1024;

type ObservedInventoryKeys = Record<InventoryHint["kind"], Set<string>>;

export type PendingHintReconciliationCommit = {
  kind: InventoryHint["kind"];
  reconciliation: Pick<
    InventoryHintReconciliation,
    "pending" | "staleCompletedHints" | "observedKeys"
  >;
};

type InventoryHintState = {
  pending: PendingInventoryHints;
  observedKeys: ObservedInventoryKeys;
  pruneExpiredHints: () => void;
  commitReconciliations: (
    completedHints: readonly InventoryHint[],
    baseline: PendingInventoryHints,
    reconciliations: readonly PendingHintReconciliationCommit[],
  ) => InventoryHint[];
};

const INVENTORY_HINT_EQUALITY_FIELDS = {
  kind: true,
  repoId: true,
  bytes: true,
  createdAt: true,
} satisfies Record<keyof InventoryHint, true>;

const INVENTORY_HINT_EQUALITY_KEYS = Object.keys(
  INVENTORY_HINT_EQUALITY_FIELDS,
) as (keyof InventoryHint)[];

function createObservedInventoryKeys(): ObservedInventoryKeys {
  return {
    model: new Set(),
    gguf: new Set(),
    dataset: new Set(),
  };
}

function setEqual(a: ReadonlySet<string>, b: ReadonlySet<string>): boolean {
  if (a.size !== b.size) {
    return false;
  }
  for (const value of a) {
    if (!b.has(value)) {
      return false;
    }
  }
  return true;
}

function inventoryHintEqual(
  a: InventoryHint | undefined,
  b: InventoryHint,
): boolean {
  if (a === undefined) {
    return false;
  }
  return INVENTORY_HINT_EQUALITY_KEYS.every((key) => {
    const left =
      key === "bytes" || key === "createdAt" ? (a[key] ?? 0) : a[key];
    const right =
      key === "bytes" || key === "createdAt" ? (b[key] ?? 0) : b[key];
    return left === right;
  });
}

export function pendingInventoryHintsEqual(
  a: PendingInventoryHints,
  b: PendingInventoryHints,
): boolean {
  for (const kind of INVENTORY_HINT_KINDS) {
    if (a[kind].size !== b[kind].size) {
      return false;
    }
    for (const [key, hint] of a[kind]) {
      if (!inventoryHintEqual(b[kind].get(key), hint)) {
        return false;
      }
    }
  }
  return true;
}

export function pendingHintReconciliationNeedsCommit(
  baseline: PendingInventoryHints,
  observedKeys: Record<InventoryHint["kind"], ReadonlySet<string>>,
  commit: PendingHintReconciliationCommit,
): boolean {
  if (!pendingInventoryHintsEqual(baseline, commit.reconciliation.pending)) {
    return true;
  }
  if (commit.reconciliation.staleCompletedHints.length > 0) {
    return true;
  }
  return !setEqual(
    observedKeys[commit.kind],
    commit.reconciliation.observedKeys,
  );
}

function observedInventoryKeysEqual(
  a: ObservedInventoryKeys,
  b: ObservedInventoryKeys,
): boolean {
  return INVENTORY_HINT_KINDS.every((kind) => setEqual(a[kind], b[kind]));
}

function pendingHasHint(
  pending: PendingInventoryHints,
  hint: InventoryHint,
): boolean {
  const existing = pending[hint.kind].get(repoKey(hint.repoId));
  if (!existing) {
    return false;
  }
  return (existing.bytes ?? 0) >= (hint.bytes ?? 0);
}

export function pendingWithInventoryHints(
  pending: PendingInventoryHints,
  hints: readonly InventoryHint[],
): PendingInventoryHints {
  let next = pruneExpiredInventoryHints(pending);
  for (const hint of hints) {
    if (!pendingHasHint(next, hint)) {
      next = rememberInventoryHint(next, hint);
    }
  }
  return next;
}

function rememberObservedInventoryKeys(
  current: ObservedInventoryKeys,
  kind: InventoryHint["kind"],
  keys: ReadonlySet<string>,
  pending: PendingInventoryHints,
): ObservedInventoryKeys {
  const observed = new Set(current[kind]);
  for (const key of keys) {
    observed.delete(key);
    observed.add(key);
  }
  evictOldestUnprotected(
    observed,
    MAX_OBSERVED_INVENTORY_KEYS_PER_KIND,
    (key) => pending[kind].has(key),
  );
  if (setEqual(current[kind], observed)) {
    return current;
  }
  return { ...current, [kind]: observed };
}

export const useInventoryHintStore = create<InventoryHintState>()(
  (set, get) => ({
    pending: createPendingInventoryHints(),
    observedKeys: createObservedInventoryKeys(),
    pruneExpiredHints: () => {
      const current = get().pending;
      const next = pruneExpiredInventoryHints(current);
      if (!pendingInventoryHintsEqual(current, next)) {
        set({ pending: clonePendingInventoryHints(next) });
      }
    },
    commitReconciliations: (completedHints, baseline, reconciliations) => {
      const state = get();
      let next = pendingWithInventoryHints(state.pending, completedHints);
      let observedKeys = state.observedKeys;
      const completedHintsToSuppress: InventoryHint[] = [];
      for (const { kind, reconciliation } of reconciliations) {
        const committed = commitInventoryHintReconciliation({
          current: next,
          kind,
          baseline,
          next: reconciliation.pending,
          staleCompletedHints: reconciliation.staleCompletedHints,
        });
        next = committed.pending;
        observedKeys = rememberObservedInventoryKeys(
          observedKeys,
          kind,
          reconciliation.observedKeys,
          next,
        );
        completedHintsToSuppress.push(...committed.completedHintsToSuppress);
      }
      const unchanged =
        pendingInventoryHintsEqual(state.pending, next) &&
        observedInventoryKeysEqual(state.observedKeys, observedKeys);
      if (!unchanged) {
        set({ pending: clonePendingInventoryHints(next), observedKeys });
      }
      return completedHintsToSuppress;
    },
  }),
);

export { reconcileInventoryHints };
