// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pinned models for the model selector's On Device list, persisted in
// localStorage so pins survive reloads. GGUF quants pin individually
// (repoId + quant); non-GGUF repos pin as a whole. Pinned entries surface
// in a "Pinned" section above the Unsloth/Downloaded group.

import { create } from "zustand";

const KEY = "unsloth_pinned_models";

// Entries are stored as strings: "repoId" pins a whole (non-GGUF) repo,
// "repoId::quant" pins one GGUF quant. Neither part contains "::".
export function pinKey(repoId: string, quant?: string): string {
  return quant ? `${repoId}::${quant}` : repoId;
}

export interface PinnedQuantEntry {
  repoId: string;
  quant: string;
}

export function makePinRank(
  pinned: readonly string[],
): (key: string) => number {
  const pinIndex = new Map(pinned.map((key, index) => [key, index]));
  return (key) => pinIndex.get(key) ?? Number.MAX_SAFE_INTEGER;
}

/** The pinned GGUF quants, in pin order. Plain repo pins are excluded. */
export function pinnedQuantEntries(pinned: string[]): PinnedQuantEntry[] {
  const out: PinnedQuantEntry[] = [];
  for (const key of pinned) {
    const sep = key.indexOf("::");
    if (sep <= 0) continue;
    const repoId = key.slice(0, sep);
    const quant = key.slice(sep + 2);
    if (repoId && quant) out.push({ repoId, quant });
  }
  return out;
}

function readPinned(): string[] {
  try {
    const raw = JSON.parse(localStorage.getItem(KEY) ?? "[]");
    return Array.isArray(raw)
      ? raw.filter((v): v is string => typeof v === "string")
      : [];
  } catch {
    return [];
  }
}

function writePinned(pinned: string[]): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(pinned));
  } catch {
    // Ignore unavailable storage; pins stay session-only.
  }
}

// A drag reorders live under the cursor, so `movePinned` runs on every
// dragenter. Persisting each of those would write localStorage dozens of times
// per drag and, worse, would make a drag the user cancels with Escape or by
// releasing outside the grid permanent: dragend can clear the drag marker but
// cannot undo writes. So a drag session snapshots the order up front, keeps
// every intermediate move in memory only, and either commits once on drop or
// restores the snapshot when the drag ends without one.
let dragSnapshot: string[] | null = null;

function sameOrder(a: readonly string[], b: readonly string[]): boolean {
  return a.length === b.length && a.every((key, index) => key === b[index]);
}

/** Same keys in the same multiset, order ignored. */
function sameKeys(a: readonly string[], b: readonly string[]): boolean {
  if (a.length !== b.length) return false;
  const counts = new Map<string, number>();
  for (const key of a) counts.set(key, (counts.get(key) ?? 0) + 1);
  for (const key of b) {
    const left = counts.get(key);
    if (!left) return false;
    counts.set(key, left - 1);
  }
  return true;
}

interface PinnedModelsState {
  pinned: string[];
  togglePinned: (repoId: string, quant?: string) => void;
  /**
   * Move `fromKey` into `toKey`'s slot. Both keys must already be pinned;
   * anything else is a no-op. Outside a drag session the new order is
   * persisted immediately, inside one it is held until `endPinnedDrag`.
   */
  movePinned: (fromKey: string, toKey: string) => void;
  /** Snapshot the current order so a cancelled drag can be undone. */
  beginPinnedDrag: () => void;
  /**
   * End a drag session. `commit` persists the reordered list; otherwise the
   * snapshot taken by `beginPinnedDrag` is restored. Idempotent, because drop
   * is followed by dragend and only the first of the two may decide.
   */
  endPinnedDrag: (commit: boolean) => void;
}

export const usePinnedModelsStore = create<PinnedModelsState>((set) => ({
  pinned: readPinned(),
  togglePinned: (repoId, quant) =>
    set((state) => {
      const key = pinKey(repoId, quant);
      // Newest pin first, so "Pin to top" literally lands on top of the
      // pinned group rather than under earlier pins.
      const next = state.pinned.includes(key)
        ? state.pinned.filter((id) => id !== key)
        : [key, ...state.pinned];
      writePinned(next);
      return { pinned: next };
    }),
  movePinned: (fromKey, toKey) =>
    set((state) => {
      const from = state.pinned.indexOf(fromKey);
      const to = state.pinned.indexOf(toKey);
      if (from < 0 || to < 0 || from === to) return state;
      const next = [...state.pinned];
      next.splice(to, 0, ...next.splice(from, 1));
      if (dragSnapshot === null) writePinned(next);
      return { pinned: next };
    }),
  beginPinnedDrag: () =>
    set((state) => {
      dragSnapshot = [...state.pinned];
      return state;
    }),
  endPinnedDrag: (commit) =>
    set((state) => {
      const snapshot = dragSnapshot;
      dragSnapshot = null;
      if (snapshot === null) return state;
      if (commit) {
        // Nothing moved, so there is nothing to persist.
        if (sameOrder(snapshot, state.pinned)) return state;
        writePinned(state.pinned);
        return state;
      }
      // Another window may have rewritten the list mid-drag via the storage
      // listener below. Restoring a snapshot that no longer describes the same
      // set of pins would resurrect pins the user just removed elsewhere, so
      // the newer list wins and only a pure reorder is rolled back.
      if (!sameKeys(snapshot, state.pinned)) return state;
      if (sameOrder(snapshot, state.pinned)) return state;
      return { pinned: snapshot };
    }),
}));

if (typeof window !== "undefined") {
  window.addEventListener("storage", (event) => {
    if (event.key === KEY || event.key === null) {
      usePinnedModelsStore.setState({ pinned: readPinned() });
    }
  });
}
