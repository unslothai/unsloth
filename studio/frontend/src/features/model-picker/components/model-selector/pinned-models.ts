// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pinned models for the model selector's On Device list, persisted in localStorage. GGUF quants
// pin individually (repoId + quant); non-GGUF repos pin as a whole. Pinned entries surface in
// a "Pinned" section above the Unsloth/Downloaded group.

import { create } from "zustand";

const KEY = "unsloth_pinned_models";

// Entries are stored as strings: "repoId" pins a whole (non-GGUF) repo, "repoId::quant" pins
// one GGUF quant. Neither part contains "::".
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

// A drag reorders live under the cursor, so `movePinned` runs on every dragenter. Persisting
// each would write localStorage dozens of times per drag and would make a drag the user
// cancels permanent, since dragend can clear the marker but cannot undo writes. So a drag
// session snapshots the order up front, keeps intermediate moves in memory only, and either
// commits once on drop or restores the snapshot.
let dragSnapshot: string[] | null = null;

// Another window can rewrite the whole list mid-drag through the storage listener below, which
// makes the snapshot stale. Rolling back to it would leave this window silently disagreeing
// with the record, and the next write would persist that stale order over the other window's
// change. So the listener records the order it installed and the session falls back to that.
// Null unless a storage event landed inside the session.
let dragExternalOrder: string[] | null = null;

function sameOrder(a: readonly string[], b: readonly string[]): boolean {
  return a.length === b.length && a.every((key, index) => key === b[index]);
}

interface PinnedModelsState {
  pinned: string[];
  togglePinned: (repoId: string, quant?: string) => void;
  /** Drop a repo's own pin and every per-quant pin under it. A whole-repo delete takes the quants
   *  with it, and a `repoId::quant` pin outlives the row that showed it: nothing lists it, so
   *  nothing can unpin it, and it reappears the day that quant is downloaded again. */
  unpinRepo: (repoId: string) => void;
  /**
   * Move `fromKey` into `toKey`'s slot. Both keys must already be pinned;
   * anything else is a no-op. Outside a drag session the new order is
   * persisted immediately, inside one it is held until `endPinnedDrag`.
   */
  movePinned: (fromKey: string, toKey: string) => void;
  /** Snapshot the current order so a cancelled drag can be undone. */
  beginPinnedDrag: () => void;
  /** End a drag session. `commit` persists the reordered list; otherwise the snapshot taken by
   *  `beginPinnedDrag` is restored. Idempotent, because drop is followed by dragend and only the
   *  first of the two may decide. */
  endPinnedDrag: (commit: boolean) => void;
}

export const usePinnedModelsStore = create<PinnedModelsState>((set) => ({
  pinned: readPinned(),
  togglePinned: (repoId, quant) =>
    set((state) => {
      const key = pinKey(repoId, quant);
      // Newest pin first, so "Pin to top" literally lands on top of the pinned group rather than under earlier pins.
      const next = state.pinned.includes(key)
        ? state.pinned.filter((id) => id !== key)
        : [key, ...state.pinned];
      writePinned(next);
      return { pinned: next };
    }),
  unpinRepo: (repoId) =>
    set((state) => {
      const prefix = `${pinKey(repoId)}::`;
      const next = state.pinned.filter(
        (key) => key !== pinKey(repoId) && !key.startsWith(prefix),
      );
      if (next.length === state.pinned.length) return state;
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
      dragExternalOrder = null;
      return state;
    }),
  endPinnedDrag: (commit) =>
    set((state) => {
      const snapshot = dragSnapshot;
      const external = dragExternalOrder;
      dragSnapshot = null;
      dragExternalOrder = null;
      if (snapshot === null) return state;
      // What this window and localStorage last agreed on: the order another window installed mid-drag
      // if there was one, else the pre-drag snapshot.
      const base = external ?? snapshot;
      if (commit) {
        // Nothing moved on top of that, so there is nothing to persist.
        if (sameOrder(base, state.pinned)) return state;
        writePinned(state.pinned);
        return state;
      }
      // A cancel writes nothing, so it has to land on the order already in localStorage. That is the
      // snapshot for an ordinary drag and the other window's list when one landed mid-drag; the
      // moves this drag previewed on top of it are dropped either way.
      if (sameOrder(base, state.pinned)) return state;
      return { pinned: base };
    }),
}));

if (typeof window !== "undefined") {
  window.addEventListener("storage", (event) => {
    if (event.key === KEY || event.key === null) {
      const next = readPinned();
      // A drag in flight rolls back to this instead of its own snapshot.
      if (dragSnapshot !== null) dragExternalOrder = next;
      usePinnedModelsStore.setState({ pinned: next });
    }
  });
}
