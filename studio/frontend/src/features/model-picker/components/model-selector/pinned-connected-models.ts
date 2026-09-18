// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pinned models for the model selector's Connected list, persisted in localStorage. Entries are
// whole `external::<connectionId>::<modelId>` ids, and they surface in a "Pinned" group above the
// provider groups.
//
// Deliberately not the On Device store in pinned-models.ts: an external id contains "::", which
// that store reads as the repoId/quant separator, so a pin filed there comes back out of
// pinnedQuantEntries as a phantom pinned quant on the On Device tab.

import { create } from "zustand";

const KEY = "unsloth_pinned_connected_models";

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

/** The stored list, or null when there is nothing to read: no key yet, or storage unavailable.
 *  Distinct from an empty list, since a toggle falling back to [] would drop this window's own
 *  pins on an install where every write has failed. */
function storedPinned(): string[] | null {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw === null) return null;
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed)
      ? parsed.filter((v): v is string => typeof v === "string")
      : null;
  } catch {
    return null;
  }
}

function writePinned(pinned: string[]): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(pinned));
  } catch {
    // Ignore unavailable storage; pins stay session-only.
  }
}

// A drag reorders live under the cursor, so movePinnedConnected runs on every dragenter. Writing
// each one would hit localStorage dozens of times per drag and make a cancelled drag permanent.
// So a session snapshots the order, keeps moves in memory, then commits on drop or restores.
// Mirrors pinned-models.ts.
let dragSnapshot: string[] | null = null;

// Another window can rewrite the list mid-drag, which makes the snapshot stale. The listener
// records the order it installed and the session rolls back to that instead.
let dragExternalOrder: string[] | null = null;

function sameOrder(a: readonly string[], b: readonly string[]): boolean {
  return a.length === b.length && a.every((key, index) => key === b[index]);
}

/** A dragged order over the stored membership. This window owns the order of the rows it dragged;
 *  which ids are pinned belongs to the record, since another window can pin or unpin during the
 *  drag and its storage event may still be in flight at the drop. So another window's additions
 *  come in at the front, where a new pin goes, and anything it unpinned stays unpinned.
 *  Unavailable storage reads as null and leaves the order alone, keeping pins session-only. */
function rebaseOnStored(order: readonly string[]): string[] {
  const stored = storedPinned();
  if (stored === null) return [...order];
  const kept = order.filter((id) => stored.includes(id));
  const added = stored.filter((id) => !order.includes(id));
  return [...added, ...kept];
}

interface PinnedConnectedModelsState {
  pinned: string[];
  /** Pin or unpin one external model id. */
  togglePinnedConnected: (modelId: string) => void;
  /**
   * Move `fromId` into `toId`'s slot. Both must already be pinned; anything else is a no-op.
   * Outside a drag session the new order persists immediately, inside one it is held until
   * `endPinnedConnectedDrag`.
   */
  movePinnedConnected: (fromId: string, toId: string) => void;
  /** Snapshot the current order so a cancelled drag can be undone. */
  beginPinnedConnectedDrag: () => void;
  /** End a drag session. `commit` persists the reordered list; otherwise the snapshot is
   *  restored. Idempotent, because drop is followed by dragend and only the first may decide. */
  endPinnedConnectedDrag: (commit: boolean) => void;
}

export const usePinnedConnectedModelsStore = create<PinnedConnectedModelsState>(
  (set) => ({
    pinned: readPinned(),
    togglePinnedConnected: (modelId) =>
      set((state) => {
        // Applied to the stored list, not to this window's copy of it, since a write replaces the
        // whole list: another window's pin can be newer than the storage event this one has
        // processed, and rewriting our own array would drop it for good.
        const base = storedPinned() ?? state.pinned;
        // Newest pin first, as On Device does, so "Pin to top" literally lands on top of the
        // pinned group rather than under earlier pins.
        const next = base.includes(modelId)
          ? base.filter((id) => id !== modelId)
          : [modelId, ...base];
        writePinned(next);
        return { pinned: next };
      }),
    movePinnedConnected: (fromId, toId) =>
      set((state) => {
        const from = state.pinned.indexOf(fromId);
        const to = state.pinned.indexOf(toId);
        if (from < 0 || to < 0 || from === to) return state;
        const next = [...state.pinned];
        next.splice(to, 0, ...next.splice(from, 1));
        if (dragSnapshot === null) writePinned(next);
        return { pinned: next };
      }),
    beginPinnedConnectedDrag: () =>
      set((state) => {
        dragSnapshot = [...state.pinned];
        dragExternalOrder = null;
        return state;
      }),
    endPinnedConnectedDrag: (commit) =>
      set((state) => {
        const snapshot = dragSnapshot;
        const external = dragExternalOrder;
        dragSnapshot = null;
        dragExternalOrder = null;
        if (snapshot === null) return state;
        // What this window and localStorage last agreed on: the order another window installed
        // mid-drag if there was one, else the pre-drag snapshot.
        const base = external ?? snapshot;
        if (commit) {
          if (sameOrder(base, state.pinned)) return state;
          // The write replaces the whole list and nothing echoes it back to this window, so a
          // plain write of the dragged order would erase a pin another window added mid-drag
          // with no event left to restore it.
          const next = rebaseOnStored(state.pinned);
          writePinned(next);
          return sameOrder(next, state.pinned) ? state : { pinned: next };
        }
        // A cancel writes nothing, so it has to land on the order already in localStorage.
        if (sameOrder(base, state.pinned)) return state;
        return { pinned: base };
      }),
  }),
);

if (typeof window !== "undefined") {
  window.addEventListener("storage", (event) => {
    if (event.key === KEY || event.key === null) {
      const next = readPinned();
      // A drag in flight rolls back to this instead of its own snapshot.
      if (dragSnapshot !== null) dragExternalOrder = next;
      usePinnedConnectedModelsStore.setState({ pinned: next });
    }
  });
}
