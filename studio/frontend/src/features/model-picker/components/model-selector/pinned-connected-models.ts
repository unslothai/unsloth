// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pinned `external::<connectionId>::<modelId>` ids for the Connected list, in localStorage.
// NOT the On Device store in pinned-models.ts: an external id contains "::", which that store
// reads as its repoId/quant separator and surfaces as a phantom pinned quant on the On Device tab.

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

/** The stored list, or null for "nothing to read". Distinct from []: a toggle falling back to []
 *  would drop this window's own pins where every write has failed. */
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

// A write that failed (quota exhausted with the key already present) leaves the RECORD older than
// this session's list, so re-reading it as a base drops the pins that never persisted. Until a
// write succeeds again this window's own list is the newer one.
let storageWritable = true;

function writePinned(pinned: string[]): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(pinned));
    storageWritable = true;
  } catch {
    storageWritable = false;
  }
}

/** The record to apply an edit to. While writes are failing the record is still FRESH for other
 *  windows and only stale for this one's own unpersisted pins, so it is merged rather than
 *  dropped: another window's additions come in, and nothing this session pinned is lost. Its
 *  removals cannot be honoured until a write succeeds, which re-adds rather than deletes. */
function persistedBase(fallback: readonly string[]): string[] {
  const stored = storedPinned();
  if (stored === null) return [...fallback];
  if (storageWritable) return stored;
  const added = stored.filter((id) => !fallback.includes(id));
  return [...added, ...fallback];
}

// movePinnedConnected runs on every dragenter: writing each would make a cancelled drag permanent.
// Snapshot, move in memory, commit on drop. Mirrors pinned-models.ts.
let dragSnapshot: string[] | null = null;

// Keep remote updates for cancellation without replacing the drag's live order.
let dragExternalOrder: string[] | null = null;

function sameOrder(a: readonly string[], b: readonly string[]): boolean {
  return a.length === b.length && a.every((key, index) => key === b[index]);
}

/** A dragged order over the stored membership: this window owns row ORDER, the record owns which
 *  ids are pinned, since another window's storage event may still be in flight at the drop.
 *  Its additions arrive at the front; null storage leaves the order alone. */
function rebaseOnStored(order: readonly string[]): string[] {
  const stored = storedPinned();
  if (stored === null) return [...order];
  const added = stored.filter((id) => !order.includes(id));
  // Writes failing: keep the whole dragged order, since the record is missing this session's own
  // pins and filtering against it would delete them, and still take the other window's additions.
  if (!storageWritable) return [...added, ...order];
  const kept = order.filter((id) => stored.includes(id));
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
        const base = persistedBase(state.pinned);
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
        // Read storage too, since its event may still be pending. persistedBase keeps this
        // session's unpersisted pins while writes are failing without dropping the other window's.
        const base = persistedBase(external ?? snapshot);
        if (commit && !sameOrder(snapshot, state.pinned)) {
          if (sameOrder(base, state.pinned)) return state;
          // The write replaces the whole list and nothing echoes it back to this window, so a
          // plain write of the dragged order would erase a pin another window added mid-drag
          // with no event left to restore it.
          const next = rebaseOnStored(state.pinned);
          writePinned(next);
          return sameOrder(next, state.pinned) ? state : { pinned: next };
        }
        // Cancellation and unchanged drags use the latest persisted order.
        if (sameOrder(base, state.pinned)) return state;
        return { pinned: base };
      }),
  }),
);

if (typeof window !== "undefined") {
  window.addEventListener("storage", (event) => {
    if (event.key === KEY || event.key === null) {
      const next = readPinned();
      if (dragSnapshot !== null) {
        dragExternalOrder = next;
        return;
      }
      usePinnedConnectedModelsStore.setState({ pinned: next });
    }
  });
}
