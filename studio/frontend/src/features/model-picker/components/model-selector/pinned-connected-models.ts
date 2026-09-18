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

// A write that failed (quota exhausted with the key already present) leaves the RECORD without the
// pins it rejected, so re-reading it as a base drops them. Only those ids are ours: the record
// stays authoritative for everything else, including a peer's removals, so the delta is tracked
// rather than inferred from "everything the record does not have", which re-added what a peer
// deleted. Cleared as soon as a write lands, which is when the record carries them again.
let storageWritable = true;
let unpersisted = new Set<string>();

function writePinned(pinned: string[]): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(pinned));
    storageWritable = true;
    unpersisted.clear();
  } catch {
    storageWritable = false;
    // REPLACED, not added to: undoing a failed pin while writes are still failing has to take the
    // id back out, else the next merge resurrects a model the user just unpinned.
    const stored = storedPinned() ?? [];
    unpersisted = new Set(pinned.filter((id) => !stored.includes(id)));
  }
}

/** Ids this window pinned and could not persist, which the record cannot be asked about. Read from
 *  the set rather than from the caller's list: a peer's storage event replaces this window's array
 *  wholesale, so by the next edit the list no longer carries them. */
function ourPins(stored: readonly string[], present: readonly string[]): string[] {
  if (storageWritable) return [];
  // Already newest first: the set is rebuilt from the list the failed write attempted.
  return [...unpersisted].filter(
    (id) => !stored.includes(id) && !present.includes(id),
  );
}

function isOurs(id: string): boolean {
  return !storageWritable && unpersisted.has(id);
}

/** An id the record now carries is no longer ours to re-add, whoever wrote it. Without this a peer
 *  persisting the same model and later unpinning it would see the pin resurrected. Called from
 *  every read of the record, not only the storage handler: a drag that ends unchanged observes a
 *  peer's write synchronously, before its event is delivered. */
function retirePersisted(stored: readonly string[]): void {
  if (unpersisted.size === 0) return;
  for (const id of stored) unpersisted.delete(id);
}

/** The record to apply an edit to. While writes are failing the record is still FRESH for other
 *  windows and only stale for this one's own unpersisted pins, so it is merged rather than
 *  dropped: another window's additions and removals both land, and only the ids this session
 *  pinned and could not persist are carried over. */
function persistedBase(fallback: readonly string[]): string[] {
  const stored = storedPinned();
  if (stored === null) return [...fallback];
  retirePersisted(stored);
  if (storageWritable) return stored;
  return [...ourPins(stored, stored), ...stored];
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
  retirePersisted(stored);
  const added = stored.filter((id) => !order.includes(id));
  // An id survives if the record still carries it, or if it is one of ours the record never
  // received. A peer's removal is honoured either way.
  const kept = order.filter((id) => stored.includes(id) || isOurs(id));
  return [...ourPins(stored, [...added, ...kept]), ...added, ...kept];
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
      retirePersisted(next);
      if (dragSnapshot !== null) {
        dragExternalOrder = next;
        return;
      }
      // Carry this window's unpersisted pins into the rendered list. Without them the row shows as
      // unpinned while every merge still holds it, so "Pin to top" takes the removal branch and
      // persists it unpinned once writes recover.
      usePinnedConnectedModelsStore.setState({
        pinned: [...ourPins(next, next), ...next],
      });
    }
  });
}
