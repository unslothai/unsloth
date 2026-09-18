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

/** What the record says, with the three answers kept apart. A single null conflated them, and the
 *  two that are not a list mean opposite things: `absent` is a peer's reset and unpins what it
 *  held, while `unreadable` (storage access revoked mid-session, or a value no one can parse) is
 *  this window losing its eyes and must change nothing on screen. */
type StoredRecord =
  | { readonly kind: "list"; readonly ids: string[] }
  | { readonly kind: "absent" }
  | { readonly kind: "unreadable" };

function storedRecord(): StoredRecord {
  let raw: string | null;
  try {
    raw = localStorage.getItem(KEY);
  } catch {
    return { kind: "unreadable" };
  }
  if (raw === null) return { kind: "absent" };
  try {
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed)
      ? {
          kind: "list",
          ids: parsed.filter((v): v is string => typeof v === "string"),
        }
      : { kind: "unreadable" };
  } catch {
    return { kind: "unreadable" };
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
    // Only a record we can READ narrows this: if storage cannot be read either, every pin on
    // screen is one this window is carrying, which is exactly what it is.
    const record = storedRecord();
    const stored = record.kind === "list" ? record.ids : [];
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
  const record = storedRecord();
  if (record.kind !== "list") {
    return record.kind === "absent" ? fallback.filter(isOurs) : [...fallback];
  }
  const stored = record.ids;
  retirePersisted(stored);
  // Nothing of ours is unwritten, so the record is authoritative exactly as it is while writes
  // land: a peer that persisted our failed pin may also have REORDERED since, and holding on to
  // this window's rendered order merely because an older write failed would overwrite that
  // reorder on the next toggle that does land.
  if (storageWritable || unpersisted.size === 0) return stored;
  // The SAME merge the drag commit uses, applied to the rendered list, so the two paths cannot
  // disagree about order: prepending every unpersisted pin to the record instead put an older
  // failed pin back above a peer addition the handler had just placed correctly, and the next
  // successful write made that durable.
  return rebaseOnStored(fallback);
}

// movePinnedConnected runs on every dragenter: writing each would make a cancelled drag permanent.
// Snapshot, move in memory, commit on drop. Mirrors pinned-models.ts.
let dragSnapshot: string[] | null = null;

function sameOrder(a: readonly string[], b: readonly string[]): boolean {
  return a.length === b.length && a.every((key, index) => key === b[index]);
}

/** A dragged order over the stored membership: this window owns row ORDER, the record owns which
 *  ids are pinned, since another window's storage event may still be in flight at the drop.
 *  Its additions arrive at the front; null storage leaves the order alone. */
function rebaseOnStored(order: readonly string[]): string[] {
  // A record that is GONE is a peer's reset (removeItem, or a clear() of the whole origin), not
  // "nothing to read": everything it held is unpinned now. Only the pins this window is carrying
  // unwritten survive it, because nothing else ever recorded them. Keeping the rendered list here
  // let the next write that landed put a peer's cleared pins straight back. A record we could not
  // READ is the opposite case and leaves the order alone.
  const record = storedRecord();
  if (record.kind !== "list") {
    return record.kind === "absent" ? order.filter(isOurs) : [...order];
  }
  const stored = record.ids;
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
        return state;
      }),
    endPinnedConnectedDrag: (commit) =>
      set((state) => {
        const snapshot = dragSnapshot;
        dragSnapshot = null;
        if (snapshot === null) return state;
        // The PRE-DRAG rendered order, not the order the event carried. Both reach the same
        // record, since persistedBase reads storage live rather than waiting on an event, but only
        // the rendered one holds this window's unpersisted pins where they are actually drawn: the
        // event's list lacks them entirely, so they were re-added at the front, above peer pins
        // that are newer than they are.
        const base = persistedBase(snapshot);
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
      // Only the RECORD retires a pin, never a queued payload. A payload carrying an id says
      // nothing about whether that write happened before or after this window's failed attempt:
      // "we failed to pin B, then a peer pinned and unpinned it" and "a peer pinned and unpinned
      // B, then we failed to pin it" deliver the same two events over the same final record, so
      // no rule can tell them apart. Retiring on the payload silently drops the pin the user just
      // made in the second ordering; keeping it leaves a session-only pin this window's user did
      // ask for. The pin the user made is the one worth being wrong about.
      const next = readPinned();
      retirePersisted(next);
      // A drag owns the rendered order until it ends; the record it is rebased onto is read
      // live at that point, so there is nothing to stash here.
      if (dragSnapshot !== null) return;
      // The same rule the toggle uses, which is the whole point of it living in one function: the
      // record wins outright unless this window is carrying pins the record lacks, and only then
      // is it merged into the rendered order. Merging unconditionally cost a peer's pure REORDER,
      // which adds no ids and so changed nothing on a window with nothing unwritten.
      usePinnedConnectedModelsStore.setState({
        pinned: persistedBase(usePinnedConnectedModelsStore.getState().pinned),
      });
    }
  });
}
