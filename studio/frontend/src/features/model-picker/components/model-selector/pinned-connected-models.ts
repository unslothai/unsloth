// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pinned `external::<connectionId>::<modelId>` ids for the Connected list, in localStorage.
// NOT pinned-models.ts: "::" is that store's repoId/quant separator, so an external id lands on the
// On Device tab as a phantom pinned quant.

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

/** `absent` is a peer's reset and unpins what the record held; `unreadable` (access revoked, or an
 *  unparseable value) is this window losing its eyes and must change nothing. A single null for
 *  both made a revoked read unpin everything. */
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

// Ids this window pinned and could not persist. The record stays authoritative for everything
// else, peer removals included, so these are tracked rather than inferred from what it lacks.
let storageWritable = true;
let unpersisted = new Set<string>();

/** `added` is the id this edit introduced, or null for a reorder, an unpin or a drag commit. */
function writePinned(pinned: string[], added: string | null = null): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(pinned));
    storageWritable = true;
    unpersisted.clear();
    return;
  } catch {
    storageWritable = false;
  }
  // Only `added` is certainly unpublished: re-reading the record here instead claimed ids a peer
  // removed between that read and this failed write, and the next write that landed undid them.
  // The intersection is the undo rule, so unpinning while writes fail takes the id back out.
  const carried = new Set([...unpersisted].filter((id) => pinned.includes(id)));
  if (added !== null) carried.add(added);
  unpersisted = carried;
}

/** Read from the set, not the caller's list: a peer's storage event replaces this window's array
 *  wholesale, so by the next edit the list no longer carries them. */
function ourPins(stored: readonly string[], present: readonly string[]): string[] {
  if (storageWritable) return [];
  return [...unpersisted].filter(
    (id) => !stored.includes(id) && !present.includes(id),
  );
}

function isOurs(id: string): boolean {
  return !storageWritable && unpersisted.has(id);
}

/** An id the record carries is no longer ours to re-add, whoever wrote it, else a peer pinning then
 *  unpinning it resurrects the pin. From every read, not just the handler: an unchanged drag sees a
 *  peer's write before its event arrives. */
function retirePersisted(stored: readonly string[]): void {
  if (unpersisted.size === 0) return;
  for (const id of stored) unpersisted.delete(id);
}

/** The record to apply an edit to. It is FRESH for other windows and stale only for our own
 *  unpersisted pins, so it is merged rather than dropped. */
function persistedBase(fallback: readonly string[]): string[] {
  const record = storedRecord();
  if (record.kind !== "list") {
    return record.kind === "absent" ? fallback.filter(isOurs) : [...fallback];
  }
  const stored = record.ids;
  retirePersisted(stored);
  // With nothing of ours unwritten the record wins outright: storageWritable describes OUR writes,
  // and holding the rendered order past a peer's reorder would overwrite it on the next toggle.
  if (storageWritable || unpersisted.size === 0) return stored;
  return rebaseOnStored(fallback);
}

// movePinnedConnected runs on every dragenter: writing each would make a cancelled drag permanent.
let dragSnapshot: string[] | null = null;

function sameOrder(a: readonly string[], b: readonly string[]): boolean {
  return a.length === b.length && a.every((key, index) => key === b[index]);
}

/** A rendered order over the stored membership: this window owns row ORDER, the record owns which
 *  ids are pinned. Its additions arrive at the front. */
function rebaseOnStored(order: readonly string[]): string[] {
  // A reset unpins what the record held, and only pins nothing else ever recorded survive it.
  const record = storedRecord();
  if (record.kind !== "list") {
    return record.kind === "absent" ? order.filter(isOurs) : [...order];
  }
  const stored = record.ids;
  retirePersisted(stored);
  const added = stored.filter((id) => !order.includes(id));
  // Survives if the record carries it or it is ours; a peer's removal is honoured either way.
  const kept = order.filter((id) => stored.includes(id) || isOurs(id));
  return [...ourPins(stored, [...added, ...kept]), ...added, ...kept];
}

interface PinnedConnectedModelsState {
  pinned: string[];
  togglePinnedConnected: (modelId: string) => void;
  /** Move `fromId` into `toId`'s slot; anything not already pinned is a no-op. Inside a drag the
   *  new order is held until `endPinnedConnectedDrag`. */
  movePinnedConnected: (fromId: string, toId: string) => void;
  beginPinnedConnectedDrag: () => void;
  /** `commit` persists the reordered list, else the snapshot is restored. Idempotent: drop is
   *  followed by dragend and only the first may decide. */
  endPinnedConnectedDrag: (commit: boolean) => void;
}

export const usePinnedConnectedModelsStore = create<PinnedConnectedModelsState>(
  (set) => ({
    pinned: readPinned(),
    togglePinnedConnected: (modelId) =>
      set((state) => {
        // A write replaces the whole list, so the edit applies to the record: rewriting this
        // window's array would drop a peer's pin newer than the event we have processed.
        const base = persistedBase(state.pinned);
        // Direction from the ROW, which draws state.pinned. A base that no longer holds the model
        // turns an unpin into a pin and writes back what the user was removing.
        const unpinning = state.pinned.includes(modelId);
        const without = base.filter((id) => id !== modelId);
        // Newest first, as On Device does, so "Pin to top" lands on top of the pinned group.
        const next = unpinning ? without : [modelId, ...without];
        writePinned(next, unpinning ? null : modelId);
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
        // The PRE-DRAG rendered order: only it holds this window's unpersisted pins where they are
        // drawn, and the record is read live below rather than stashed from an event.
        const base = persistedBase(snapshot);
        if (commit && !sameOrder(snapshot, state.pinned)) {
          if (sameOrder(base, state.pinned)) return state;
          // Nothing echoes the write back here, so a plain write of the dragged order would erase
          // a pin another window added mid-drag with no event left to restore it.
          const next = rebaseOnStored(state.pinned);
          writePinned(next);
          return sameOrder(next, state.pinned) ? state : { pinned: next };
        }
        if (sameOrder(base, state.pinned)) return state;
        return { pinned: base };
      }),
  }),
);

if (typeof window !== "undefined") {
  window.addEventListener("storage", (event) => {
    if (event.key === KEY || event.key === null) {
      // The RECORD retires a pin, never event.newValue: a peer pinning and unpinning B before our
      // failed attempt and after it deliver the same payloads, and on that tie the pin the user
      // just made is the one worth keeping.
      const next = readPinned();
      retirePersisted(next);
      // A drag owns the rendered order until it ends, and reads the record live then.
      if (dragSnapshot !== null) return;
      // The toggle's rule: merging unconditionally instead lost a peer's pure REORDER, which adds
      // no ids and so changed nothing on a window with nothing unwritten.
      usePinnedConnectedModelsStore.setState({
        pinned: persistedBase(usePinnedConnectedModelsStore.getState().pinned),
      });
    }
  });
}
