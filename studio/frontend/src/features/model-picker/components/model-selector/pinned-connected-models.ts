// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Separate from pinned-models.ts: "::" is that store's repoId/quant separator, so external ids
// would show as phantom pinned quants on On Device.

import { create } from "zustand";

import { mirrorPins, onPinsRestored } from "../../../../lib/pins-mirror.ts";

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

/** `absent` is a peer's reset and unpins; `unreadable` (revoked or unparseable) must change nothing. */
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

// Ids this window pinned but could not persist; tracked, not inferred from what the record lacks.
let storageWritable = true;
let unpersisted = new Set<string>();

function writePinned(pinned: string[], added: string | null = null): void {
  mirrorPins("connected", pinned);
  try {
    localStorage.setItem(KEY, JSON.stringify(pinned));
    storageWritable = true;
    unpersisted.clear();
    return;
  } catch {
    storageWritable = false;
  }
  // Only `added` is certainly unpublished; re-reading the record would resurrect peer removals.
  const carried = new Set([...unpersisted].filter((id) => pinned.includes(id)));
  if (added !== null) carried.add(added);
  unpersisted = carried;
}

/** Read from the set: a peer's storage event replaces this window's array wholesale. */
function ourPins(stored: readonly string[], present: readonly string[]): string[] {
  if (storageWritable) return [];
  return [...unpersisted].filter(
    (id) => !stored.includes(id) && !present.includes(id),
  );
}

function isOurs(id: string): boolean {
  return !storageWritable && unpersisted.has(id);
}

/** An id the record carries is no longer ours to re-add, else a peer's pin+unpin resurrects it. */
function retirePersisted(stored: readonly string[]): void {
  if (unpersisted.size === 0) return;
  for (const id of stored) unpersisted.delete(id);
}

/** Fresh for other windows, stale only for our unpersisted pins, so it is merged. */
function persistedBase(fallback: readonly string[]): string[] {
  const record = storedRecord();
  if (record.kind !== "list") {
    return record.kind === "absent" ? fallback.filter(isOurs) : [...fallback];
  }
  const stored = record.ids;
  retirePersisted(stored);
  // Nothing of ours unwritten: the record wins, else a peer's reorder is overwritten on next toggle.
  if (storageWritable || unpersisted.size === 0) return stored;
  return rebaseOnStored(fallback);
}

// movePinnedConnected runs on every dragenter: writing each would make a cancelled drag permanent.
let dragSnapshot: string[] | null = null;

function sameOrder(a: readonly string[], b: readonly string[]): boolean {
  return a.length === b.length && a.every((key, index) => key === b[index]);
}

/** This window owns row ORDER, the record owns membership. Additions arrive at the front. */
function rebaseOnStored(order: readonly string[]): string[] {
  const record = storedRecord();
  if (record.kind !== "list") {
    return record.kind === "absent" ? order.filter(isOurs) : [...order];
  }
  const stored = record.ids;
  retirePersisted(stored);
  const added = stored.filter((id) => !order.includes(id));
  const kept = order.filter((id) => stored.includes(id) || isOurs(id));
  return [...ourPins(stored, [...added, ...kept]), ...added, ...kept];
}

interface PinnedConnectedModelsState {
  pinned: string[];
  togglePinnedConnected: (modelId: string) => void;
  movePinnedConnected: (fromId: string, toId: string) => void;
  beginPinnedConnectedDrag: () => void;
  /** Idempotent: drop is followed by dragend and only the first may decide. */
  endPinnedConnectedDrag: (commit: boolean) => void;
}

export const usePinnedConnectedModelsStore = create<PinnedConnectedModelsState>(
  (set) => ({
    pinned: readPinned(),
    togglePinnedConnected: (modelId) =>
      set((state) => {
        // Apply the edit to the record: rewriting this window's array would drop a newer peer pin.
        const base = persistedBase(state.pinned);
        // Direction from the rendered row, else an unpin could become a pin.
        const unpinning = state.pinned.includes(modelId);
        const without = base.filter((id) => id !== modelId);
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
        // The pre-drag order holds this window's unpersisted pins where they are drawn.
        const base = persistedBase(snapshot);
        if (commit && !sameOrder(snapshot, state.pinned)) {
          if (sameOrder(base, state.pinned)) return state;
          // No event echoes this write, so rebase or a peer's mid-drag pin is erased.
          const next = rebaseOnStored(state.pinned);
          writePinned(next);
          return sameOrder(next, state.pinned) ? state : { pinned: next };
        }
        if (sameOrder(base, state.pinned)) return state;
        return { pinned: base };
      }),
  }),
);

function reloadFromRecord(): void {
  // The RECORD retires a pin, never event.newValue: equal payloads can hide a peer pin+unpin.
  const next = readPinned();
  retirePersisted(next);
  if (dragSnapshot !== null) return;
  // Same rule as toggle: an unconditional merge loses a peer's pure reorder.
  usePinnedConnectedModelsStore.setState({
    pinned: persistedBase(usePinnedConnectedModelsStore.getState().pinned),
  });
}

onPinsRestored("connected", reloadFromRecord);

if (typeof window !== "undefined") {
  window.addEventListener("storage", (event) => {
    if (event.key === KEY || event.key === null) reloadFromRecord();
  });
}
