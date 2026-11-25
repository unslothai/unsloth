// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

import { looksLikeLocalPath } from "../../../../lib/local-path.ts";
import { mirrorPins, onPinsRestored } from "../../../../lib/pins-mirror.ts";

const KEY = "unsloth_pinned_models";

// "repoId" pins a whole non-GGUF repo, "repoId::quant" one GGUF quant.
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

export function pinCoversId(pinned: readonly string[], id: string): boolean {
  const base = id.trim().toLowerCase();
  return pinned.some((key) => {
    const k = key.trim().toLowerCase();
    return k === base || k.startsWith(`${base}::`);
  });
}

/** Ranks a repo by its own pin, the one a drag moves, else by the first pin that covers it. */
export function makeRepoPinRank(
  pinned: readonly string[],
): (repoId: string) => number {
  const own = makePinRank(pinned);
  const covering = new Map<string, number>();
  pinned.forEach((key, index) => {
    const k = key.trim().toLowerCase();
    for (let at = k.indexOf("::"); at >= 0; at = k.indexOf("::", at + 1)) {
      if (!covering.has(k.slice(0, at))) covering.set(k.slice(0, at), index);
    }
    if (!covering.has(k)) covering.set(k, index);
  });
  return (repoId) => {
    const rank = own(repoId);
    return rank < Number.MAX_SAFE_INTEGER
      ? rank
      : (covering.get(repoId.trim().toLowerCase()) ?? Number.MAX_SAFE_INTEGER);
  };
}

/** Key to `movePinned` onto so `fromKey` lands on `edge` of `targetKey`, or null if nothing moves. */
export function pinDropAnchor(
  pinned: readonly string[],
  fromKey: string,
  targetKey: string,
  edge: "top" | "bottom",
): string | null {
  const from = pinned.indexOf(fromKey);
  const to = pinned.indexOf(targetKey);
  if (from < 0 || to < 0 || from === to) return null;
  const slot =
    from < to
      ? edge === "bottom"
        ? to
        : to - 1
      : edge === "top"
        ? to
        : to + 1;
  return slot === from ? null : (pinned[slot] ?? null);
}

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
  mirrorPins("pinned", pinned);
  try {
    localStorage.setItem(KEY, JSON.stringify(pinned));
  } catch {
    // Ignore unavailable storage; the server copy still has them.
  }
}

// movePinned runs on every dragenter: a drag snapshots, moves in memory, and commits once on drop
// or restores the snapshot, so a cancelled drag writes nothing.
let dragSnapshot: string[] | null = null;

// Order a storage event installed mid-drag; cancelling falls back to it instead of the stale snapshot.
let dragExternalOrder: string[] | null = null;

function sameOrder(a: readonly string[], b: readonly string[]): boolean {
  return a.length === b.length && a.every((key, index) => key === b[index]);
}

interface PinnedModelsState {
  pinned: string[];
  togglePinned: (repoId: string, quant?: string) => void;
  /** Also drops per-quant pins, which would otherwise outlive the row and be impossible to unpin. */
  unpinRepo: (repoId: string) => void;
  toggleRepoPinned: (repoId: string) => void;
  replacePinned: (fromKey: string, toKey: string) => void;
  /** Both keys must be pinned. Persisted immediately outside a drag, else held until endPinnedDrag. */
  movePinned: (fromKey: string, toKey: string) => void;
  beginPinnedDrag: () => void;
  /** Idempotent: drop is followed by dragend and only the first may decide. */
  endPinnedDrag: (commit: boolean) => void;
}

export const usePinnedModelsStore = create<PinnedModelsState>((set, get) => ({
  pinned: readPinned(),
  togglePinned: (repoId, quant) =>
    set((state) => {
      const key = pinKey(repoId, quant);
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
  toggleRepoPinned: (repoId) => {
    const { pinned } = get();
    if (!pinCoversId(pinned, repoId)) return get().togglePinned(repoId);
    // A local path keeps its letter case; a Hub id clears every pin it was counted by.
    if (looksLikeLocalPath(repoId)) return get().unpinRepo(repoId);
    const next = pinned.filter((key) => !pinCoversId([key], repoId));
    writePinned(next);
    set({ pinned: next });
  },
  replacePinned: (fromKey, toKey) =>
    set((state) => {
      if (!state.pinned.includes(fromKey) || fromKey === toKey) return state;
      const next = state.pinned.includes(toKey)
        ? state.pinned.filter((key) => key !== fromKey)
        : state.pinned.map((key) => (key === fromKey ? toKey : key));
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
      const base = external ?? snapshot;
      if (commit) {
        if (sameOrder(base, state.pinned)) return state;
        writePinned(state.pinned);
        return state;
      }
      // A cancel writes nothing, so it must land on the order already in localStorage.
      if (sameOrder(base, state.pinned)) return state;
      return { pinned: base };
    }),
}));

onPinsRestored("pinned", () =>
  usePinnedModelsStore.setState({ pinned: readPinned() }),
);

if (typeof window !== "undefined") {
  window.addEventListener("storage", (event) => {
    if (event.key === KEY || event.key === null) {
      const next = readPinned();
      if (dragSnapshot !== null) dragExternalOrder = next;
      usePinnedModelsStore.setState({ pinned: next });
    }
  });
}
