// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

// Here, not in the journal, so the startup hook can read it without loading the sync runtime.
export const PIN_JOURNAL_PREFIX = "unsloth_chat_pin_journal:";

export interface PinnedChatsActions {
  setPinned: (ids: string[], pinned: boolean) => void;
}

export interface PinnedChatsState extends PinnedChatsActions {
  /** Stable chat IDs (or pair IDs for comparisons), in manual pin order. */
  pinnedIds: string[];
  togglePin: (id: string) => void;
  unpin: (id: string) => void;
}

let actions: PinnedChatsActions | null = null;

// Edits made before the lazy runtime binds; they survive a reload. Memory when storage is absent.
const UNBOUND_KEY = "unsloth_chat_pin_unbound";
let unboundMemory: Array<[string[], boolean]> = [];

function readUnbound(): Array<[string[], boolean]> {
  try {
    if (typeof localStorage === "undefined") return unboundMemory;
    const saved: unknown = JSON.parse(localStorage.getItem(UNBOUND_KEY) ?? "[]");
    return Array.isArray(saved)
      ? saved.filter(
          (entry): entry is [string[], boolean] =>
            Array.isArray(entry) &&
            Array.isArray(entry[0]) &&
            entry[0].every((id: unknown) => typeof id === "string") &&
            typeof entry[1] === "boolean",
        )
      : [];
  } catch {
    return unboundMemory;
  }
}

function writeUnbound(edits: Array<[string[], boolean]>): void {
  unboundMemory = edits;
  try {
    if (typeof localStorage === "undefined") return;
    if (edits.length) localStorage.setItem(UNBOUND_KEY, JSON.stringify(edits));
    else localStorage.removeItem(UNBOUND_KEY);
  } catch {
    // Storage full or blocked: the edit still reaches the runtime if it binds this session.
  }
}

/** Sign-out: drop edits made before the runtime bound and the pins they showed. */
export function discardUnboundPinEdits(): void {
  writeUnbound([]);
  usePinnedChatsStore.setState({ pinnedIds: [] });
}

/** Bind persistence at runtime without pulling API/auth modules into this state store. */
export function setPinnedChatsActions(next: PinnedChatsActions | null): void {
  actions = next;
  if (!next) return;
  const queued = readUnbound();
  writeUnbound([]);
  for (const [ids, pinned] of queued) next.setPinned(ids, pinned);
}

export const usePinnedChatsStore = create<PinnedChatsState>((set, get) => ({
  pinnedIds: [],
  togglePin: (id) => get().setPinned([id], !get().pinnedIds.includes(id)),
  unpin: (id) => get().setPinned([id], false),
  setPinned: (ids, pinned) => {
    if (actions) {
      actions.setPinned(ids, pinned);
      return;
    }
    writeUnbound([...readUnbound(), [ids, pinned]]);
    set((state) => {
      if (!pinned) {
        const dropping = new Set(ids);
        const next = state.pinnedIds.filter((id) => !dropping.has(id));
        return next.length === state.pinnedIds.length
          ? state
          : { pinnedIds: next };
      }
      const additions = [...new Set(ids)].filter(
        (id) => !state.pinnedIds.includes(id),
      );
      return additions.length === 0
        ? state
        : { pinnedIds: [...additions, ...state.pinnedIds] };
    });
  },
}));
