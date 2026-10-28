// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

import { mirrorPins, onPinsRestored } from "../../../lib/pins-mirror.ts";

export const EMBEDDING_PINS_STORAGE_KEY = "unsloth_embedding_pins";

/** Embedding models pinned to the RAG menu for quick switching. A plain id list in localStorage,
 *  mirrored to the account so an account switch's purge does not lose it. */
interface EmbeddingPinsState {
  pinned: string[];
  togglePin: (model: string) => void;
}

function readPins(): string[] {
  try {
    const raw: unknown = JSON.parse(localStorage.getItem(EMBEDDING_PINS_STORAGE_KEY) ?? "[]");
    return Array.isArray(raw) ? raw.filter((v): v is string => typeof v === "string") : [];
  } catch {
    return [];
  }
}

function writePins(pinned: string[]): void {
  mirrorPins("embedding", pinned);
  try {
    localStorage.setItem(EMBEDDING_PINS_STORAGE_KEY, JSON.stringify(pinned));
  } catch {
    // Storage unavailable; the account copy still has them.
  }
}

export const useEmbeddingPinsStore = create<EmbeddingPinsState>()((set, get) => ({
  pinned: readPins(),
  togglePin: (model) => {
    const id = model.trim();
    if (!id) return;
    const current = get().pinned;
    const pinned = current.includes(id)
      ? current.filter((pin) => pin !== id)
      : [...current, id];
    writePins(pinned);
    set({ pinned });
  },
}));

onPinsRestored("embedding", () => useEmbeddingPinsStore.setState({ pinned: readPins() }));

if (typeof window !== "undefined") {
  window.addEventListener("storage", (event) => {
    if (event.key === EMBEDDING_PINS_STORAGE_KEY || event.key === null) {
      useEmbeddingPinsStore.setState({ pinned: readPins() });
    }
  });
}
