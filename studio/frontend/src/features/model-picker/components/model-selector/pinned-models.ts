// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pinned models for the model selector's On Device list, persisted in
// localStorage so pins survive reloads. GGUF quants pin individually
// (repoId + quant); non-GGUF repos pin as a whole. Pinned entries surface
// in a "Pinned" section above the Unsloth/Downloaded group.

import { create } from "zustand";

const KEY = "unsloth_pinned_models";

// Entries are stored as strings: "repoId" pins a whole (non-GGUF) repo,
// "repoId::quant" pins one GGUF quant. Neither part contains "::".
export function pinKey(repoId: string, quant?: string): string {
  return quant ? `${repoId}::${quant}` : repoId;
}

export interface PinnedQuantEntry {
  repoId: string;
  quant: string;
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

interface PinnedModelsState {
  pinned: string[];
  togglePinned: (repoId: string, quant?: string) => void;
}

export const usePinnedModelsStore = create<PinnedModelsState>((set) => ({
  pinned: readPinned(),
  togglePinned: (repoId, quant) =>
    set((state) => {
      const key = pinKey(repoId, quant);
      // Newest pin first, so "Pin to top" literally lands on top of the
      // pinned group rather than under earlier pins.
      const next = state.pinned.includes(key)
        ? state.pinned.filter((id) => id !== key)
        : [key, ...state.pinned];
      writePinned(next);
      return { pinned: next };
    }),
}));
