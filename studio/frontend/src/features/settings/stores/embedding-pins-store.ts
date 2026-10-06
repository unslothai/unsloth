// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";

/** Embedding models pinned to the RAG menu for quick switching. Per browser. */
interface EmbeddingPinsState {
  pinned: string[];
  togglePin: (model: string) => void;
}

export const useEmbeddingPinsStore = create<EmbeddingPinsState>()(
  persist(
    (set) => ({
      pinned: [],
      togglePin: (model) =>
        set((state) => {
          const id = model.trim();
          if (!id) return state;
          return {
            pinned: state.pinned.includes(id)
              ? state.pinned.filter((pin) => pin !== id)
              : [...state.pinned, id],
          };
        }),
    }),
    { name: "unsloth_embedding_pins" },
  ),
);
