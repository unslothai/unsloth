// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { type RewardRecord, listRewards } from "../api/rewards-api";

// Not persisted: the dataset's first row and the reward library, shared by the RL sections.
interface RlWorkspaceState {
  previewRow: Record<string, unknown> | null;
  setPreviewRow: (row: Record<string, unknown> | null) => void;
  library: RewardRecord[];
  libraryError: string | null;
  refreshLibrary: () => Promise<void>;
}

export const useRlWorkspaceStore = create<RlWorkspaceState>()((set) => ({
  previewRow: null,
  setPreviewRow: (previewRow) => set({ previewRow }),
  library: [],
  libraryError: null,
  refreshLibrary: async () => {
    try {
      set({ library: await listRewards(), libraryError: null });
    } catch (err) {
      set({ libraryError: err instanceof Error ? err.message : String(err) });
    }
  },
}));

export function previewCell(value: unknown): string {
  if (value === null || value === undefined) {
    return "";
  }
  return typeof value === "string" ? value : JSON.stringify(value);
}
