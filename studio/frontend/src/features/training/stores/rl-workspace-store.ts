// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  AUTH_SESSION_CLEARED_EVENT,
  getAuthSessionEpoch,
} from "@/features/auth";
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
    // A reply that outlives a sign-out belongs to the previous account.
    const epoch = getAuthSessionEpoch();
    try {
      const library = await listRewards();
      if (epoch === getAuthSessionEpoch()) {
        set({ library, libraryError: null });
      }
    } catch (err) {
      if (epoch === getAuthSessionEpoch()) {
        set({ libraryError: err instanceof Error ? err.message : String(err) });
      }
    }
  },
}));

// A sign-out leaves the app mounted: the next account must not see this one's rewards or rows.
if (typeof window !== "undefined") {
  window.addEventListener(AUTH_SESSION_CLEARED_EVENT, () =>
    useRlWorkspaceStore.setState({
      previewRow: null,
      library: [],
      libraryError: null,
    }),
  );
}

export function previewCell(value: unknown): string {
  if (value === null || value === undefined) {
    return "";
  }
  return typeof value === "string" ? value : JSON.stringify(value);
}
