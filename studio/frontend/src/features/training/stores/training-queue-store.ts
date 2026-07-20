// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import type { TrainingQueueItem, TrainingQueueState } from "../types/queue";

// Not persisted: the backend queue is the source of truth; this store mirrors
// the latest GET /api/train/queue snapshot.
interface TrainingQueueStoreState {
  items: TrainingQueueItem[];
  paused: boolean;
  pausedReason: "restart" | "user" | null;
  pendingCount: number;
  maxPending: number;
  activeJobId: string | null;
  hasHydrated: boolean;
  restartBannerDismissed: boolean;
}

interface TrainingQueueStore extends TrainingQueueStoreState {
  applyState: (state: TrainingQueueState) => void;
  dismissRestartBanner: () => void;
  reset: () => void;
}

const initialState: TrainingQueueStoreState = {
  items: [],
  paused: false,
  pausedReason: null,
  pendingCount: 0,
  maxPending: 5,
  activeJobId: null,
  hasHydrated: false,
  restartBannerDismissed: false,
};

export const useTrainingQueueStore = create<TrainingQueueStore>()((set) => ({
  ...initialState,

  applyState: (state) =>
    set({
      items: state.items,
      paused: state.paused,
      pausedReason: state.paused_reason,
      pendingCount: state.pending_count,
      maxPending: state.max_pending,
      activeJobId: state.active_job_id,
      hasHydrated: true,
    }),

  dismissRestartBanner: () => set({ restartBannerDismissed: true }),

  reset: () => set(initialState),
}));
