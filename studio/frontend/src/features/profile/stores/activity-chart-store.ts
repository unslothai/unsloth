// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";

export const ACTIVITY_CHART_STORAGE_KEY = "unsloth_profile_activity_chart";

interface ActivityChartState {
  showChart: boolean;
  setShowChart: (showChart: boolean) => void;
}

/** Opt-in: Weekly and Cumulative keep the block grid until the user picks Chart. */
export const useActivityChartStore = create<ActivityChartState>()(
  persist(
    (set) => ({
      showChart: false,
      setShowChart: (showChart) => set({ showChart }),
    }),
    { name: ACTIVITY_CHART_STORAGE_KEY },
  ),
);
