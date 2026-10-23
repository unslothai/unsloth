// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";

import {
  MONITOR_DETAIL_METRICS,
  type MonitorDetailMetric,
  selectedDetailMetrics,
} from "@/components/monitor-details";

interface MonitorOverlayState {
  isOpen: boolean;
  isMinimized: boolean;
  /** Off by default: the monitor looks as it always has until a user opts in. */
  showDetails: boolean;
  detailMetrics: MonitorDetailMetric[];
  setIsOpen: (open: boolean) => void;
  toggleMinimized: () => void;
  setShowDetails: (show: boolean) => void;
  setDetailMetric: (metric: MonitorDetailMetric, on: boolean) => void;
}

export const useMonitorOverlayStore = create<MonitorOverlayState>()(
  persist(
    (set) => ({
      isOpen: false,
      isMinimized: false,
      showDetails: false,
      detailMetrics: [...MONITOR_DETAIL_METRICS],
      setIsOpen: (isOpen) => set({ isOpen }),
      toggleMinimized: () => set((state) => ({ isMinimized: !state.isMinimized })),
      setShowDetails: (showDetails) => set({ showDetails }),
      setDetailMetric: (metric, on) =>
        set((state) => {
          const current = selectedDetailMetrics(state.detailMetrics);
          return {
            detailMetrics: MONITOR_DETAIL_METRICS.filter((m) =>
              m === metric ? on : current.has(m),
            ),
          };
        }),
    }),
    { name: "unsloth_monitor_overlay" }
  )
);
