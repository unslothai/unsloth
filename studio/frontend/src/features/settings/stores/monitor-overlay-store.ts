// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";

import {
  MONITOR_DETAIL_METRICS,
  type MonitorDetailMetric,
  selectedDetailMetrics,
} from "@/components/monitor-details";
import {
  DEFAULT_MONITOR_REFRESH_MS,
  monitorRefreshMs,
} from "@/components/monitor-refresh";

interface MonitorOverlayState {
  isOpen: boolean;
  isMinimized: boolean;
  /** Off by default: the monitor looks as it always has until a user opts in. */
  showDetails: boolean;
  detailMetrics: MonitorDetailMetric[];
  refreshMs: number;
  setIsOpen: (open: boolean) => void;
  toggleMinimized: () => void;
  setShowDetails: (show: boolean) => void;
  /** Splits the memory bar into Unsloth, other apps and free. */
  showUnslothShare: boolean;
  setShowUnslothShare: (show: boolean) => void;
  /** Lists each loaded model's memory and the app's own share. */
  showModelBreakdown: boolean;
  setShowModelBreakdown: (show: boolean) => void;
  setDetailMetric: (metric: MonitorDetailMetric, on: boolean) => void;
  setRefreshMs: (ms: number) => void;
}

export const useMonitorOverlayStore = create<MonitorOverlayState>()(
  persist(
    (set) => ({
      isOpen: false,
      isMinimized: false,
      showDetails: false,
      showUnslothShare: false,
      showModelBreakdown: false,
      detailMetrics: [...MONITOR_DETAIL_METRICS],
      refreshMs: DEFAULT_MONITOR_REFRESH_MS,
      setIsOpen: (isOpen) => set({ isOpen }),
      toggleMinimized: () => set((state) => ({ isMinimized: !state.isMinimized })),
      setShowDetails: (showDetails) => set({ showDetails }),
      setShowUnslothShare: (showUnslothShare) => set({ showUnslothShare }),
      setShowModelBreakdown: (showModelBreakdown) => set({ showModelBreakdown }),
      setDetailMetric: (metric, on) =>
        set((state) => {
          const current = selectedDetailMetrics(state.detailMetrics);
          return {
            detailMetrics: MONITOR_DETAIL_METRICS.filter((m) =>
              m === metric ? on : current.has(m),
            ),
          };
        }),
      setRefreshMs: (ms) => set({ refreshMs: monitorRefreshMs(ms) }),
    }),
    { name: "unsloth_monitor_overlay" }
  )
);
