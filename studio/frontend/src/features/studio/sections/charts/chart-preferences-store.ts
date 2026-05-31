// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import type { OutlierMode, ScaleMode } from "./types";
import { clamp } from "./utils";

type ChartPreferencesState = {
  availableSteps: number;
  windowSize: number | null;
  smoothing: number;
  showRaw: boolean;
  showSmoothed: boolean;
  showAvgLine: boolean;
  lossScale: ScaleMode;
  lrScale: ScaleMode;
  gradScale: ScaleMode;
  lossOutlierMode: OutlierMode;
  gradOutlierMode: OutlierMode;
  lrOutlierMode: OutlierMode;
  setAvailableSteps: (value: number) => void;
  setWindowSize: (value: number | null) => void;
  setSmoothing: (value: number) => void;
  setShowRaw: (value: boolean) => void;
  setShowSmoothed: (value: boolean) => void;
  setShowAvgLine: (value: boolean) => void;
  setLossScale: (value: ScaleMode) => void;
  setLrScale: (value: ScaleMode) => void;
  setGradScale: (value: ScaleMode) => void;
  setLossOutlierMode: (value: OutlierMode) => void;
  setGradOutlierMode: (value: OutlierMode) => void;
  setLrOutlierMode: (value: OutlierMode) => void;
  resetPreferences: () => void;
};

const defaultPreferences = {
  windowSize: null as number | null, // null = show full training history
  smoothing: 0.6,
  showRaw: true,
  showSmoothed: true,
  showAvgLine: true,
  lossScale: "linear" as ScaleMode,
  lrScale: "linear" as ScaleMode,
  gradScale: "linear" as ScaleMode,
  lossOutlierMode: "none" as OutlierMode,
  gradOutlierMode: "none" as OutlierMode,
  lrOutlierMode: "none" as OutlierMode,
};

export const useChartPreferencesStore = create<ChartPreferencesState>(
  (set) => ({
    availableSteps: 0,
    ...defaultPreferences,
    setAvailableSteps: (value) =>
      set((state) => {
        const availableSteps = Math.max(0, Math.round(value));
        if (state.windowSize == null || availableSteps <= 0) {
          return { availableSteps };
        }

        if (state.windowSize >= availableSteps) {
          return { availableSteps, windowSize: null };
        }

        return {
          availableSteps,
          windowSize: clamp(Math.round(state.windowSize), 1, availableSteps),
        };
      }),
    setWindowSize: (value) =>
      set((state) => {
        if (value == null || state.availableSteps <= 0) {
          return { windowSize: null };
        }

        const next = clamp(Math.round(value), 1, state.availableSteps);
        return { windowSize: next >= state.availableSteps ? null : next };
      }),
    setSmoothing: (value) => set({ smoothing: clamp(value, 0, 0.9) }),
    setShowRaw: (value) => set({ showRaw: value }),
    setShowSmoothed: (value) => set({ showSmoothed: value }),
    setShowAvgLine: (value) => set({ showAvgLine: value }),
    setLossScale: (value) => set({ lossScale: value }),
    setLrScale: (value) => set({ lrScale: value }),
    setGradScale: (value) => set({ gradScale: value }),
    setLossOutlierMode: (value) => set({ lossOutlierMode: value }),
    setGradOutlierMode: (value) => set({ gradOutlierMode: value }),
    setLrOutlierMode: (value) => set({ lrOutlierMode: value }),
    resetPreferences: () => set({ ...defaultPreferences }),
  }),
);
