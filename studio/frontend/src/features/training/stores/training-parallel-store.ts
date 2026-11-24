// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";
import { parallelRequestFields } from "../lib/parallel-plan";
import type {
  ParallelCapabilities,
  ParallelModeChoice,
  ParallelOverride,
  ParallelPlan,
  ParallelRequestFields,
} from "../types/parallel";

export type ParallelFetchStatus =
  | "idle"
  | "loading"
  | "ready"
  | "unsupported"
  | "error";

interface TrainingParallelState {
  override: ParallelOverride;
  capabilities: ParallelCapabilities | null;
  capabilitiesStatus: ParallelFetchStatus;
  planStatus: ParallelFetchStatus;
  plan: ParallelPlan | null;
  planError: string | null;
}

interface TrainingParallelActions {
  setOverrideEnabled: (enabled: boolean) => void;
  setOverrideMode: (mode: ParallelModeChoice) => void;
  setOverrideGpuIds: (gpuIds: number[] | null) => void;
  setOverrideSpec: (spec: string) => void;
  setCapabilities: (
    status: ParallelFetchStatus,
    capabilities: ParallelCapabilities | null,
  ) => void;
  setPlanLoading: () => void;
  setPlan: (plan: ParallelPlan) => void;
  setPlanUnavailable: (
    status: "unsupported" | "error",
    message?: string,
  ) => void;
  clearPlan: () => void;
}

export const DEFAULT_PARALLEL_OVERRIDE: ParallelOverride = {
  enabled: false,
  mode: "auto",
  gpuIds: null,
  spec: "",
};

export const useTrainingParallelStore = create<
  TrainingParallelState & TrainingParallelActions
>()(
  persist(
    (set) => ({
      override: DEFAULT_PARALLEL_OVERRIDE,
      capabilities: null,
      capabilitiesStatus: "idle",
      planStatus: "idle",
      plan: null,
      planError: null,
      setOverrideEnabled: (enabled) =>
        set((s) => ({ override: { ...s.override, enabled } })),
      setOverrideMode: (mode) =>
        set((s) => ({ override: { ...s.override, mode } })),
      setOverrideGpuIds: (gpuIds) =>
        set((s) => ({ override: { ...s.override, gpuIds } })),
      setOverrideSpec: (spec) =>
        set((s) => ({ override: { ...s.override, spec } })),
      setCapabilities: (capabilitiesStatus, capabilities) =>
        set({ capabilitiesStatus, capabilities }),
      // Keeps the last plan on screen while the next one loads, so the line does not flicker.
      setPlanLoading: () => set({ planStatus: "loading", planError: null }),
      setPlan: (plan) => set({ planStatus: "ready", plan, planError: null }),
      setPlanUnavailable: (planStatus, message) =>
        set({ planStatus, plan: null, planError: message ?? null }),
      clearPlan: () => set({ planStatus: "idle", plan: null, planError: null }),
    }),
    {
      name: "unsloth_training_parallel_override",
      // Only the user's choice persists; capabilities and the plan are re-read every visit.
      partialize: (s) => ({ override: s.override }),
      merge: (persisted, current) => ({
        ...current,
        override: {
          ...DEFAULT_PARALLEL_OVERRIDE,
          ...((
            persisted as { override?: Partial<ParallelOverride> } | undefined
          )?.override ?? {}),
        },
      }),
      version: 1,
    },
  ),
);

/** What /start and the API example add for the GPUs section; {} on a 0 or 1 GPU host. */
export function currentParallelRequestFields(): ParallelRequestFields {
  const { override, capabilities } = useTrainingParallelStore.getState();
  return parallelRequestFields(override, capabilities);
}
