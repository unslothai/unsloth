// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import type { EvalDatasetValue } from "../components/eval-dataset-fields";

interface EvalConfigState {
  modelIdentifier: string;
  hfToken: string;
  dataset: EvalDatasetValue;
  metricName: string;
  metricConfig: Record<string, unknown>;
  runAll: boolean;
  limit: number;
  maxNewTokens: number;
  temperature: number;

  setModelIdentifier: (v: string) => void;
  setHfToken: (v: string) => void;
  setDataset: (v: EvalDatasetValue) => void;
  setMetricName: (v: string) => void;
  setMetricConfig: (v: Record<string, unknown>) => void;
  // Select a metric AND seed its config defaults in one atomic update (used when the user changes the metric).
  selectMetric: (name: string, config: Record<string, unknown>) => void;
  setRunAll: (v: boolean) => void;
  setLimit: (v: number) => void;
  setMaxNewTokens: (v: number) => void;
  setTemperature: (v: number) => void;
}

const DEFAULT_DATASET: EvalDatasetValue = {
  isLocal: false,
  name: "",
  path: "",
  split: "train",
  subset: "",
  inputColumn: "",
  referenceColumn: "",
};

export const useEvalConfigStore = create<EvalConfigState>()(
  persist(
    (set) => ({
      modelIdentifier: "",
      hfToken: "",
      dataset: DEFAULT_DATASET,
      metricName: "",
      metricConfig: {},
      runAll: false,
      limit: 100,
      maxNewTokens: 256,
      temperature: 0,

      setModelIdentifier: (v) => set({ modelIdentifier: v }),
      setHfToken: (v) => set({ hfToken: v }),
      setDataset: (v) => set({ dataset: v }),
      setMetricName: (v) => set({ metricName: v }),
      setMetricConfig: (v) => set({ metricConfig: v }),
      selectMetric: (name, config) => set({ metricName: name, metricConfig: config }),
      setRunAll: (v) => set({ runAll: v }),
      setLimit: (v) => set({ limit: v }),
      setMaxNewTokens: (v) => set({ maxNewTokens: v }),
      setTemperature: (v) => set({ temperature: v }),
    }),
    {
      name: "unsloth-eval-config",
      storage: createJSONStorage(() => localStorage),
    },
  ),
);
