// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { hasAuthToken } from "@/features/auth";
import { useEffect } from "react";
import {
  getTrainingMetrics,
  getTrainingStatus,
  isAbortError,
} from "../api/train-api";
import { useTrainingRuntimeStore } from "../stores/training-runtime-store";

const STATUS_POLL_INTERVAL_MS = 2000;
const METRICS_POLL_INTERVAL_MS = 3000;

export function useTrainingRuntimeLifecycle(): void {
  useEffect(() => {
    let disposed = false;

    const runtimeStore = useTrainingRuntimeStore;

    const pollMetrics = async () => {
      if (!hasAuthToken()) return;
      const gen = runtimeStore.getState().resetGeneration;
      try {
        const metrics = await getTrainingMetrics();
        if (disposed || runtimeStore.getState().resetGeneration !== gen) {
          return;
        }
        runtimeStore.getState().applyMetrics(metrics);
      } catch (error) {
        if (!isAbortError(error) && !disposed && hasAuthToken()) {
          // silent — next poll will retry
        }
      }
    };

    const pollStatus = async () => {
      if (!hasAuthToken()) return;
      const gen = runtimeStore.getState().resetGeneration;
      try {
        const status = await getTrainingStatus();
        if (disposed || runtimeStore.getState().resetGeneration !== gen) {
          return;
        }
        runtimeStore.getState().applyStatus(status);
      } catch (error) {
        if (!isAbortError(error) && !disposed && hasAuthToken()) {
          // silent — next poll will retry
        }
      }
    };

    const hydrate = async () => {
      runtimeStore.getState().setHydrating(true);
      try {
        await Promise.all([pollStatus(), pollMetrics()]);
      } finally {
        if (!disposed) {
          runtimeStore.getState().setHydrating(false);
          runtimeStore.getState().setHasHydrated(true);
        }
      }
    };

    void hydrate();

    const statusTimer = setInterval(() => {
      void pollStatus();
    }, STATUS_POLL_INTERVAL_MS);

    const metricsTimer = setInterval(() => {
      const state = runtimeStore.getState();
      if (state.isTrainingRunning || state.phase === "training" || state.currentStep > 0) {
        void pollMetrics();
      }
    }, METRICS_POLL_INTERVAL_MS);

    return () => {
      disposed = true;
      clearInterval(statusTimer);
      clearInterval(metricsTimer);
    };
  }, []);
}
