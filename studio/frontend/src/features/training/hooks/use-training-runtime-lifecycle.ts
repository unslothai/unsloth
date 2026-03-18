// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { hasAuthToken } from "@/features/auth";
import { useEffect } from "react";
import {
  getTrainingMetrics,
  getTrainingStatus,
  isAbortError,
  streamTrainingProgress,
} from "../api/train-api";
import { useTrainingRuntimeStore } from "../stores/training-runtime-store";
import type { TrainingRuntimeStore } from "../types/runtime";

const STATUS_POLL_INTERVAL_MS = 3000;
const METRICS_POLL_INTERVAL_MS = 5000;
const STREAM_RECONNECT_DELAY_MS = 1500;
const AUTH_STATUS_RETRY_INTERVAL_MS = 3000;
const AUTH_STATUS_TIMEOUT_MS = 3000;
const INITIAL_HYDRATE_TIMEOUT_MS = 4000;

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function shouldUseLiveSync(state: TrainingRuntimeStore): boolean {
  return state.isTrainingRunning || state.phase === "training";
}

export function useTrainingRuntimeLifecycle(): void {
  useEffect(() => {
    let disposed = false;
    let openingStream = false;
    let streamController: AbortController | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    /** HF Spaces / auth-disabled: no JWT, but train APIs allow anonymous access. */
    let authDisabled = false;

    const runtimeStore = useTrainingRuntimeStore;

    const canUseTrainApi = () => hasAuthToken() || authDisabled;
    let authProbeInFlight = false;
    let lastAuthProbeStartedAt = 0;

    const maybeRefreshAuthMode = async (force = false) => {
      if (disposed || hasAuthToken() || authDisabled) return;
      if (authProbeInFlight) return;

      const now = Date.now();
      if (
        !force &&
        now - lastAuthProbeStartedAt < AUTH_STATUS_RETRY_INTERVAL_MS
      ) {
        return;
      }

      authProbeInFlight = true;
      lastAuthProbeStartedAt = now;
      const controller = new AbortController();
      const timeout = setTimeout(() => {
        controller.abort();
      }, AUTH_STATUS_TIMEOUT_MS);
      try {
        const res = await fetch("/api/auth/status", { signal: controller.signal });
        if (!res.ok) return;
        const data = (await res.json()) as { auth_disabled?: boolean };
        authDisabled = Boolean(data.auth_disabled);
      } catch {
        // Keep previous mode and retry later.
      } finally {
        clearTimeout(timeout);
        authProbeInFlight = false;
      }
    };

    const clearReconnect = () => {
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
    };

    const stopStream = () => {
      clearReconnect();
      if (streamController) {
        streamController.abort();
        streamController = null;
      }
      runtimeStore.getState().setSseConnected(false);
    };

    const pollMetrics = async () => {
      if (!canUseTrainApi()) {
        void maybeRefreshAuthMode();
        return;
      }
      const gen = runtimeStore.getState().resetGeneration;
      try {
        const metrics = await getTrainingMetrics();
        if (disposed || runtimeStore.getState().resetGeneration !== gen) {
          return;
        }
        runtimeStore.getState().applyMetrics(metrics);
      } catch (error) {
        if (!isAbortError(error) && !disposed && canUseTrainApi()) {
          runtimeStore.getState().setSseConnected(false);
        }
      }
    };

    const pollStatus = async () => {
      if (!canUseTrainApi()) {
        void maybeRefreshAuthMode();
        return;
      }
      const gen = runtimeStore.getState().resetGeneration;
      try {
        const status = await getTrainingStatus();
        if (disposed || runtimeStore.getState().resetGeneration !== gen) {
          return;
        }

        runtimeStore.getState().applyStatus(status);

        const nextState = runtimeStore.getState();
        if (shouldUseLiveSync(nextState)) {
          void ensureStream();
        } else {
          stopStream();
        }
      } catch (error) {
        if (!isAbortError(error) && !disposed && canUseTrainApi()) {
          runtimeStore.getState().setSseConnected(false);
        }
      }
    };

    const ensureStream = async () => {
      const state = runtimeStore.getState();
      if (
        disposed ||
        openingStream ||
        streamController ||
        !shouldUseLiveSync(state)
      ) {
        return;
      }

      clearReconnect();
      openingStream = true;
      const controller = new AbortController();
      streamController = controller;

      try {
        await streamTrainingProgress({
          signal: controller.signal,
          lastEventId: state.lastEventId,
          onOpen: () => {
            runtimeStore.getState().setSseConnected(true);
          },
          onEvent: (event) => {
            const liveStore = runtimeStore.getState();
            if (typeof event.id === "number") {
              liveStore.setLastEventId(event.id);
            }

            liveStore.applyProgress(event.payload, event.id ?? undefined);

            if (event.event === "complete") {
              void pollStatus();
              void pollMetrics();
              stopStream();
            }

            if (event.event === "error") {
              liveStore.setRuntimeError("Training stream error");
              stopStream();
            }
          },
        });
      } catch (error) {
        if (!disposed && !controller.signal.aborted && !isAbortError(error)) {
          runtimeStore.getState().setSseConnected(false);
        }
      } finally {
        openingStream = false;
        if (streamController === controller) {
          streamController = null;
        }
        runtimeStore.getState().setSseConnected(false);

        if (!disposed && !controller.signal.aborted) {
          const liveState = runtimeStore.getState();
          if (shouldUseLiveSync(liveState)) {
            reconnectTimer = setTimeout(() => {
              void ensureStream();
            }, STREAM_RECONNECT_DELAY_MS);
          }
        }
      }
    };

    const hydrate = async () => {
      runtimeStore.getState().setHydrating(true);
      try {
        await maybeRefreshAuthMode(true);
        await Promise.race([
          Promise.allSettled([pollStatus(), pollMetrics()]).then(() => undefined),
          wait(INITIAL_HYDRATE_TIMEOUT_MS),
        ]);
      } finally {
        if (!disposed) {
          runtimeStore.getState().setHydrating(false);
          runtimeStore.getState().setHasHydrated(true);
        }
      }
    };

    let statusTimer: ReturnType<typeof setInterval> | null = null;
    let metricsTimer: ReturnType<typeof setInterval> | null = null;

    void hydrate();
    statusTimer = setInterval(() => {
      void pollStatus();
    }, STATUS_POLL_INTERVAL_MS);
    metricsTimer = setInterval(() => {
      const state = runtimeStore.getState();
      if (shouldUseLiveSync(state) || state.currentStep > 0) {
        void pollMetrics();
      }
    }, METRICS_POLL_INTERVAL_MS);

    return () => {
      disposed = true;
      if (statusTimer) clearInterval(statusTimer);
      if (metricsTimer) clearInterval(metricsTimer);
      stopStream();
    };
  }, []);
}
