// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { hasAuthToken } from "@/features/auth";
import {
  notifyNative,
  safeNotificationLabel,
  sanitizeNotificationBody,
} from "@/lib/native-notifications";
import { useEffect } from "react";
import {
  getTrainingMetrics,
  getTrainingStatus,
  isAbortError,
  streamTrainingProgress,
} from "../api/train-api";
import { useTrainingConfigStore } from "../stores/training-config-store";
import { useTrainingRuntimeStore } from "../stores/training-runtime-store";
import type { TrainingRuntimeStore } from "../types/runtime";

const STATUS_POLL_INTERVAL_MS = 3000;
const METRICS_POLL_INTERVAL_MS = 5000;
const IDLE_POLL_INTERVAL_MS = 30000;
const STREAM_RECONNECT_DELAY_MS = 1500;
const GENERIC_TRAINING_STREAM_ERROR = "Training stream error";
const DEFAULT_TRAINING_ERROR_BODY = "Your training run stopped with an error.";

function shouldUseLiveSync(state: TrainingRuntimeStore): boolean {
  return state.isTrainingRunning || state.phase === "training";
}

function getTrainingErrorBody(state: TrainingRuntimeStore): string {
  return sanitizeNotificationBody(
    state.error ?? state.message,
    DEFAULT_TRAINING_ERROR_BODY,
  );
}

function shouldNotifyTrainingError(
  before: TrainingRuntimeStore,
  after: TrainingRuntimeStore,
): boolean {
  if (after.phase !== "error") {
    return false;
  }
  if (before.phase !== "error" || before.jobId !== after.jobId) {
    return true;
  }
  if (before.error !== GENERIC_TRAINING_STREAM_ERROR) {
    return false;
  }

  return getTrainingErrorBody(before) !== getTrainingErrorBody(after);
}

function maybeNotifyTrainingTerminalTransition(
  before: TrainingRuntimeStore,
  after: TrainingRuntimeStore,
): void {
  if (!after.jobId) {
    return;
  }
  if (
    before.stopRequested ||
    after.stopRequested ||
    after.phase === "stopped"
  ) {
    return;
  }

  const sameJob = before.jobId === after.jobId;
  const samePhase = before.phase === after.phase;

  if (after.phase === "completed" && !(sameJob && samePhase)) {
    const model = safeNotificationLabel(
      after.startModelName ?? useTrainingConfigStore.getState().selectedModel,
      "Your training run",
    );
    notifyNative({
      key: `training-completed:${after.jobId}`,
      title: "Training finished",
      body: `${model} is complete.`,
      requestPermission: false,
    }).catch(() => undefined);
  }

  if (shouldNotifyTrainingError(before, after)) {
    notifyNative({
      key: `training-error:${after.jobId}`,
      title: "Training failed",
      body: getTrainingErrorBody(after),
      requestPermission: false,
    }).catch(() => undefined);
  }
}

export function useTrainingRuntimeLifecycle(): void {
  useEffect(() => {
    let disposed = false;
    let openingStream = false;
    let streamController: AbortController | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    const runtimeStore = useTrainingRuntimeStore;

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
          runtimeStore.getState().setSseConnected(false);
        }
      }
    };

    const pollStatus = async (options?: {
      suppressNativeNotifications?: boolean;
    }) => {
      if (!hasAuthToken()) return;
      const gen = runtimeStore.getState().resetGeneration;
      try {
        const status = await getTrainingStatus();
        if (disposed || runtimeStore.getState().resetGeneration !== gen) {
          return;
        }

        const previousState = runtimeStore.getState();
        runtimeStore.getState().applyStatus(status);

        const nextState = runtimeStore.getState();
        if (!options?.suppressNativeNotifications) {
          maybeNotifyTrainingTerminalTransition(previousState, nextState);
        }
        if (shouldUseLiveSync(nextState)) {
          void ensureStream();
        } else {
          stopStream();
        }
      } catch (error) {
        if (!isAbortError(error) && !disposed && hasAuthToken()) {
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
              liveStore.setRuntimeError(GENERIC_TRAINING_STREAM_ERROR);
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
        await Promise.all([
          pollStatus({ suppressNativeNotifications: true }),
          pollMetrics(),
        ]);
      } finally {
        if (!disposed) {
          runtimeStore.getState().setHydrating(false);
          runtimeStore.getState().setHasHydrated(true);
        }
      }
    };

    void hydrate();

    const isIdle = () => {
      const s = runtimeStore.getState();
      return s.phase === "idle" && !s.isTrainingRunning;
    };

    const statusTimer = setInterval(() => {
      const s = runtimeStore.getState();
      if (!s.hasHydrated || s.isHydrating) {
        return;
      }
      if (isIdle()) {
        return;
      }
      void pollStatus();
    }, STATUS_POLL_INTERVAL_MS);

    const metricsTimer = setInterval(() => {
      if (isIdle()) return;
      const s = runtimeStore.getState();
      if (shouldUseLiveSync(s) || s.currentStep > 0) {
        void pollMetrics();
      }
    }, METRICS_POLL_INTERVAL_MS);

    // Low-frequency background poll to recover from failed hydration or detect
    // out-of-band state changes (e.g. training started from another client).
    const idleTimer = setInterval(() => {
      const s = runtimeStore.getState();
      if (!s.hasHydrated || s.isHydrating) {
        return;
      }
      if (!isIdle()) {
        return;
      }
      void pollStatus();
    }, IDLE_POLL_INTERVAL_MS);

    return () => {
      disposed = true;
      clearInterval(statusTimer);
      clearInterval(metricsTimer);
      clearInterval(idleTimer);
      stopStream();
    };
  }, []);
}
