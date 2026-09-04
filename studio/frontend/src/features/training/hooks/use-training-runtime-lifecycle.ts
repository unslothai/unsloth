// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { hasAuthToken } from "@/features/auth";
import { useHfTokenStore } from "@/features/hub";
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
import {
  isTrainingStatusRequestCurrent,
  trainingStatusRequestKey,
} from "../lib/training-status-request";
import {
  createTrainingStreamScope,
  isTrainingProgressForScope,
  isTrainingStreamScopeCurrent,
} from "../lib/training-stream-scope";
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
    const clearStartError = () => {
      const runtime = useTrainingRuntimeStore.getState();
      if (runtime.startError !== null) {
        runtime.setStartError(null);
      }
    };
    const unsubscribeConfig = useTrainingConfigStore.subscribe(
      (state, previousState) => {
        if (state.userEditRevision !== previousState.userEditRevision) {
          clearStartError();
        }
      },
    );
    const unsubscribeToken = useHfTokenStore.subscribe(
      (state, previousState) => {
        if (state.token !== previousState.token) {
          clearStartError();
        }
      },
    );
    return () => {
      unsubscribeConfig();
      unsubscribeToken();
    };
  }, []);

  useEffect(() => {
    let disposed = false;
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
      const controller = streamController;
      streamController = null;
      controller?.abort();
      runtimeStore.getState().setSseConnected(false);
    };

    const pollMetrics = async (options?: { fresh?: boolean }) => {
      if (!hasAuthToken()) return;
      const initial = runtimeStore.getState();
      if (!initial.jobId) return;
      const requestKey = trainingStatusRequestKey(initial);
      try {
        const metrics = await getTrainingMetrics(initial.jobId, requestKey, {
          fresh: options?.fresh,
        });
        const current = runtimeStore.getState();
        if (disposed || !isTrainingStatusRequestCurrent(requestKey, current)) {
          return;
        }
        current.applyMetrics(metrics);
      } catch (error) {
        if (!isAbortError(error) && !disposed && hasAuthToken()) {
          runtimeStore.getState().setSseConnected(false);
        }
      }
    };

    const pollStatus = async (options?: {
      fresh?: boolean;
      suppressNativeNotifications?: boolean;
    }) => {
      if (!hasAuthToken()) return;
      const initial = runtimeStore.getState();
      const requestKey = trainingStatusRequestKey(initial);
      try {
        const status = await getTrainingStatus(requestKey, {
          fresh: options?.fresh,
        });
        const current = runtimeStore.getState();
        if (disposed || !isTrainingStatusRequestCurrent(requestKey, current)) {
          return;
        }

        const previousState = initial;
        current.applyStatus(status);

        const nextState = runtimeStore.getState();
        if (!options?.suppressNativeNotifications) {
          maybeNotifyTrainingTerminalTransition(previousState, nextState);
        }
        const acceptedLocalStart =
          nextState.isStarting &&
          status.start_request_state === "accepted" &&
          status.start_request_id === nextState.startRequestId;
        if (
          shouldUseLiveSync(nextState) &&
          status.start_request_state !== "pending" &&
          (!nextState.isStarting || acceptedLocalStart)
        ) {
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
      const scope = createTrainingStreamScope(state);
      if (disposed || streamController || !scope || !shouldUseLiveSync(state)) {
        return;
      }

      clearReconnect();
      const controller = new AbortController();
      streamController = controller;
      const isCurrentStream = () =>
        streamController === controller &&
        !controller.signal.aborted &&
        isTrainingStreamScopeCurrent(scope, runtimeStore.getState());

      try {
        await streamTrainingProgress({
          signal: controller.signal,
          expectedJobId: scope.jobId,
          lastEventId: state.lastEventId,
          onOpen: () => {
            if (!isCurrentStream()) {
              controller.abort();
              return;
            }
            runtimeStore.getState().setSseConnected(true);
          },
          onEvent: (event) => {
            if (!isCurrentStream()) {
              controller.abort();
              return;
            }
            if (!isTrainingProgressForScope(scope, event.payload)) {
              stopStream();
              return;
            }
            const liveStore = runtimeStore.getState();
            liveStore.applyProgress(event.payload, event.id ?? undefined);

            if (event.event === "complete") {
              stopStream();
              void pollStatus({ fresh: true });
              void pollMetrics({ fresh: true });
              return;
            }

            if (event.event === "error") {
              liveStore.setRuntimeError(GENERIC_TRAINING_STREAM_ERROR);
              stopStream();
            }
          },
        });
      } catch (error) {
        if (
          isCurrentStream() &&
          !disposed &&
          !controller.signal.aborted &&
          !isAbortError(error)
        ) {
          runtimeStore.getState().setSseConnected(false);
        }
      } finally {
        if (streamController === controller) {
          streamController = null;
          runtimeStore.getState().setSseConnected(false);

          if (!disposed && !controller.signal.aborted) {
            const liveState = runtimeStore.getState();
            if (
              isTrainingStreamScopeCurrent(scope, liveState) &&
              shouldUseLiveSync(liveState)
            ) {
              reconnectTimer = setTimeout(() => {
                reconnectTimer = null;
                if (
                  !disposed &&
                  isTrainingStreamScopeCurrent(scope, runtimeStore.getState())
                ) {
                  void ensureStream();
                }
              }, STREAM_RECONNECT_DELAY_MS);
            }
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

    const unsubscribeRuntime = runtimeStore.subscribe(
      (state, previousState) => {
        if (state.jobId !== previousState.jobId) {
          stopStream();
        }
      },
    );

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

    // Low-frequency poll: recovers failed hydration and catches out-of-band state changes.
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
      unsubscribeRuntime();
      stopStream();
    };
  }, []);
}
