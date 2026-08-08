// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { useCallback } from "react";
import {
  cancelTrainingStartRequest,
  resetTraining,
  stopTraining,
} from "../api/train-api";
import { emitTrainingRunsChanged } from "../events";
import { resumeTrainingRun } from "../lib/resume-training-run";
import { startFreshTrainingRun } from "../lib/start-fresh-training-run";
import { syncTrainingRuntimeFromBackend } from "../lib/sync-runtime";
import {
  type TrainingStopScope,
  trainingStopScope,
} from "../lib/training-stop-scope";
import {
  isTrainingStartPending,
  useTrainingRuntimeStore,
} from "../stores/training-runtime-store";

export function useTrainingActions() {
  const t = useT();
  const startBlocked = useTrainingRuntimeStore(isTrainingStartPending);
  const stopRequested = useTrainingRuntimeStore((state) => state.stopRequested);
  const startError = useTrainingRuntimeStore((state) => state.startError);

  const startTrainingRun = useCallback(
    async (): Promise<boolean> => startFreshTrainingRun(),
    [],
  );

  const stopTrainingRun = useCallback(
    async (save = true): Promise<boolean> => {
      const runtimeStore = useTrainingRuntimeStore.getState();
      const scope = trainingStopScope(runtimeStore);
      const expectedResetGeneration = runtimeStore.resetGeneration;
      runtimeStore.setStartError(null);

      if (scope === null) {
        await syncTrainingRuntimeFromBackend().catch(() => undefined);
        return false;
      }

      try {
        if (scope.kind === "start") {
          const response = await cancelTrainingStartRequest(
            scope.startRequestId,
          );
          const currentRuntime = useTrainingRuntimeStore.getState();
          if (
            response.start_request_id !== scope.startRequestId ||
            response.state !== "rejected" ||
            currentRuntime.startRequestId !== scope.startRequestId ||
            currentRuntime.resetGeneration !== expectedResetGeneration
          ) {
            await syncTrainingRuntimeFromBackend().catch(() => undefined);
            return false;
          }
          currentRuntime.resetRuntime();
          emitTrainingRunsChanged();
          return true;
        }
        await stopTraining(save, { expectedJobId: scope.jobId });
      } catch (error) {
        const currentRuntime = useTrainingRuntimeStore.getState();
        if (
          !runtimeMatchesStopScope(currentRuntime, scope) ||
          currentRuntime.resetGeneration !== expectedResetGeneration
        ) {
          await syncTrainingRuntimeFromBackend().catch(() => undefined);
          return false;
        }
        const message =
          error instanceof Error
            ? error.message
            : t("studio.training.stopFailed");
        if (scope.kind === "start") {
          currentRuntime.setStopRequested(false);
          currentRuntime.setStartPending(null, message, scope.startRequestId);
          currentRuntime.setStartError(message);
        } else {
          currentRuntime.setRuntimeError(message);
        }
        await syncTrainingRuntimeFromBackend().catch(() => undefined);
        return false;
      }
      await syncTrainingRuntimeFromBackend().catch(() => undefined);
      const currentRuntime = useTrainingRuntimeStore.getState();
      return (
        scope.kind === "job" &&
        currentRuntime.jobId === scope.jobId &&
        currentRuntime.resetGeneration === expectedResetGeneration
      );
    },
    [t],
  );

  const resumeTrainingRunFromHistory = useCallback(
    async (runId: string): Promise<boolean> => resumeTrainingRun(runId),
    [],
  );

  const dismissTrainingRun = useCallback(async (): Promise<void> => {
    try {
      const runtimeStore = useTrainingRuntimeStore.getState();
      const scope = trainingStopScope(runtimeStore);
      const expectedResetGeneration = runtimeStore.resetGeneration;
      if (scope === null) {
        runtimeStore.resetRuntime();
        return;
      }
      if (scope.kind === "start") {
        const response = await cancelTrainingStartRequest(scope.startRequestId);
        const currentRuntime = useTrainingRuntimeStore.getState();
        if (
          response.start_request_id !== scope.startRequestId ||
          response.state !== "rejected" ||
          !runtimeMatchesStopScope(currentRuntime, scope) ||
          currentRuntime.resetGeneration !== expectedResetGeneration
        ) {
          await syncTrainingRuntimeFromBackend().catch(() => undefined);
          return;
        }
        currentRuntime.resetRuntime();
        return;
      }
      const response = await resetTraining({ expectedJobId: scope.jobId });
      const currentRuntime = useTrainingRuntimeStore.getState();
      if (
        response.status === "superseded" ||
        currentRuntime.jobId !== scope.jobId ||
        currentRuntime.resetGeneration !== expectedResetGeneration
      ) {
        await syncTrainingRuntimeFromBackend().catch(() => undefined);
        return;
      }
      currentRuntime.resetRuntime();
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : t("studio.training.stopBeforeConfig");
      toast.error(t("studio.training.trainingStillActiveTitle"), {
        description: message,
      });
      await syncTrainingRuntimeFromBackend().catch(() => undefined);
    }
  }, [t]);

  return {
    startBlocked,
    stopRequested,
    startError,
    startTrainingRun,
    resumeTrainingRunFromHistory,
    stopTrainingRun,
    dismissTrainingRun,
  };
}

function runtimeMatchesStopScope(
  runtime: ReturnType<typeof useTrainingRuntimeStore.getState>,
  scope: TrainingStopScope,
): boolean {
  return scope.kind === "job"
    ? runtime.jobId === scope.jobId
    : runtime.startRequestId === scope.startRequestId;
}
