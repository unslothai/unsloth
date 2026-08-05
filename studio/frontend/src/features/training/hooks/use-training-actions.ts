// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import { useCallback } from "react";
import { resetTraining, stopTraining } from "../api/train-api";
import { resumeTrainingRun } from "../lib/resume-training-run";
import { startFreshTrainingRun } from "../lib/start-fresh-training-run";
import { syncTrainingRuntimeFromBackend } from "../lib/sync-runtime";
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
      const expectedJobId = runtimeStore.jobId;
      const expectedResetGeneration = runtimeStore.resetGeneration;
      runtimeStore.setStartError(null);

      try {
        await stopTraining(save, expectedJobId ? { expectedJobId } : undefined);
      } catch (error) {
        const currentRuntime = useTrainingRuntimeStore.getState();
        if (
          currentRuntime.jobId !== expectedJobId ||
          currentRuntime.resetGeneration !== expectedResetGeneration
        ) {
          await syncTrainingRuntimeFromBackend().catch(() => undefined);
          return false;
        }
        const message =
          error instanceof Error
            ? error.message
            : t("studio.training.stopFailed");
        currentRuntime.setRuntimeError(message);
        await syncTrainingRuntimeFromBackend().catch(() => undefined);
        return false;
      }
      await syncTrainingRuntimeFromBackend().catch(() => undefined);
      const currentRuntime = useTrainingRuntimeStore.getState();
      return (
        currentRuntime.jobId === expectedJobId &&
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
      const expectedJobId = runtimeStore.jobId;
      const expectedResetGeneration = runtimeStore.resetGeneration;
      const response = await resetTraining(
        expectedJobId ? { expectedJobId } : undefined,
      );
      const currentRuntime = useTrainingRuntimeStore.getState();
      if (
        response.status === "superseded" ||
        currentRuntime.jobId !== expectedJobId ||
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
