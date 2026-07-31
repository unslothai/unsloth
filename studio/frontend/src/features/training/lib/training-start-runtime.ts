// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type TranslationKey, translate } from "@/i18n";
import {
  getTrainingStatus,
  resetTraining,
  stopTraining,
} from "../api/train-api";
import { emitTrainingRunsChanged } from "../events";
import { useTrainingRuntimeStore } from "../stores/training-runtime-store";
import { syncTrainingRuntimeFromBackend } from "./sync-runtime";
import { statusConfirmsActiveTrainingStart } from "./training-start-reconciliation";

export const TRAINING_SETUP_CHANGED_ERROR =
  "studio.training.setupChanged" satisfies TranslationKey;

export interface TrainingStartLease {
  resetGeneration: number;
}

export function tryAcquireTrainingStart(): TrainingStartLease | null {
  const runtime = useTrainingRuntimeStore.getState();
  if (!runtime.tryBeginStarting()) {
    return null;
  }
  return {
    resetGeneration: useTrainingRuntimeStore.getState().resetGeneration,
  };
}

export function isTrainingStartLeaseActive(lease: TrainingStartLease): boolean {
  const runtime = useTrainingRuntimeStore.getState();
  return (
    runtime.resetGeneration === lease.resetGeneration &&
    runtime.isStarting &&
    !runtime.stopRequested
  );
}

export function releaseTrainingStart(
  lease: TrainingStartLease,
  error?: string | null,
): false {
  if (!isTrainingStartLeaseActive(lease)) {
    return false;
  }
  const runtime = useTrainingRuntimeStore.getState();
  if (error !== undefined) {
    runtime.setStartError(
      error === TRAINING_SETUP_CHANGED_ERROR
        ? translate(TRAINING_SETUP_CHANGED_ERROR)
        : error,
    );
  }
  runtime.setStarting(false);
  return false;
}

async function resetSupersededBackendJob(jobId: string): Promise<void> {
  await stopTraining(false, { expectedJobId: jobId });
  await resetTraining({ expectedJobId: jobId });
}

async function cancelSupersededTrainingStart(jobId: string): Promise<void> {
  await resetSupersededBackendJob(jobId).catch(() => undefined);
  await syncTrainingRuntimeFromBackend().catch(() => undefined);
  emitTrainingRunsChanged();
}

export async function settleAcceptedTrainingStart(
  lease: TrainingStartLease,
  jobId: string,
  message: string,
): Promise<boolean> {
  if (!isTrainingStartLeaseActive(lease)) {
    await cancelSupersededTrainingStart(jobId);
    return false;
  }
  useTrainingRuntimeStore.getState().setStartQueued(jobId, message);
  await Promise.allSettled([
    Promise.resolve().then(emitTrainingRunsChanged),
    syncTrainingRuntimeFromBackend(),
  ]);
  return true;
}

export async function reconcileTrainingStartTransportFailure(
  lease: TrainingStartLease,
): Promise<boolean> {
  if (!isTrainingStartLeaseActive(lease)) {
    return false;
  }
  const status = await getTrainingStatus().catch(() => null);
  if (!status || !statusConfirmsActiveTrainingStart(status)) {
    return false;
  }
  if (!isTrainingStartLeaseActive(lease)) {
    return false;
  }

  const runtime = useTrainingRuntimeStore.getState();
  runtime.setStartQueued(status.job_id, status.message);
  await Promise.allSettled([
    Promise.resolve().then(emitTrainingRunsChanged),
    syncTrainingRuntimeFromBackend().catch(() => {
      useTrainingRuntimeStore.getState().applyStatus(status);
    }),
  ]);
  return true;
}
