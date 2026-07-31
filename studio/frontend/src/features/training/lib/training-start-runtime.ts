// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type TranslationKey, translate } from "@/i18n";
import {
  acknowledgeTrainingStartRequest,
  getTrainingStartRequestStatus,
  resetTraining,
  stopTraining,
} from "../api/train-api";
import { emitTrainingRunsChanged } from "../events";
import { useTrainingRuntimeStore } from "../stores/training-runtime-store";
import { syncTrainingRuntimeFromBackend } from "./sync-runtime";
import { resolveTrainingStartRequestOutcome } from "./training-start-reconciliation";

export const TRAINING_SETUP_CHANGED_ERROR =
  "studio.training.setupChanged" satisfies TranslationKey;

export interface TrainingStartLease {
  resetGeneration: number;
  startRequestId: string;
}

export type TrainingStartRecoveryResult =
  | { kind: "recovered" }
  | { kind: "rejected"; error: string }
  | { kind: "unknown" };

const START_RECONCILIATION_DELAYS_MS = [
  0, 250, 750, 1500, 2500, 5000, 5000, 5000, 5000, 5000,
] as const;
const START_REGISTRATION_ATTEMPTS = 5;

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function tryAcquireTrainingStart(): TrainingStartLease | null {
  const runtime = useTrainingRuntimeStore.getState();
  if (!runtime.tryBeginStarting()) {
    return null;
  }
  return {
    resetGeneration: useTrainingRuntimeStore.getState().resetGeneration,
    startRequestId: crypto.randomUUID(),
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
): Promise<TrainingStartRecoveryResult> {
  let pending: { jobId: string; message: string } | null = null;
  for (const [attempt, delayMs] of START_RECONCILIATION_DELAYS_MS.entries()) {
    if (attempt >= START_REGISTRATION_ATTEMPTS && pending === null) {
      break;
    }
    if (!isTrainingStartLeaseActive(lease)) {
      return { kind: "unknown" };
    }
    if (delayMs > 0) {
      await wait(delayMs);
    }
    if (!isTrainingStartLeaseActive(lease)) {
      return { kind: "unknown" };
    }

    const status = await getTrainingStartRequestStatus(
      lease.startRequestId,
    ).catch(() => null);
    if (!status) {
      continue;
    }
    const outcome = resolveTrainingStartRequestOutcome(
      status,
      lease.startRequestId,
    );
    if (outcome.kind === "rejected") {
      await acknowledgeTrainingStartRequest(lease.startRequestId).catch(
        () => undefined,
      );
      return outcome;
    }
    if (outcome.kind === "pending") {
      pending = outcome;
      continue;
    }
    if (outcome.kind === "unmatched") {
      continue;
    }
    if (!isTrainingStartLeaseActive(lease)) {
      return { kind: "unknown" };
    }

    useTrainingRuntimeStore
      .getState()
      .setStartQueued(outcome.jobId, outcome.message);
    await Promise.allSettled([
      Promise.resolve().then(emitTrainingRunsChanged),
      syncTrainingRuntimeFromBackend(),
    ]);
    return { kind: "recovered" };
  }
  if (pending && isTrainingStartLeaseActive(lease)) {
    useTrainingRuntimeStore
      .getState()
      .setStartQueued(pending.jobId, pending.message);
    await Promise.allSettled([
      Promise.resolve().then(emitTrainingRunsChanged),
      syncTrainingRuntimeFromBackend(),
    ]);
    return { kind: "recovered" };
  }
  return { kind: "unknown" };
}
