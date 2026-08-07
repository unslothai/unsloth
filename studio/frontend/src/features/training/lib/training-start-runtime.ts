


import { type TranslationKey, translate } from "@/i18n";
import {
  acknowledgeTrainingStartRequest,
  getTrainingStartRequestStatus,
  getTrainingStatus,
  resetTraining,
  stopTraining,
} from "../api/train-api";
import { emitTrainingRunsChanged } from "../events";
import { useTrainingRuntimeStore } from "../stores/training-runtime-store";
import { syncTrainingRuntimeFromBackend } from "./sync-runtime";
import {
  resolveTrainingStartRequestOutcome,
  statusConfirmsActiveTrainingStart,
} from "./training-start-reconciliation";
import { createTrainingStartRequestId } from "./training-start-request-id";
import {
  isTrainingStatusRequestCurrent,
  trainingStatusRequestKey,
} from "./training-status-request";

export const TRAINING_SETUP_CHANGED_ERROR =
  "studio.training.setupChanged" satisfies TranslationKey;

export interface TrainingStartLease {
  startRequestId: string;
}

export type TrainingStartRecoveryResult =
  | { kind: "recovered" }
  | { kind: "rejected"; error: string; errorCode: string | null }
  | { kind: "unknown" };

const START_RECONCILIATION_DELAYS_MS = [
  0, 250, 750, 1500, 2500, 5000, 5000, 5000, 5000, 5000,
] as const;
const START_REGISTRATION_ATTEMPTS = 5;

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function tryAcquireTrainingStart(): TrainingStartLease | null {
  const startRequestId = createTrainingStartRequestId();
  const runtime = useTrainingRuntimeStore.getState();
  if (!runtime.tryBeginStarting(startRequestId)) {
    return null;
  }
  return { startRequestId };
}

export function isTrainingStartLeaseActive(lease: TrainingStartLease): boolean {
  const runtime = useTrainingRuntimeStore.getState();
  return (
    runtime.startRequestId === lease.startRequestId &&
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

function adoptAcceptedTrainingStart(jobId: string, message: string): void {
  useTrainingRuntimeStore.getState().setStartPending(jobId, message);
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
  adoptAcceptedTrainingStart(jobId, message);
  await Promise.allSettled([
    Promise.resolve().then(emitTrainingRunsChanged),
    syncTrainingRuntimeFromBackend(),
  ]);
  return true;
}

export function settleUnconfirmedTrainingStart(
  lease: TrainingStartLease,
  message: string,
): boolean {
  if (!isTrainingStartLeaseActive(lease)) {
    return false;
  }
  useTrainingRuntimeStore
    .getState()
    .setStartPending(null, message, lease.startRequestId);
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

    adoptAcceptedTrainingStart(outcome.jobId, outcome.message);
    await Promise.allSettled([
      Promise.resolve().then(emitTrainingRunsChanged),
      syncTrainingRuntimeFromBackend(),
    ]);
    return { kind: "recovered" };
  }
  if (pending && isTrainingStartLeaseActive(lease)) {
    // The backend reserves the request id and job id before the heavy preflight, so a start still
    // pending when the reconciliation window closes may yet be rejected. Adopting pending.jobId here
    // reported success for an unconfirmed start and pinned a job id that never became current_job_id,
    // leaving the rejected state unclearable. Keep it unconfirmed so the outcome can be acknowledged.
    settleUnconfirmedTrainingStart(lease, pending.message);
    return { kind: "unknown" };
  }
  let statusRuntime = useTrainingRuntimeStore.getState();
  const requestKey = trainingStatusRequestKey(statusRuntime);
  const status = await getTrainingStatus(requestKey).catch(() => null);
  statusRuntime = useTrainingRuntimeStore.getState();
  if (
    status &&
    isTrainingStatusRequestCurrent(requestKey, statusRuntime) &&
    isTrainingStartLeaseActive(lease) &&
    statusConfirmsActiveTrainingStart(status, lease.startRequestId)
  ) {
    statusRuntime.applyStatus(status);
    useTrainingRuntimeStore.getState().setStarting(false);
    emitTrainingRunsChanged();
    return { kind: "recovered" };
  }
  return { kind: "unknown" };
}
