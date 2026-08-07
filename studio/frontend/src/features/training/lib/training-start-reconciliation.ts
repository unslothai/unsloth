


import type { TrainingStartRequestStatusResponse } from "../types/api";
import type { TrainingStatusResponse } from "../types/runtime";

export type TrainingStartRequestOutcome =
  | { kind: "pending"; jobId: string; message: string }
  | { kind: "accepted"; jobId: string; message: string }
  | { kind: "rejected"; error: string; errorCode: string | null }
  | { kind: "unmatched" };

export function resolveTrainingStartRequestOutcome(
  status: TrainingStartRequestStatusResponse,
  expectedStartRequestId: string,
): TrainingStartRequestOutcome {
  if (status.start_request_id !== expectedStartRequestId) {
    return { kind: "unmatched" };
  }
  if (status.state === "rejected") {
    return {
      kind: "rejected",
      error: status.error || status.message,
      errorCode: status.error_code ?? null,
    };
  }
  if (!status.job_id.trim()) {
    return { kind: "unmatched" };
  }
  return {
    kind: status.state,
    jobId: status.job_id,
    message: status.message,
  };
}

export function statusConfirmsActiveTrainingStart(
  status: Pick<
    TrainingStatusResponse,
    "job_id" | "is_training_running" | "start_request_id"
  >,
  expectedStartRequestId: string,
): boolean {
  return (
    status.is_training_running &&
    status.job_id.trim().length > 0 &&
    status.start_request_id === expectedStartRequestId
  );
}
