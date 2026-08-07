


import type { TrainingRuntimeState } from "../types/runtime";

export type TrainingStopScope =
  | { kind: "job"; jobId: string }
  | { kind: "start"; startRequestId: string };

export function trainingStopScope(
  runtime: Pick<TrainingRuntimeState, "jobId" | "startRequestId">,
): TrainingStopScope | null {
  const startRequestId = runtime.startRequestId?.trim();
  if (startRequestId) {
    return { kind: "start", startRequestId };
  }
  const jobId = runtime.jobId?.trim();
  if (jobId) {
    return { kind: "job", jobId };
  }
  return null;
}
