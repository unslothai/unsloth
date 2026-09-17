// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  TrainingProgressPayload,
  TrainingRuntimeState,
} from "../types/runtime";

export interface TrainingStreamScope {
  jobId: string;
}

export function createTrainingStreamScope(
  state: Pick<TrainingRuntimeState, "jobId">,
): TrainingStreamScope | null {
  return state.jobId ? { jobId: state.jobId } : null;
}

export function isTrainingStreamScopeCurrent(
  scope: TrainingStreamScope,
  state: Pick<TrainingRuntimeState, "jobId">,
): boolean {
  return state.jobId === scope.jobId;
}

export function isTrainingProgressForJob(
  jobId: string | null,
  progressJobId: string,
): boolean {
  return jobId !== null && jobId.length > 0 && progressJobId === jobId;
}

export function isTrainingProgressForScope(
  scope: TrainingStreamScope,
  payload: Pick<TrainingProgressPayload, "job_id">,
): boolean {
  return isTrainingProgressForJob(scope.jobId, payload.job_id);
}
