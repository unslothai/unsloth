// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { TrainingRuntimeState } from "../types/runtime";

export function trainingStatusRequestKey(
  state: Pick<
    TrainingRuntimeState,
    "jobId" | "resetGeneration" | "startRequestId"
  >,
): string {
  return JSON.stringify([
    state.resetGeneration,
    state.jobId,
    state.startRequestId,
  ]);
}

export function isTrainingStatusRequestCurrent(
  requestKey: string,
  state: Pick<
    TrainingRuntimeState,
    "jobId" | "resetGeneration" | "startRequestId"
  >,
): boolean {
  return requestKey === trainingStatusRequestKey(state);
}
