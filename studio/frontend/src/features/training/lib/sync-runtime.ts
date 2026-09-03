// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getTrainingMetrics, getTrainingStatus } from "../api/train-api";
import { useTrainingRuntimeStore } from "../stores/training-runtime-store";
import type { TrainingStatusResponse } from "../types/runtime";
import {
  isTrainingStatusRequestCurrent,
  trainingStatusRequestKey,
} from "./training-status-request";

export async function syncTrainingRuntimeFromBackend(): Promise<TrainingStatusResponse> {
  const initial = useTrainingRuntimeStore.getState();
  const requestKey = trainingStatusRequestKey(initial);
  const status = await getTrainingStatus(requestKey);

  let runtimeStore = useTrainingRuntimeStore.getState();
  if (!isTrainingStatusRequestCurrent(requestKey, runtimeStore)) {
    return status;
  }
  runtimeStore.applyStatus(status);

  runtimeStore = useTrainingRuntimeStore.getState();
  const jobId = status.job_id;
  if (!jobId || status.start_request_state === "pending") {
    return status;
  }
  const metricsRequestKey = trainingStatusRequestKey(runtimeStore);
  try {
    const metrics = await getTrainingMetrics(jobId, metricsRequestKey);
    runtimeStore = useTrainingRuntimeStore.getState();
    if (
      isTrainingStatusRequestCurrent(metricsRequestKey, runtimeStore) &&
      runtimeStore.jobId === jobId
    ) {
      runtimeStore.applyMetrics(metrics);
    }
  } catch {
    return status;
  }

  return status;
}
