// SPDX-License-Identifier: AGPL-3.0-only - See /studio/LICENSE.AGPL-3.0
// Copyright © 2025 Unsloth AI

import {
  getTrainingMetrics,
  getTrainingStatus,
} from "../api/train-api";
import { useTrainingRuntimeStore } from "../stores/training-runtime-store";
import type { TrainingStatusResponse } from "../types/runtime";

export async function syncTrainingRuntimeFromBackend(): Promise<TrainingStatusResponse> {
  const gen = useTrainingRuntimeStore.getState().resetGeneration;

  const [status, metrics] = await Promise.all([
    getTrainingStatus(),
    getTrainingMetrics(),
  ]);

  const runtimeStore = useTrainingRuntimeStore.getState();
  if (runtimeStore.resetGeneration !== gen) {
    return status;
  }
  runtimeStore.applyStatus(status);
  runtimeStore.applyMetrics(metrics);

  return status;
}
