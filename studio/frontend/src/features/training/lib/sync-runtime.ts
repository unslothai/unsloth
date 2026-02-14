import {
  getTrainingMetrics,
  getTrainingStatus,
} from "../api/train-api";
import { useTrainingRuntimeStore } from "../stores/training-runtime-store";
import type { TrainingStatusResponse } from "../types/runtime";

export async function syncTrainingRuntimeFromBackend(): Promise<TrainingStatusResponse> {
  const [status, metrics] = await Promise.all([
    getTrainingStatus(),
    getTrainingMetrics(),
  ]);

  const runtimeStore = useTrainingRuntimeStore.getState();
  runtimeStore.applyStatus(status);
  runtimeStore.applyMetrics(metrics);

  return status;
}
