import { useCallback } from "react";
import { checkDatasetFormat } from "../api/datasets-api";
import { buildTrainingStartPayload } from "../api/mappers";
import { startTraining, stopTraining, resetTraining } from "../api/train-api";
import { syncTrainingRuntimeFromBackend } from "../lib/sync-runtime";
import { validateTrainingConfig } from "../lib/validation";
import { useDatasetPreviewDialogStore } from "../stores/dataset-preview-dialog-store";
import { useTrainingConfigStore } from "../stores/training-config-store";
import { useTrainingRuntimeStore } from "../stores/training-runtime-store";
import type { TrainingConfigState } from "../types/config";

export function useTrainingActions() {
  const isStarting = useTrainingRuntimeStore((state) => state.isStarting);
  const startError = useTrainingRuntimeStore((state) => state.startError);

  const startTrainingRun = useCallback(async (): Promise<boolean> => {
    const config = useTrainingConfigStore.getState();
    const runtimeStore = useTrainingRuntimeStore.getState();
    const dialogStore = useDatasetPreviewDialogStore.getState();

    runtimeStore.setStartError(null);
    const validation = validateTrainingConfig(config);
    if (!validation.ok) {
      runtimeStore.setStartError(validation.message);
      return false;
    }

    runtimeStore.setStarting(true);

    try {
      const datasetName = getDatasetName(config);
      const isVlm = config.isVisionModel && config.isDatasetMultimodal === true;

      if (datasetName) {
        const check = await checkDatasetFormat({
          datasetName,
          hfToken: config.hfToken.trim() || null,
          subset: config.datasetSubset,
          split: config.datasetSplit,
          isVlm,
        });

        if (check.requires_manual_mapping && !hasManualMapping(config)) {
          const hintInput = isVlm
            ? check.detected_image_column
            : pickRoleColumn(check.suggested_mapping, "user");
          const hintOutput = isVlm
            ? check.detected_text_column
            : pickRoleColumn(check.suggested_mapping, "assistant");

          if (hintInput || hintOutput) {
            useTrainingConfigStore.getState().setDatasetManualMapping({
              input: hintInput ?? null,
              output: hintOutput ?? null,
            });
          }

          runtimeStore.setStarting(false);
          dialogStore.openMapping(check);
          return false;
        }
      }

      const payload = buildTrainingStartPayload(config);
      const response = await startTraining(payload);

      if (response.status === "error") {
        runtimeStore.setStartError(response.error || response.message);
        runtimeStore.setStarting(false);
        return false;
      }

      runtimeStore.setStartQueued(response.job_id, response.message);
      await syncTrainingRuntimeFromBackend();
      return true;
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Failed to start training";
      runtimeStore.setStartError(message);
      runtimeStore.setStarting(false);
      return false;
    }
  }, []);

  const stopTrainingRun = useCallback(async (save = true): Promise<boolean> => {
    const runtimeStore = useTrainingRuntimeStore.getState();
    runtimeStore.setStartError(null);

    try {
      await stopTraining(save);
      await syncTrainingRuntimeFromBackend();
      return true;
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Failed to stop training";
      runtimeStore.setRuntimeError(message);
      return false;
    }
  }, []);

  const dismissTrainingRun = useCallback(async (): Promise<void> => {
    useTrainingRuntimeStore.getState().resetRuntime();
    try {
      await resetTraining();
    } catch {
      // Frontend already reset; backend will catch up on next poll
    }
  }, []);

  return {
    isStarting,
    startError,
    startTrainingRun,
    stopTrainingRun,
    dismissTrainingRun,
  };
}

function getDatasetName(config: TrainingConfigState): string | null {
  return config.datasetSource === "huggingface"
    ? config.dataset
    : config.uploadedFile;
}

function hasManualMapping(config: TrainingConfigState): boolean {
  return (
    !!config.datasetManualMapping.input && !!config.datasetManualMapping.output
  );
}

function pickRoleColumn(
  mapping: Record<string, string> | null | undefined,
  role: string,
): string | null {
  if (!mapping) return null;
  for (const [col, mappedRole] of Object.entries(mapping)) {
    if (mappedRole === role) return col;
  }
  return null;
}
