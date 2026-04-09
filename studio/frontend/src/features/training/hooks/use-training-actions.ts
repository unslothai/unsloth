// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
import { toast } from "sonner";

/** Chatml → format-specific role remap (only for formats that differ from chatml). */
const ROLE_REMAP: Record<string, Record<string, string>> = {
  alpaca: { user: "instruction", system: "input", assistant: "output" },
  sharegpt: { user: "human", assistant: "gpt", system: "system" },
};

function normalizeTrainingStartError(message: string): string {
  const normalized = message.toLowerCase();
  const isLegacyDatasetScriptError =
    normalized.includes("failed to check dataset format") &&
    normalized.includes("dataset scripts are no longer supported");

  if (isLegacyDatasetScriptError) {
    return "This Hub dataset relies on a legacy custom script and isn’t supported in this training flow.";
  }

  return message;
}

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
      let isVlm = config.isVisionModel && config.isDatasetImage === true;

      if (datasetName) {
        const check = await checkDatasetFormat({
          datasetName,
          hfToken: config.hfToken.trim() || null,
          subset: config.datasetSubset,
          split: config.datasetSplit,
          isVlm,
        });

        // Backend auto-detects image/audio from dataset content.
        // Sync these flags into the store so buildTrainingStartPayload picks them up.
        const isAudio = !!check.is_audio;
        const isImage = !!check.is_image;

        if (isImage && config.isVisionModel) {
          isVlm = true;
        }
        if (isImage !== config.isDatasetImage || isAudio !== config.isDatasetAudio) {
          useTrainingConfigStore.setState({
            isDatasetImage: isImage,
            isDatasetAudio: isAudio,
          });
        }

        const needsReview = check.requires_manual_mapping || check.detected_format === "custom_heuristic";
        if (needsReview && !hasManualMapping(config, isVlm, isAudio)) {
          // Pre-fill from suggested_mapping or VLM detected columns
          const hint: Record<string, string> = {};
          if (check.suggested_mapping) {
            const table = ROLE_REMAP[config.datasetFormat];
            for (const [col, role] of Object.entries(check.suggested_mapping)) {
              hint[col] = table ? (table[role] ?? role) : role;
            }
          } else if (isAudio) {
            if (check.detected_audio_column) hint[check.detected_audio_column] = "audio";
            if (check.detected_text_column) hint[check.detected_text_column] = "text";
            if (check.detected_speaker_column) hint[check.detected_speaker_column] = "speaker_id";
          } else if (isVlm) {
            if (check.detected_image_column) hint[check.detected_image_column] = "image";
            if (check.detected_text_column) hint[check.detected_text_column] = "text";
          }

          if (Object.keys(hint).length > 0) {
            useTrainingConfigStore.getState().setDatasetManualMapping(hint);
          }

          runtimeStore.setStarting(false);
          dialogStore.openMapping(check);
          return false;
        }
      }

      // Abort if cancel was requested during dataset check
      if (useTrainingRuntimeStore.getState().stopRequested) {
        runtimeStore.setStarting(false);
        return false;
      }

      // Re-read config after potential store updates from dataset check
      const payload = buildTrainingStartPayload(useTrainingConfigStore.getState());
      const response = await startTraining(payload);

      if (response.status === "error") {
        const rawMessage = response.error || response.message;
        const safeMessage = normalizeTrainingStartError(rawMessage);
        runtimeStore.setStartError(safeMessage);
        runtimeStore.setStarting(false);
        return false;
      }

      runtimeStore.setStartQueued(response.job_id, response.message);
      await syncTrainingRuntimeFromBackend();
      return true;
    } catch (error) {
      const rawMessage =
        error instanceof Error ? error.message : "Failed to start training";
      const safeMessage = normalizeTrainingStartError(rawMessage);
      runtimeStore.setStartError(safeMessage);
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
    try {
      await resetTraining();
      useTrainingRuntimeStore.getState().resetRuntime();
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : "Stop training first, then return to configuration.";
      toast.error("Training still active", {
        description: message,
      });
      await syncTrainingRuntimeFromBackend();
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

function hasManualMapping(config: TrainingConfigState, isVlm = false, isAudio = false): boolean {
  const mapping = config.datasetManualMapping;
  const roles = new Set(Object.values(mapping));
  if (isAudio) return roles.has("audio") && roles.has("text");
  if (isVlm) return roles.has("image") && roles.has("text");
  const fmt = config.datasetFormat;
  if (fmt === "alpaca") return roles.has("instruction") && roles.has("output");
  if (fmt === "sharegpt") return roles.has("human") && roles.has("gpt");
  return roles.has("user") && roles.has("assistant");
}
