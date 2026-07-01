// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { primeNativeNotificationPermission } from "@/lib/native-notifications";
import { confirmRemoteCodeIfNeeded } from "@/features/security";
import { useCallback } from "react";
import { toast } from "@/lib/toast";
import { checkDatasetFormat } from "../api/datasets-api";
import { emitTrainingRunsChanged } from "../events";
import { getTrainingRun } from "../api/history-api";
import { buildTrainingStartPayload } from "../api/mappers";
import { resetTraining, startTraining, stopTraining } from "../api/train-api";
import { isRawTextDatasetFormat } from "../lib/training-methods";
import { syncTrainingRuntimeFromBackend } from "../lib/sync-runtime";
import { validateTrainingConfig } from "../lib/validation";
import { useDatasetPreviewDialogStore } from "../stores/dataset-preview-dialog-store";
import { useTrainingConfigStore } from "../stores/training-config-store";
import { useTrainingRuntimeStore } from "../stores/training-runtime-store";
import type { TrainingStartRequest } from "../types/api";
import type { TrainingConfigState } from "../types/config";

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

    primeNativeNotificationPermission().catch(() => undefined);

    runtimeStore.setStartResources(
      config.selectedModel ?? null,
      getHfDatasetName(config),
      false,
      config.projectName || "",
    );
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

        // Backend auto-detects image/audio from dataset content; sync the flags
        // into the store so buildTrainingStartPayload picks them up.
        const isAudio = !!check.is_audio;
        const isImage = !!check.is_image;

        if (isImage && config.isVisionModel) {
          isVlm = true;
        }
        if (isImage !== config.isDatasetImage || isAudio !== config.isDatasetAudio) {
          useTrainingConfigStore.setState({
            isDatasetImage: isImage,
            isDatasetAudio: isAudio,
            // Streaming is unsupported for image/audio datasets; clear the flag
            // so buildTrainingStartPayload never ships dataset_streaming=true
            // for a modality the backend would reject with a 422.
            ...(isImage || isAudio ? { datasetStreaming: false } : {}),
          });
        }

        const isRawFormat = isRawTextDatasetFormat(config.datasetFormat);
        const needsReview =
          !isRawFormat &&
          (check.requires_manual_mapping || check.detected_format === "custom_heuristic");
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

      // Consent gate for the selected model's custom (auto_map) code.
      if (config.selectedModel) {
        const remoteCodeOk = await confirmRemoteCodeIfNeeded({
          modelName: config.selectedModel,
          hfToken: config.hfToken.trim() || null,
          requiresTrustRemoteCode: config.trustRemoteCode,
          onApprove: (fingerprint) =>
            useTrainingConfigStore.setState({
              trustRemoteCode: true,
              approvedRemoteCodeFingerprint: fingerprint,
            }),
        });
        if (!remoteCodeOk) {
          runtimeStore.setStarting(false);
          return false;
        }
      }

      // Re-read config after potential store updates from dataset check
      const payload = buildTrainingStartPayload(useTrainingConfigStore.getState());
      runtimeStore.setStartResources(
        payload.model_name,
        payload.hf_dataset,
        false,
        payload.project_name ?? "",
      );
      const response = await startTraining(payload);

      if (response.status === "error") {
        const rawMessage = response.error || response.message;
        const safeMessage = normalizeTrainingStartError(rawMessage);
        runtimeStore.setStartError(safeMessage);
        runtimeStore.setStarting(false);
        return false;
      }

      runtimeStore.setStartQueued(response.job_id, response.message);
      emitTrainingRunsChanged();
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

  const resumeTrainingRunFromHistory = useCallback(async (runId: string): Promise<boolean> => {
    const runtimeStore = useTrainingRuntimeStore.getState();
    runtimeStore.setStartError(null);
    runtimeStore.setStartResources(null, null, true, null);
    runtimeStore.setStarting(true);

    try {
      const detail = await getTrainingRun(runId);
      const outputDir = detail.run.output_dir;
      if (!detail.run.can_resume || !outputDir) {
        throw new Error("Only stopped runs with a saved checkpoint can be resumed.");
      }

      primeNativeNotificationPermission().catch(() => undefined);

      const config = useTrainingConfigStore.getState();
      const savedConfig = detail.config as Partial<TrainingStartRequest>;
      const payload = {
        ...savedConfig,
        hf_token:
          typeof savedConfig.hf_token === "string"
            ? savedConfig.hf_token
            : config.hfToken.trim() || null,
        wandb_token: null,
        resume_from_checkpoint: outputDir,
      } as TrainingStartRequest;

      runtimeStore.setStartResources(
        payload.model_name,
        payload.hf_dataset,
        true,
        payload.project_name ?? "",
      );

      // Resume goes straight to startTraining, so it runs the same consent gate as a
      // fresh start; otherwise a resumed custom-code run hits the worker block with no dialog.
      if (payload.model_name) {
        let trustRemoteCode = Boolean(payload.trust_remote_code);
        let approvedRemoteCodeFingerprint =
          payload.approved_remote_code_fingerprint ?? null;
        const remoteCodeOk = await confirmRemoteCodeIfNeeded({
          modelName: payload.model_name,
          hfToken: payload.hf_token ?? null,
          requiresTrustRemoteCode: trustRemoteCode,
          onApprove: (fingerprint) => {
            trustRemoteCode = true;
            approvedRemoteCodeFingerprint = fingerprint;
          },
        });
        if (!remoteCodeOk) {
          runtimeStore.setStarting(false);
          return false;
        }
        payload.trust_remote_code = trustRemoteCode;
        payload.approved_remote_code_fingerprint = approvedRemoteCodeFingerprint;
      }

      const response = await startTraining(payload);
      if (response.status === "error") {
        throw new Error(response.error || response.message);
      }

      runtimeStore.setStartQueued(response.job_id, response.message);
      emitTrainingRunsChanged();
      await syncTrainingRuntimeFromBackend();
      return true;
    } catch (error) {
      const rawMessage =
        error instanceof Error ? error.message : "Failed to resume training";
      const safeMessage = normalizeTrainingStartError(rawMessage);
      runtimeStore.setStartError(safeMessage);
      runtimeStore.setStarting(false);
      toast.error("Could not resume training", {
        description: safeMessage,
      });
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
    resumeTrainingRunFromHistory,
    stopTrainingRun,
    dismissTrainingRun,
  };
}

function getDatasetName(config: TrainingConfigState): string | null {
  return config.datasetSource === "huggingface"
    ? config.dataset
    : config.uploadedFile;
}

function getHfDatasetName(config: TrainingConfigState): string | null {
  return config.datasetSource === "huggingface" ? config.dataset : null;
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
