// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { RecipeExecutionKind } from "../execution-types";
import type { RecipeRunSettings } from "../stores/recipe-executions";
import type { RecipePayload } from "../utils/payload/types";

function toPositiveInt(
  value: number,
  fallback: number,
  min = 1,
  max = Number.MAX_SAFE_INTEGER,
): number {
  if (!Number.isFinite(value)) {
    return fallback;
  }
  const next = Math.floor(value);
  if (next < min) {
    return min;
  }
  if (next > max) {
    return max;
  }
  return next;
}

function toNonNegativeInt(
  value: number,
  fallback: number,
  max = Number.MAX_SAFE_INTEGER,
): number {
  if (!Number.isFinite(value)) {
    return fallback;
  }
  const next = Math.floor(value);
  if (next < 0) {
    return 0;
  }
  if (next > max) {
    return max;
  }
  return next;
}

function toRatio(value: number, fallback: number): number {
  if (!Number.isFinite(value)) {
    return fallback;
  }
  if (value < 0) {
    return 0;
  }
  if (value > 1) {
    return 1;
  }
  return value;
}

export function sanitizeExecutionRows(
  rows: number,
  kind: RecipeExecutionKind,
): number {
  return toPositiveInt(rows, kind === "preview" ? 5 : 1000);
}

export function normalizeRunSettings(settings: RecipeRunSettings): RecipeRunSettings {
  return {
    batchSize: toPositiveInt(settings.batchSize, 1000, 1, 200_000),
    batchEnabled: Boolean(settings.batchEnabled),
    mergeBatches: Boolean(settings.mergeBatches),
    llmParallelRequests:
      typeof settings.llmParallelRequests === "number"
        ? toPositiveInt(settings.llmParallelRequests, 4, 1, 2048)
        : null,
    nonInferenceWorkers: toPositiveInt(
      settings.nonInferenceWorkers,
      4,
      1,
      2048,
    ),
    maxConversationRestarts: toNonNegativeInt(
      settings.maxConversationRestarts,
      5,
      100,
    ),
    maxConversationCorrectionSteps: toNonNegativeInt(
      settings.maxConversationCorrectionSteps,
      0,
      100,
    ),
    disableEarlyShutdown: Boolean(settings.disableEarlyShutdown),
    shutdownErrorRate: toRatio(settings.shutdownErrorRate, 0.5),
    shutdownErrorWindow: toPositiveInt(settings.shutdownErrorWindow, 10, 1, 10_000),
  };
}

function buildRunConfigPayload(
  settings: RecipeRunSettings,
  rows: number,
  kind: RecipeExecutionKind,
): Record<string, unknown> {
  const useBatching = kind === "full" && settings.batchEnabled;
  return {
    buffer_size: useBatching ? settings.batchSize : toPositiveInt(rows, 1000, 1, 200_000),
    non_inference_max_parallel_workers: settings.nonInferenceWorkers,
    max_conversation_restarts: settings.maxConversationRestarts,
    max_conversation_correction_steps: settings.maxConversationCorrectionSteps,
    disable_early_shutdown: settings.disableEarlyShutdown,
    shutdown_error_rate: settings.shutdownErrorRate,
    shutdown_error_window: settings.shutdownErrorWindow,
  };
}

function applyGlobalParallelismOverride(
  payload: RecipePayload,
  llmParallelRequests: number | null,
): RecipePayload {
  if (typeof llmParallelRequests !== "number") {
    return payload;
  }

  const modelConfigs = payload.recipe.model_configs.map((modelConfig) => {
    const nextModelConfig = { ...modelConfig };
    const inferenceRaw = modelConfig.inference_parameters;
    const inference =
      inferenceRaw &&
      typeof inferenceRaw === "object" &&
      !Array.isArray(inferenceRaw)
        ? { ...(inferenceRaw as Record<string, unknown>) }
        : {};
    inference.max_parallel_requests = llmParallelRequests;
    nextModelConfig.inference_parameters = inference;
    return nextModelConfig;
  });

  return {
    ...payload,
    recipe: {
      ...payload.recipe,
      model_configs: modelConfigs,
    },
  };
}

export function buildExecutionPayload(input: {
  payload: RecipePayload;
  kind: RecipeExecutionKind;
  rows: number;
  settings: RecipeRunSettings;
  runName?: string | null;
}): RecipePayload {
  const normalizedSettings = normalizeRunSettings(input.settings);
  const payloadWithParallelism = applyGlobalParallelismOverride(
    input.payload,
    normalizedSettings.llmParallelRequests,
  );
  return {
    ...payloadWithParallelism,
    run: {
      ...payloadWithParallelism.run,
      rows: input.rows,
      execution_type: input.kind,
      run_config: buildRunConfigPayload(normalizedSettings, input.rows, input.kind),
      merge_batches:
        input.kind === "full" &&
        normalizedSettings.batchEnabled &&
        normalizedSettings.mergeBatches,
      run_name: input.kind === "full" ? (input.runName ?? null) : null,
    },
  };
}
