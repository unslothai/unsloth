// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Recipe Studio's run dialog settings <-> a recipe group's `run`. The backend (`nodes/data.run_settings`) turns the
// saved `run_config` into the job's run: `buffer_size` only batches a full run, otherwise it is the row count.

import { normalizeRunSettings } from "@/features/recipe-studio/executions/run-settings";
import type { RecipeRunSettings } from "@/features/recipe-studio/stores/recipe-executions";

type Run = Record<string, unknown>;

function num(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

export function settingsFromRun(run: Run | null | undefined): RecipeRunSettings {
  const saved = run ?? {};
  const config =
    saved.run_config && typeof saved.run_config === "object"
      ? (saved.run_config as Record<string, unknown>)
      : {};
  return normalizeRunSettings({
    batchSize: num(config.buffer_size, 1000),
    batchEnabled: typeof config.buffer_size === "number",
    mergeBatches: saved.merge_batches === true,
    llmParallelRequests:
      typeof saved.llm_parallel_requests === "number"
        ? saved.llm_parallel_requests
        : null,
    nonInferenceWorkers: num(config.non_inference_max_parallel_workers, 4),
    maxConversationRestarts: num(config.max_conversation_restarts, 5),
    maxConversationCorrectionSteps: num(
      config.max_conversation_correction_steps,
      0,
    ),
    disableEarlyShutdown: config.disable_early_shutdown !== false,
    shutdownErrorRate: num(config.shutdown_error_rate, 0.5),
    shutdownErrorWindow: num(config.shutdown_error_window, 10),
  });
}

/** The group's `run` with these settings; other saved keys (rows, output formats) are kept. */
export function runWithSettings(
  run: Run | null | undefined,
  settings: RecipeRunSettings,
  runName: string,
): Run {
  const s = normalizeRunSettings(settings);
  const next: Run = { ...(run ?? {}) };
  // biome-ignore lint/style/useNamingConvention: backend schema
  next.run_config = {
    ...(s.batchEnabled && { buffer_size: s.batchSize }),
    // biome-ignore lint/style/useNamingConvention: backend schema
    non_inference_max_parallel_workers: s.nonInferenceWorkers,
    // biome-ignore lint/style/useNamingConvention: backend schema
    max_conversation_restarts: s.maxConversationRestarts,
    // biome-ignore lint/style/useNamingConvention: backend schema
    max_conversation_correction_steps: s.maxConversationCorrectionSteps,
    // biome-ignore lint/style/useNamingConvention: backend schema
    disable_early_shutdown: s.disableEarlyShutdown,
    // biome-ignore lint/style/useNamingConvention: backend schema
    shutdown_error_rate: s.shutdownErrorRate,
    // biome-ignore lint/style/useNamingConvention: backend schema
    shutdown_error_window: s.shutdownErrorWindow,
  };
  // biome-ignore lint/style/useNamingConvention: backend schema
  next.merge_batches = s.batchEnabled && s.mergeBatches;
  if (s.llmParallelRequests === null) delete next.llm_parallel_requests;
  // biome-ignore lint/style/useNamingConvention: backend schema
  else next.llm_parallel_requests = s.llmParallelRequests;
  if (runName.trim()) next.run_name = runName.trim();
  else delete next.run_name;
  return next;
}
