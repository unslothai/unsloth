// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useTrainingConfigStore } from "@/features/training";
import type { RecipeExecutionRecord } from "../execution-types";
import type { TrainingCardConfig } from "../types";

/** The generated-dataset artifact wired into a Train card from an upstream run. */
export type WiredArtifact = {
  executionId: string;
  artifactPath: string;
  runLabel: string;
  rows: number;
};

/**
 * Pick the freshest completed execution that produced a dataset artifact. The
 * executions list is kept sorted most-recent-first, so the first match wins.
 */
export function pickWiredArtifact(
  executions: RecipeExecutionRecord[],
): WiredArtifact | null {
  for (const execution of executions) {
    if (execution.status !== "completed") {
      continue;
    }
    const artifactPath = execution.artifact_path?.trim();
    if (!artifactPath) {
      continue;
    }
    return {
      executionId: execution.id,
      artifactPath,
      runLabel: execution.run_name?.trim() || execution.kind,
      rows: execution.rows,
    };
  }
  return null;
}

/** Resolve the dataset that a Train card should train on, honoring its source mode. */
export function resolveTrainDataset(
  config: TrainingCardConfig,
  wired: WiredArtifact | null,
): { hfDataset: string | null; uploadedFile: string | null } {
  if (config.datasetSource === "huggingface") {
    return { hfDataset: config.hfDataset.trim() || null, uploadedFile: null };
  }
  if (config.datasetSource === "upload") {
    return {
      hfDataset: null,
      uploadedFile: config.uploadedFile.trim() || null,
    };
  }
  // "recipe": auto-wire from the upstream completed run.
  return { hfDataset: null, uploadedFile: wired?.artifactPath ?? null };
}

/**
 * Map a Train card's config into the global training-config store so the
 * existing training runtime can launch it. Dataset fields are derived from the
 * card's source mode (auto-wired artifact, uploaded file, or Hugging Face id).
 */
export function applyTrainCardToTrainingStore(
  config: TrainingCardConfig,
  wired: WiredArtifact | null,
): void {
  const { hfDataset, uploadedFile } = resolveTrainDataset(config, wired);
  const usingHf = config.datasetSource === "huggingface" && Boolean(hfDataset);

  useTrainingConfigStore.setState({
    selectedModel: config.baseModel.trim() || null,
    projectName: config.outputName.trim(),
    trainingMethod: config.trainingMethod,
    hfToken: config.hfToken,
    datasetSource: usingHf ? "huggingface" : "upload",
    datasetFormat: config.datasetFormat,
    dataset: usingHf ? hfDataset : null,
    datasetSubset: usingHf ? config.hfSubset.trim() || null : null,
    datasetSplit: usingHf ? config.hfSplit.trim() || null : null,
    uploadedFile: usingHf ? null : uploadedFile,
    epochs: config.epochs,
    contextLength: config.contextLength,
    learningRate: config.learningRate,
    optimizerType: config.optimizerType,
    lrSchedulerType: config.lrSchedulerType,
    loraRank: config.loraRank,
    loraAlpha: config.loraAlpha,
    loraDropout: config.loraDropout,
    loraVariant: config.loraVariant,
    batchSize: config.batchSize,
    gradientAccumulation: config.gradientAccumulation,
    weightDecay: config.weightDecay,
    warmupSteps: config.warmupSteps,
    maxSteps: config.maxSteps,
    packing: config.packing,
    trainOnCompletions: config.trainOnCompletions,
    gradientCheckpointing: config.gradientCheckpointing,
    randomSeed: config.randomSeed,
    targetModules: [...config.targetModules],
    enableWandb: config.enableWandb,
    wandbProject: config.wandbProject,
    enableTensorboard: config.enableTensorboard,
    tensorboardDir: config.tensorboardDir,
  });
}
