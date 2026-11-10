// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type MediaGenerationKind = "image" | "video" | "training";

// Model-load options are deliberately absent; the resident build owns them.
export interface MediaGenerationPreset<Params> {
  name: string;
  params: Params;
}

export interface MediaGenerationPresetState<Params> {
  currentParams: Params;
  activePreset: string;
}

export interface MediaGenerationPresetSettings<Params>
  extends MediaGenerationPresetState<Params> {
  customPresets: MediaGenerationPreset<Params>[];
  saved?: boolean;
}

export interface ImageGenerationPresetParams {
  negativePrompt: string;
  width: number;
  height: number;
  steps: number;
  guidance: number;
  batchSize: number;
  runs: number;
}

export interface VideoGenerationPresetParams {
  negativePrompt: string;
  width: number;
  height: number;
  durationSeconds: number;
  steps: number;
  guidance: number;
  flowShift: number | null;
  audioFlowShift: number | null;
}

export interface TrainingPresetParams {
  familyName: string;
  baseChoice: string;
  customBase: string;
  durationUnit: "steps" | "epochs";
  steps: number;
  epochs: number;
  learningRate: number;
  rank: number;
  resolution: number;
  batchSize: number;
  gradAccum: number;
  seed: number;
  saveSteps: number;
  sampleEvery: number;
  saveTotalLimit: number;
  lrScheduler: "constant" | "constant_with_warmup" | "cosine" | "linear";
  lrWarmupSteps: number;
  gradCheckpoint: boolean;
  bucketing: boolean;
  randomFlip: boolean;
  cfgDropout: number;
  loraDropout: number;
  flowShiftMode: "default" | "custom";
  flowShift: number;
  precision: "bf16" | "fp16" | "no";
  basePrecision: "nf4" | "bf16" | "int8" | "fp8" | "mxfp8" | "auto";
  compileTransformer: "off" | "on" | "auto";
}
