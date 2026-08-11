// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type MediaGenerationKind = "image" | "video";
export type MediaPresetSource = "builtin-default" | "custom" | "modified";

export interface MediaGenerationPreset<Params, LoadConfig> {
  name: string;
  params: Params;
  loadConfig?: LoadConfig;
}

export interface MediaGenerationPresetSettings<Params, LoadConfig> {
  currentParams: Params;
  currentLoadConfig?: LoadConfig | null;
  customPresets: MediaGenerationPreset<Params, LoadConfig>[];
  activePreset: string;
  activePresetSource: MediaPresetSource;
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

export interface ImageGenerationPresetLoadConfig {
  speedMode: "auto" | "off" | "eager" | "default" | "max";
  transformerQuant: "auto" | "none" | "fp8" | "int8" | "nvfp4" | "mxfp8";
  attentionBackend: "auto" | "native" | "cudnn" | "flash3" | "sage";
  memoryMode: "auto" | "fast" | "balanced" | "low_vram";
  transformerCache: "auto" | "off" | "fbcache";
  cpuOffload: boolean;
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

export interface VideoGenerationPresetLoadConfig {
  memoryMode: "auto" | "fast" | "balanced" | "low_vram";
  speedMode: "auto" | "off" | "eager" | "default" | "max";
  attentionBackend: "auto" | "native" | "cudnn" | "flash3" | "sage";
  transformerCache: "auto" | "off" | "fbcache";
  transformerQuant: "auto" | "none" | "fp8" | "int8" | "nvfp4" | "mxfp8";
}
