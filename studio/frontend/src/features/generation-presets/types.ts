// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type MediaGenerationKind = "image" | "video";

export interface MediaGenerationPreset<Params, LoadConfig> {
  name: string;
  params: Params;
  loadConfig?: LoadConfig;
}

export interface MediaGenerationPresetState<Params, LoadConfig> {
  currentParams: Params;
  currentLoadConfig?: LoadConfig | null;
  activePreset: string;
}

export interface MediaGenerationPresetSettings<Params, LoadConfig>
  extends MediaGenerationPresetState<Params, LoadConfig> {
  customPresets: MediaGenerationPreset<Params, LoadConfig>[];
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

export interface MediaGenerationPresetLoadConfig {
  speedMode: "auto" | "off" | "eager" | "default" | "max";
  transformerQuant: "auto" | "none" | "fp8" | "int8" | "nvfp4" | "mxfp8";
  attentionBackend: "auto" | "native" | "cudnn" | "flash3" | "sage";
  memoryMode: "auto" | "fast" | "balanced" | "low_vram";
  transformerCache: "auto" | "off" | "fbcache";
}

export interface ImageGenerationPresetLoadConfig
  extends MediaGenerationPresetLoadConfig {
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

export type VideoGenerationPresetLoadConfig = MediaGenerationPresetLoadConfig;
