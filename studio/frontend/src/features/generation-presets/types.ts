// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type MediaGenerationKind = "image" | "video";

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
