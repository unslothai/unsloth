// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type MediaGenerationKind = "image" | "video";

// A preset is a generation recipe and nothing else. Model-load options are deliberately absent:
// they apply only on a reload, follow the hardware and checkpoint rather than the recipe, and the
// resident build already owns them (see features/resident-load).
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
