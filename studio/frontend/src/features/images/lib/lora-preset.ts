// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DiffusionLoraInfo } from "../api.ts";

export interface StepRecipe {
  steps: number;
  guidance: number;
}

/** The steps / guidance a selected step-distilled LoRA runs at (e.g. Lightning 8 steps at CFG 1), or null. */
export function loraPresetRecipe(
  selected: ReadonlyArray<{ id: string }>,
  catalog: ReadonlyArray<DiffusionLoraInfo>,
): StepRecipe | null {
  for (const spec of selected) {
    const id = spec.id.trim();
    const preset = catalog.find((l) => l.id === id);
    if (preset?.recommended_steps) {
      return { steps: preset.recommended_steps, guidance: preset.recommended_guidance ?? 1 };
    }
  }
  return null;
}

/** The recipe a model pick applies: a still-selected distilled LoRA keeps its own, else the model's. */
export function pickRecipe(
  modelDefaults: StepRecipe,
  selected: ReadonlyArray<{ id: string }>,
  catalog: ReadonlyArray<DiffusionLoraInfo>,
): StepRecipe {
  return loraPresetRecipe(selected, catalog) ?? modelDefaults;
}
