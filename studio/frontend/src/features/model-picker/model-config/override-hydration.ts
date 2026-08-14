// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const MODEL_OVERRIDE_HYDRATION_MAX_ATTEMPTS = 2;

export interface OverrideHydrationDecisionInput {
  requestGeneration: number;
  currentGeneration: number;
  requestRememberGeneration: number;
  currentRememberGeneration: number;
  hasLocalLlamaExtraArgs: boolean;
  hasServerLlamaExtraArgs: boolean;
}

export function decideOverrideHydration({
  requestGeneration,
  currentGeneration,
  requestRememberGeneration,
  currentRememberGeneration,
  hasLocalLlamaExtraArgs,
  hasServerLlamaExtraArgs,
}: OverrideHydrationDecisionInput): {
  applyArgs: boolean;
  applyRemember: boolean;
} {
  const current =
    requestGeneration === currentGeneration && hasServerLlamaExtraArgs;
  return {
    applyArgs: current && !hasLocalLlamaExtraArgs,
    applyRemember:
      current && requestRememberGeneration === currentRememberGeneration,
  };
}
