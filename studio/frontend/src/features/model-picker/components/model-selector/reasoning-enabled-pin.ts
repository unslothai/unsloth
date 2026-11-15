// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const REASONING_ENABLED_PREFIX = "unsloth_model_reasoning_enabled::";

function readPin(modelId: string): boolean | null {
  try {
    const raw = localStorage.getItem(`${REASONING_ENABLED_PREFIX}${modelId}`);
    if (raw === "true") return true;
    if (raw === "false") return false;
  } catch {
    // Private mode, or storage is full.
  }
  return null;
}

export function pinnedReasoningEnabled(
  modelId: string | null | undefined,
): boolean | null {
  return modelId ? readPin(modelId) : null;
}

export function persistReasoningEnabled(
  modelId: string,
  enabled: boolean,
): void {
  if (!modelId) return;
  try {
    localStorage.setItem(
      `${REASONING_ENABLED_PREFIX}${modelId}`,
      String(enabled),
    );
  } catch {
    // Private mode, or storage is full.
  }
}

export function resolveReasoningEnabledOnLoad(options: {
  reasoningAlwaysOn: boolean;
  supportsReasoning: boolean;
  supportsReasoningOff: boolean;
  pinned: boolean | null;
  reloadingSameModel: boolean;
  previousEnabled: boolean;
  reasoningDefault: boolean;
}): boolean {
  if (options.reasoningAlwaysOn) return true;
  if (!options.supportsReasoning) return options.reasoningDefault;
  if (!options.supportsReasoningOff) return true;
  // Not the pin: a chat switch restores its value without writing it.
  if (options.reloadingSameModel) return options.previousEnabled;
  if (options.pinned !== null) return options.pinned;
  return options.reasoningDefault;
}
