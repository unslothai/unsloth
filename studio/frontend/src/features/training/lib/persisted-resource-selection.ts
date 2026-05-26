// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const MODEL_FORMATS = new Set([
  "gguf",
  "safetensors",
  "adapter",
  "checkpoint",
  "unknown",
]);

export function migratePersistedResourceSelections(
  state: Record<string, unknown>,
): void {
  if (typeof state.modelKnownCached !== "boolean") {
    state.modelKnownCached = false;
  }
  if (typeof state.modelLocalPath !== "string") {
    state.modelLocalPath = null;
  }
  if (typeof state.modelFormat !== "string" || !MODEL_FORMATS.has(state.modelFormat)) {
    state.modelFormat = null;
  }
  if (typeof state.datasetKnownCached !== "boolean") {
    state.datasetKnownCached = false;
  }
  if (typeof state.datasetLocalPath !== "string") {
    state.datasetLocalPath = null;
  }
  state.datasetManualMapping ??= {};
  state.datasetLabelMapping ??= {};
  state.datasetSystemPrompt ??= "";
  state.datasetUserTemplate ??= "";
  state.datasetAssistantTemplate ??= "";
  state.datasetAdvisorNotification ??= null;
  state.datasetSliceStart ??= null;
  state.datasetSliceEnd ??= null;
  state.uploadedFile ??= null;
  state.uploadedEvalFile ??= null;
}
