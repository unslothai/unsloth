// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Identity of one ModelConfigPage editor instance.
//
// ModelConfigPage seeds its editable state from `loadedConfig` in a useState
// initializer, so it reads that prop exactly once per mounted instance. A host
// that opens the page before /api/inference/status has hydrated, or while the
// target is still loading, gets `loadedConfig` null first and the live config a
// moment later; without the config in the React key the same instance survives
// that flip and keeps showing the saved/default values for a model that is
// running with something else, which Apply then writes back over it.

import type { PerModelConfig } from "./per-model-config";

// Serialize the per-model GPU knobs with the same "absent == default"
// coalescing the store applies: mode auto/absent, gpuLayers Auto (< 0) /
// absent, nCpuMoe 0 / absent, and the GPU pick (null / absent = all GPUs).
export function gpuFieldsSignature(config: PerModelConfig): string {
  return [
    config.gpuMemoryMode ?? "auto",
    config.gpuLayers == null || config.gpuLayers < 0 ? -1 : config.gpuLayers,
    config.nCpuMoe ?? 0,
    config.selectedGpuIds == null
      ? "all"
      : [...config.selectedGpuIds].sort((a, b) => a - b).join(","),
  ].join("|");
}

function hashString(value: string): number {
  let hash = 5381;
  for (let i = 0; i < value.length; i += 1) {
    hash = (Math.imul(hash, 33) ^ value.charCodeAt(i)) >>> 0;
  }
  return hash;
}

/**
 * Signature of the live config an editor was seeded from.
 *
 * `null` (no live config, because the model is not resident or status has not
 * answered yet) is deliberately its own value, distinct from every real config:
 * the arrival of the live config is exactly the transition that has to remount.
 */
export function loadedConfigSignature(
  config: PerModelConfig | null | undefined,
): string {
  if (!config) {
    return "none";
  }
  return [
    config.customContextLength ?? "",
    config.maxSeqLength ?? "",
    config.kvCacheDtype ?? "",
    config.speculativeType ?? "",
    config.specDraftNMax ?? "",
    config.nParallel ?? "",
    config.tensorParallel ? "1" : "0",
    config.chatTemplateOverride == null
      ? ""
      : `${config.chatTemplateOverride.length}:${hashString(config.chatTemplateOverride)}`,
    gpuFieldsSignature(config),
  ].join("|");
}

/**
 * React key for one ModelConfigPage instance. Every host mounts it under this so
 * they agree on when the editor is re-seeded: on a different model, a different
 * quant, or a change in the live config it is meant to be showing.
 */
export function modelConfigInstanceKey(
  modelId: string,
  ggufVariant: string | null | undefined,
  loadedConfig: PerModelConfig | null | undefined,
): string {
  return `${modelId}::${ggufVariant ?? ""}::${loadedConfigSignature(loadedConfig)}`;
}
