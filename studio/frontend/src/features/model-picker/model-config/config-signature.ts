// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Identity of one ModelConfigPage editor instance. The page seeds its state from `loadedConfig`
// in a useState initializer, so it reads that prop once per mount: a host that opens it before
// status hydrates gets null first and the live config a moment later, and without the config
// in the React key the instance keeps showing saved values, which Apply then writes back.

import type { PerModelConfig } from "./per-model-config";

// Serialize the GPU knobs with the store's "absent == default" coalescing: mode auto, gpuLayers
// Auto (< 0), nCpuMoe 0, and null or absent GPU picks as automatic.
export function gpuFieldsSignature(config: PerModelConfig): string {
  const gpuSelection =
    config.selectedGpuIds == null
      ? "automatic"
      : [
          [...config.selectedGpuIds].sort((a, b) => a - b).join(","),
          config.selectedGpuIndexKind === undefined
            ? "physical"
            : (config.selectedGpuIndexKind ?? "deferred"),
        ].join("@");
  return [
    config.gpuMemoryMode ?? "auto",
    config.gpuLayers == null || config.gpuLayers < 0 ? -1 : config.gpuLayers,
    config.nCpuMoe ?? 0,
    gpuSelection,
  ].join("|");
}

function hashString(value: string): number {
  let hash = 5381;
  for (let i = 0; i < value.length; i += 1) {
    hash = (Math.imul(hash, 33) ^ value.charCodeAt(i)) >>> 0;
  }
  return hash;
}

/** Signature of the live config an editor was seeded from. `null` (not resident, or status has
 *  not answered yet) is its own value: the arrival of the live config must remount. */
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
    config.mlxKvBits ?? "",
    config.speculativeType ?? "",
    config.specDraftNMax ?? "",
    config.specDraftCacheDtype ?? "",
    config.nParallel ?? "",
    config.nBatch ?? "",
    config.nUbatch ?? "",
    config.loadMode ?? "",
    config.ctxCheckpoints ?? "",
    config.cacheRam ?? "",
    config.tensorParallel ? "1" : "0",
    config.disableVision ? "1" : "0",
    config.chatTemplateOverride == null
      ? ""
      : `${config.chatTemplateOverride.length}:${hashString(config.chatTemplateOverride)}`,
    gpuFieldsSignature(config),
  ].join("|");
}

/** React key for one ModelConfigPage instance, so every host agrees on when the editor re-seeds:
 *  a different model, a different quant, or a change in the live config. */
export function modelConfigInstanceKey(
  modelId: string,
  ggufVariant: string | null | undefined,
  loadedConfig: PerModelConfig | null | undefined,
): string {
  return `${modelId}::${ggufVariant ?? ""}::${loadedConfigSignature(loadedConfig)}`;
}
