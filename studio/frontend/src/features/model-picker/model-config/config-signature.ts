// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// ModelConfigPage seeds state once per mount, so the live config must be in the React key or Apply
// writes back stale saved values.

import type { PerModelConfig } from "./per-model-config";

// Uses the store's "absent == default" coalescing.
export function gpuFieldsSignature(config: PerModelConfig): string {
  const gpuSelection =
    config.selectedGpuIds == null
      ? "automatic"
      : [
          // Order-preserving: list order is device order, so a reorder must count as a change.
          config.selectedGpuIds.join(","),
          config.selectedGpuIndexKind === undefined
            ? "physical"
            : (config.selectedGpuIndexKind ?? "deferred"),
        ].join("@");
  return [
    config.gpuMemoryMode ?? "auto",
    config.gpuLayers == null || config.gpuLayers < 0 ? -1 : config.gpuLayers,
    config.nCpuMoe ?? 0,
    gpuSelection,
    (config.tensorSplit ?? []).join(","),
  ].join("|");
}

function hashString(value: string): number {
  let hash = 5381;
  for (let i = 0; i < value.length; i += 1) {
    hash = (Math.imul(hash, 33) ^ value.charCodeAt(i)) >>> 0;
  }
  return hash;
}

/** `null` is its own value: the live config's arrival must remount. */
export function loadedConfigSignature(
  config: PerModelConfig | null | undefined,
): string {
  if (!config) {
    return "none";
  }
  return [
    config.engine ?? "auto",
    config.enginePrecision ?? "auto",
    config.engineParallelism ?? "tensor",
    config.customContextLength ?? "",
    config.maxSeqLength ?? "",
    config.kvCacheDtype ?? "",
    config.mlxKvQuant ?? "",
    config.mlxInt8Prefill ? "1" : "0",
    config.speculativeType ?? "",
    config.specDraftNMax ?? "",
    config.specDraftCacheDtype ?? "",
    config.nParallel ?? "",
    config.reasoningBudget ?? "",
    `${(config.reasoningBudgetMessage ?? "").length}:${hashString(config.reasoningBudgetMessage ?? "")}`,
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
    config.llamaExtraArgs == null
      ? ""
      : `${config.llamaExtraArgs.length}:${hashString(config.llamaExtraArgs.join("\u0000"))}`,
    gpuFieldsSignature(config),
  ].join("|");
}

export function modelConfigInstanceKey(
  modelId: string,
  ggufVariant: string | null | undefined,
  loadedConfig: PerModelConfig | null | undefined,
): string {
  return `${modelId}::${ggufVariant ?? ""}::${loadedConfigSignature(loadedConfig)}`;
}
