// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { PerModelConfig } from "./per-model-config";

interface SettingReset {
  isDefault: (config: PerModelConfig) => boolean;
  patch: Partial<PerModelConfig>;
}

// Each patch matches what picking the row's own default writes, so a field reset and a manual one store the same thing.
const SETTING_RESETS = {
  contextLength: {
    isDefault: (c) => c.customContextLength == null,
    patch: { customContextLength: null },
  },
  kvCacheDtype: {
    isDefault: (c) => c.kvCacheDtype == null,
    patch: { kvCacheDtype: null },
  },
  mlxKvQuant: {
    isDefault: (c) => c.mlxKvQuant == null,
    patch: { mlxKvQuant: null },
  },
  // The draft depth and draft cache only exist under a strategy, so they go with it.
  speculative: {
    // The Auto option and a loaded model's status both write "auto", which reads as the default.
    isDefault: (c) => {
      const type = c.speculativeType?.trim().toLowerCase();
      return !type || type === "auto" || type === "default";
    },
    patch: {
      speculativeType: null,
      specDraftNMax: null,
      specDraftCacheDtype: null,
    },
  },
  specDraftNMax: {
    isDefault: (c) => c.specDraftNMax == null,
    patch: { specDraftNMax: null },
  },
  specDraftCacheDtype: {
    isDefault: (c) => c.specDraftCacheDtype == null,
    patch: { specDraftCacheDtype: null },
  },
  nParallel: {
    isDefault: (c) => c.nParallel == null,
    patch: { nParallel: null },
  },
  nBatch: {
    isDefault: (c) => c.nBatch == null,
    patch: { nBatch: null },
  },
  nUbatch: {
    isDefault: (c) => c.nUbatch == null,
    patch: { nUbatch: null },
  },
  tensorParallel: {
    isDefault: (c) => !c.tensorParallel,
    patch: { tensorParallel: false },
  },
  vision: {
    isDefault: (c) => !c.disableVision,
    patch: { disableVision: false },
  },
  reasoningBudget: {
    isDefault: (c) => c.reasoningBudget === -1,
    patch: { reasoningBudget: -1 },
  },
  reasoningBudgetMessage: {
    isDefault: (c) => c.reasoningBudgetMessage === "",
    patch: { reasoningBudgetMessage: "" },
  },
  // Same clear as choosing Default in the GPU Memory select.
  gpuMemory: {
    isDefault: (c) =>
      (c.gpuMemoryMode ?? "auto") === "auto" &&
      (c.gpuLayers == null || c.gpuLayers < 0) &&
      (c.nCpuMoe == null || c.nCpuMoe === 0) &&
      c.selectedGpuIds == null,
    patch: {
      gpuMemoryMode: "auto",
      gpuLayers: undefined,
      nCpuMoe: undefined,
      selectedGpuIds: undefined,
      selectedGpuIndexKind: undefined,
      tensorSplit: null,
    },
  },
  loadMode: {
    isDefault: (c) => c.loadMode == null,
    patch: { loadMode: null },
  },
  ctxCheckpoints: {
    isDefault: (c) => c.ctxCheckpoints == null,
    patch: { ctxCheckpoints: null },
  },
  cacheRam: {
    isDefault: (c) => c.cacheRam == null,
    patch: { cacheRam: null },
  },
} satisfies Record<string, SettingReset>;

export type ResettableSetting = keyof typeof SETTING_RESETS;

export function settingIsDefault(
  config: PerModelConfig,
  setting: ResettableSetting,
): boolean {
  return SETTING_RESETS[setting].isDefault(config);
}

export function settingResetPatch(
  setting: ResettableSetting,
): Partial<PerModelConfig> {
  return { ...SETTING_RESETS[setting].patch };
}
