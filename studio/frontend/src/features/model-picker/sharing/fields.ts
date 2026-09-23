// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { formatExtraArgs } from "../model-config/llama-extra-args";
import type { PerModelConfig } from "../model-config/per-model-config";

export type SharedConfigKey = keyof PerModelConfig;
type Field = { label: string; text?: boolean; error?: string };

export const SHARED_CONFIG_FIELDS: Record<SharedConfigKey, Field> = {
  customContextLength: { label: "Context length" },
  maxSeqLength: { label: "Max sequence length" },
  kvCacheDtype: { label: "KV cache type" },
  mlxKvBits: { label: "MLX KV bits" },
  speculativeType: { label: "Speculative decoding" },
  specDraftNMax: { label: "Draft tokens" },
  specDraftCacheDtype: { label: "Draft KV cache type" },
  nParallel: { label: "Parallel slots" },
  reasoningBudget: { label: "Reasoning budget" },
  reasoningBudgetMessage: {
    label: "Reasoning budget message",
    text: true,
    error:
      "Custom reasoning messages cannot be shared through links. Configure them locally instead.",
  },
  nBatch: { label: "Batch size" },
  nUbatch: { label: "Micro-batch size" },
  loadMode: { label: "Load mode" },
  ctxCheckpoints: { label: "Context checkpoints" },
  cacheRam: { label: "Prompt cache RAM" },
  tensorParallel: { label: "Tensor parallel" },
  disableVision: { label: "Disable vision" },
  chatTemplateOverride: {
    label: "Chat template",
    text: true,
    error:
      "Custom template code cannot be shared through links. Configure it locally instead.",
  },
  llamaExtraArgs: {
    label: "Extra arguments",
    error:
      "Shared extra arguments require supported inference options and valid values. File, network, tool and template options cannot be shared.",
  },
  gpuMemoryMode: { label: "GPU memory mode" },
  gpuLayers: { label: "GPU layers" },
  nCpuMoe: { label: "CPU MoE layers" },
  selectedGpuIds: { label: "GPU devices" },
  selectedGpuIndexKind: { label: "GPU device index type" },
};

export const SHARED_CONFIG_KEYS = Object.keys(
  SHARED_CONFIG_FIELDS,
) as SharedConfigKey[];

export function isSharedConfigKey(key: string): key is SharedConfigKey {
  return Object.hasOwn(SHARED_CONFIG_FIELDS, key);
}

export function formatSharedConfigValue(
  key: SharedConfigKey,
  config: Partial<PerModelConfig>,
): string {
  if (key === "llamaExtraArgs") {
    return formatExtraArgs(config.llamaExtraArgs) || "No extra arguments";
  }
  const value = config[key];
  return value == null || (key === "chatTemplateOverride" && value === "")
    ? "Default"
    : typeof value === "string" && value !== ""
      ? value
      : JSON.stringify(value);
}

export function mergeSharedRunConfig(
  defaults: PerModelConfig,
  patch: Partial<PerModelConfig>,
  isGguf: boolean,
): PerModelConfig {
  const provided = Object.fromEntries(
    SHARED_CONFIG_KEYS.filter((key) => Object.hasOwn(patch, key))
      .map((key) => [key, patch[key]] as const)
      .filter(([, value]) => value !== undefined)
      .map(([key, value]) => [key, Array.isArray(value) ? [...value] : value]),
  );
  if (isGguf && Object.hasOwn(provided, "maxSeqLength")) {
    provided.customContextLength ??= provided.maxSeqLength;
    provided.maxSeqLength = null;
  }
  if (
    Object.hasOwn(provided, "customContextLength") &&
    !Object.hasOwn(provided, "maxSeqLength")
  ) {
    provided.maxSeqLength = null;
  }
  if (
    Object.hasOwn(provided, "maxSeqLength") &&
    !Object.hasOwn(provided, "customContextLength")
  ) {
    provided.customContextLength = null;
  }
  return { ...defaults, ...provided };
}
