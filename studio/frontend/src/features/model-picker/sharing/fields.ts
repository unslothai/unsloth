// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { formatExtraArgs } from "../model-config/llama-extra-args";
import {
  type PerModelConfig,
  CACHE_RAM_MAX,
  CACHE_RAM_MIN,
  CONTEXT_LENGTH_MIN,
  CTX_CHECKPOINTS_MAX,
  CTX_CHECKPOINTS_MIN,
  KV_CACHE_DTYPES,
  LOAD_MODES,
  N_BATCH_MAX,
  N_BATCH_MIN,
  N_PARALLEL_MAX,
  N_PARALLEL_MIN,
  SPECULATIVE_TYPES,
} from "../model-config/per-model-config";
import { validSharedExtraArgs } from "./extra-args";

type Validator = (value: unknown) => boolean;

const integer = (value: unknown, min: number, max: number): boolean =>
  typeof value === "number" &&
  Number.isSafeInteger(value) &&
  value >= min &&
  value <= max;
const nullable =
  (valid: Validator): Validator =>
  (value) =>
    value === null || valid(value);
const choice = (value: unknown, values: readonly unknown[]): boolean =>
  values.includes(value);
const boolean: Validator = (value) => typeof value === "boolean";

const gpuId: Validator = (value) => integer(value, 0, 255);
const cacheType = nullable(
  (value) => value === "f16" || choice(value, KV_CACHE_DTYPES),
);

// Engine choices stay local: the recipient may not have vLLM or SGLang installed.
export type SharedConfigKey = Exclude<
  keyof PerModelConfig,
  | "chatTemplateOverride"
  | "tensorSplit"
  | "maxSeqLength"
  | "mlxKvQuant"
  | "mlxInt8Prefill"
  | "engine"
  | "enginePrecision"
  | "engineParallelism"
>;
type Field = { label: string; valid: Validator; error?: string };

export const SHARED_CONFIG_FIELDS: Record<SharedConfigKey, Field> = {
  customContextLength: {
    label: "Context length",
    valid: nullable((value) =>
      integer(value, CONTEXT_LENGTH_MIN, 2_147_483_647),
    ),
  },
  kvCacheDtype: { label: "KV cache type", valid: cacheType },
  speculativeType: {
    label: "Speculative decoding",
    valid: nullable((value) => choice(value, SPECULATIVE_TYPES)),
  },
  specDraftNMax: {
    label: "Draft tokens",
    valid: nullable((value) => integer(value, 1, 16)),
  },
  specDraftCacheDtype: { label: "Draft KV cache type", valid: cacheType },
  nParallel: {
    label: "Parallel slots",
    valid: nullable((value) => integer(value, N_PARALLEL_MIN, N_PARALLEL_MAX)),
  },
  reasoningBudget: {
    label: "Reasoning budget",
    valid: (value) => integer(value, -1, 2_147_483_647),
  },
  reasoningBudgetMessage: {
    label: "Reasoning budget message",
    valid: (value) => value === "",
    error:
      "Custom reasoning messages cannot be shared through links. Configure them locally instead.",
  },
  nBatch: {
    label: "Batch size",
    valid: nullable((value) => integer(value, N_BATCH_MIN, N_BATCH_MAX)),
  },
  nUbatch: {
    label: "Micro-batch size",
    valid: nullable((value) => integer(value, N_BATCH_MIN, N_BATCH_MAX)),
  },
  loadMode: {
    label: "Load mode",
    valid: nullable((value) => choice(value, LOAD_MODES)),
  },
  ctxCheckpoints: {
    label: "Context checkpoints",
    valid: nullable((value) =>
      integer(value, CTX_CHECKPOINTS_MIN, CTX_CHECKPOINTS_MAX),
    ),
  },
  cacheRam: {
    label: "Prompt cache RAM",
    valid: nullable((value) => integer(value, CACHE_RAM_MIN, CACHE_RAM_MAX)),
  },
  tensorParallel: { label: "Tensor parallel", valid: boolean },
  disableVision: { label: "Disable vision", valid: boolean },
  llamaExtraArgs: {
    label: "Extra arguments",
    valid: nullable(validSharedExtraArgs),
    error:
      "Shared extra arguments require supported inference options and valid values. File, network, tool and template options cannot be shared.",
  },
  gpuMemoryMode: {
    label: "GPU memory mode",
    valid: (value) => value === "auto" || value === "manual",
  },
  gpuLayers: {
    label: "GPU layers",
    valid: (value) => integer(value, -1, 2_147_483_647),
  },
  nCpuMoe: {
    label: "CPU MoE layers",
    valid: (value) => integer(value, 0, 2_147_483_647),
  },
  selectedGpuIds: {
    label: "GPU devices",
    valid: nullable(
      (value) =>
        Array.isArray(value) &&
        value.length > 0 &&
        value.length <= 256 &&
        value.every(gpuId) &&
        new Set(value).size === value.length,
    ),
  },
  selectedGpuIndexKind: {
    label: "GPU device index type",
    valid: nullable((value) => value === "physical" || value === "vulkan"),
  },
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
  return value == null
    ? "Default"
    : typeof value === "string" && value !== ""
      ? value
      : JSON.stringify(value);
}

export function mergeSharedRunConfig(
  defaults: PerModelConfig,
  patch: Partial<PerModelConfig>,
): PerModelConfig {
  const provided = Object.fromEntries(
    SHARED_CONFIG_KEYS.filter((key) => Object.hasOwn(patch, key))
      .map((key) => [key, patch[key]] as const)
      .filter(([, value]) => value !== undefined)
      .map(([key, value]) => [key, Array.isArray(value) ? [...value] : value]),
  );
  const gpuSelectionChanged = (
    ["selectedGpuIds", "selectedGpuIndexKind"] as const
  ).some(
    (key) =>
      Object.hasOwn(provided, key) &&
      JSON.stringify(provided[key] ?? null) !==
        JSON.stringify(defaults[key] ?? null),
  );
  if (
    gpuSelectionChanged ||
    provided.gpuMemoryMode === "auto" ||
    provided.gpuLayers === -1
  ) {
    provided.tensorSplit = null;
  }
  return { ...defaults, ...provided };
}
