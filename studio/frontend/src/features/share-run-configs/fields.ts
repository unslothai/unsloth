// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { PerModelConfig } from "@/features/model-picker";
import {
  CACHE_RAM_MAX,
  CACHE_RAM_MIN,
  CONTEXT_LENGTH_MIN,
  CTX_CHECKPOINTS_MAX,
  CTX_CHECKPOINTS_MIN,
  KV_CACHE_DTYPES,
  LOAD_MODES,
  MAX_REASONING_BUDGET_MESSAGE_BYTES,
  MAX_SEQ_LENGTH_MAX,
  MAX_SEQ_LENGTH_MIN,
  MLX_KV_BITS,
  N_BATCH_MAX,
  N_BATCH_MIN,
  N_PARALLEL_MAX,
  N_PARALLEL_MIN,
  SPECULATIVE_TYPES,
} from "../model-picker/model-config/per-model-config";
import { validSharedExtraArgs } from "./extra-args";

export type SharedConfigKey = keyof PerModelConfig;
type Validator = (value: unknown) => boolean;
type Field = {
  label: string;
  valid: Validator;
  text?: boolean;
  error?: string;
};
const encoder = new TextEncoder();

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

function invisibleControl(code: number): boolean {
  return (
    (code >= 0x7f && code <= 0x9f) ||
    (code >= 0x202a && code <= 0x202e) ||
    (code >= 0x2066 && code <= 0x2069)
  );
}

function validText(value: unknown, limit: number): value is string {
  if (typeof value !== "string" || value.length > limit) {
    return false;
  }
  for (const character of value) {
    const code = character.codePointAt(0) ?? 0;
    if (code < 32 && code !== 9 && code !== 10 && code !== 13) {
      return false;
    }
    if (code >= 0xd800 && code <= 0xdfff) {
      return false;
    }
    if (invisibleControl(code)) {
      return false;
    }
  }
  return encoder.encode(value).byteLength <= limit;
}

const gpuId: Validator = (value) => integer(value, 0, 255);
const cacheType = nullable(
  (value) => value === "f16" || choice(value, KV_CACHE_DTYPES),
);

export const SHARED_CONFIG_FIELDS: Record<SharedConfigKey, Field> = {
  customContextLength: {
    label: "Context length",
    valid: nullable((value) =>
      integer(value, CONTEXT_LENGTH_MIN, 2_147_483_647),
    ),
  },
  maxSeqLength: {
    label: "Max sequence length",
    valid: nullable((value) =>
      integer(value, MAX_SEQ_LENGTH_MIN, MAX_SEQ_LENGTH_MAX),
    ),
  },
  kvCacheDtype: { label: "KV cache type", valid: cacheType },
  mlxKvBits: {
    label: "MLX KV bits",
    valid: nullable((value) => choice(value, MLX_KV_BITS)),
  },
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
    valid: (value) => validText(value, MAX_REASONING_BUDGET_MESSAGE_BYTES),
    text: true,
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
  chatTemplateOverride: {
    label: "Chat template",
    valid: nullable((value) => value === ""),
    text: true,
    error:
      "Custom template code cannot be shared through links. Configure it locally instead.",
  },
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
