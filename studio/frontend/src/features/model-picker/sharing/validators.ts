// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  CACHE_RAM_MAX,
  CACHE_RAM_MIN,
  CONTEXT_LENGTH_MIN,
  CTX_CHECKPOINTS_MAX,
  CTX_CHECKPOINTS_MIN,
  KV_CACHE_DTYPES,
  LOAD_MODES,
  MAX_SEQ_LENGTH_MAX,
  MAX_SEQ_LENGTH_MIN,
  MLX_KV_BITS,
  N_BATCH_MAX,
  N_BATCH_MIN,
  N_PARALLEL_MAX,
  N_PARALLEL_MIN,
  SPECULATIVE_TYPES,
  isReasoningBudgetMessageValid,
} from "../model-config/per-model-config";
import { validSharedExtraArgs } from "./extra-args";
import type { SharedConfigKey } from "./fields";

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

function validSharedReasoningMessage(value: unknown): value is string {
  if (typeof value !== "string" || !isReasoningBudgetMessageValid(value)) {
    return false;
  }
  for (const character of value) {
    const code = character.codePointAt(0) ?? 0;
    if (
      (code < 32 && code !== 9 && code !== 10 && code !== 13) ||
      (code >= 0x7f && code <= 0x9f) ||
      (code >= 0xd800 && code <= 0xdfff) ||
      code === 0x200b ||
      (code >= 0x202a && code <= 0x202e) ||
      (code >= 0x2060 && code <= 0x2064) ||
      (code >= 0x2066 && code <= 0x2069) ||
      code === 0xfeff
    ) {
      return false;
    }
  }
  return true;
}

const gpuId: Validator = (value) => integer(value, 0, 255);
const cacheType = nullable(
  (value) => value === "f16" || choice(value, KV_CACHE_DTYPES),
);

export const SHARED_CONFIG_VALIDATORS: Record<SharedConfigKey, Validator> = {
  customContextLength: nullable((value) =>
    integer(value, CONTEXT_LENGTH_MIN, 2_147_483_647),
  ),
  maxSeqLength: nullable((value) =>
    integer(value, MAX_SEQ_LENGTH_MIN, MAX_SEQ_LENGTH_MAX),
  ),
  kvCacheDtype: cacheType,
  mlxKvBits: nullable((value) => choice(value, MLX_KV_BITS)),
  speculativeType: nullable((value) => choice(value, SPECULATIVE_TYPES)),
  specDraftNMax: nullable((value) => integer(value, 1, 16)),
  specDraftCacheDtype: cacheType,
  nParallel: nullable((value) =>
    integer(value, N_PARALLEL_MIN, N_PARALLEL_MAX),
  ),
  reasoningBudget: (value) => integer(value, -1, 2_147_483_647),
  reasoningBudgetMessage: validSharedReasoningMessage,
  nBatch: nullable((value) => integer(value, N_BATCH_MIN, N_BATCH_MAX)),
  nUbatch: nullable((value) => integer(value, N_BATCH_MIN, N_BATCH_MAX)),
  loadMode: nullable((value) => choice(value, LOAD_MODES)),
  ctxCheckpoints: nullable((value) =>
    integer(value, CTX_CHECKPOINTS_MIN, CTX_CHECKPOINTS_MAX),
  ),
  cacheRam: nullable((value) => integer(value, CACHE_RAM_MIN, CACHE_RAM_MAX)),
  tensorParallel: boolean,
  disableVision: boolean,
  chatTemplateOverride: nullable((value) => value === ""),
  llamaExtraArgs: nullable(validSharedExtraArgs),
  gpuMemoryMode: (value) => value === "auto" || value === "manual",
  gpuLayers: (value) => integer(value, -1, 2_147_483_647),
  nCpuMoe: (value) => integer(value, 0, 2_147_483_647),
  selectedGpuIds: nullable(
    (value) =>
      Array.isArray(value) &&
      value.length > 0 &&
      value.length <= 256 &&
      value.every(gpuId) &&
      new Set(value).size === value.length,
  ),
  selectedGpuIndexKind: nullable(
    (value) => value === "physical" || value === "vulkan",
  ),
};
