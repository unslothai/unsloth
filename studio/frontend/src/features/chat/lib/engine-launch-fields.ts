// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { LoadModelRequest } from "../types/api";
import type { InferenceParams } from "../types/runtime";

/** vLLM / SGLang launch options beyond precision and parallelism; unset means the engine default. */
export type EngineLaunchFields = Required<
  Pick<
    LoadModelRequest,
    "engine_kv_cache_dtype" | "engine_seed" | "engine_gpu_memory_utilization"
  >
>;

export const DEFAULT_ENGINE_LAUNCH_FIELDS: EngineLaunchFields = {
  engine_kv_cache_dtype: "auto",
  engine_seed: null,
  engine_gpu_memory_utilization: null,
};

export function engineLaunchFieldsFromParams(
  params: Pick<
    InferenceParams,
    "engineKvCacheDtype" | "engineSeed" | "engineGpuMemoryUtilization"
  >,
): EngineLaunchFields {
  return {
    engine_kv_cache_dtype: params.engineKvCacheDtype ?? "auto",
    engine_seed: params.engineSeed ?? null,
    engine_gpu_memory_utilization: params.engineGpuMemoryUtilization ?? null,
  };
}

/** What a load or status response says the resident engine was launched with. */
export function engineLaunchFieldsFrom(
  response: Partial<EngineLaunchFields>,
): EngineLaunchFields {
  return {
    engine_kv_cache_dtype: response.engine_kv_cache_dtype ?? "auto",
    engine_seed: response.engine_seed ?? null,
    engine_gpu_memory_utilization:
      response.engine_gpu_memory_utilization ?? null,
  };
}
