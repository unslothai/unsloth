// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { LoadModelRequest } from "../types/api";
import {
  type ServerTuningValues,
  serverTuningLoadPayload,
} from "./server-tuning-fields";

export interface LlamaServerLoadSettings extends ServerTuningValues {
  reasoningBudget?: number | null;
  reasoningBudgetMessage?: string | null;
  nBatch?: number | null;
  nUbatch?: number | null;
  /** undefined: this config never read them, so the route keeps the stored list. */
  llamaExtraArgs?: string[] | null;
}

// Omit blank knobs: a sent null strips inherited flags via model_fields_set.
export function llamaServerLoadFields(
  settings: LlamaServerLoadSettings,
  target: { isGguf: boolean; isDiffusion: boolean },
): Partial<LoadModelRequest> {
  const llamaServer = target.isGguf && !target.isDiffusion;
  return {
    reasoning_budget: llamaServer ? (settings.reasoningBudget ?? -1) : -1,
    reasoning_budget_message: llamaServer
      ? (settings.reasoningBudgetMessage ?? "")
      : "",
    ...(target.isGguf && settings.nBatch != null
      ? { n_batch: settings.nBatch }
      : {}),
    ...(target.isGguf && settings.nUbatch != null
      ? { n_ubatch: settings.nUbatch }
      : {}),
    ...(llamaServer ? serverTuningLoadPayload(settings) : {}),
    ...(llamaServer && settings.llamaExtraArgs !== undefined
      ? // biome-ignore lint/style/useNamingConvention: API schema
        { llama_extra_args: settings.llamaExtraArgs ?? [] }
      : {}),
  };
}

export function wireLoadRequest(
  payload: LoadModelRequest,
  hfToken: string | null,
) {
  return {
    ...payload,
    hf_token: hfToken,
    native_path_lease: payload.nativePathLease ?? null,
    nativePathLease: undefined,
  };
}

// Forward every load field; an allow-list previously dropped settings that affect validation.
export function validateRequestBody(
  payload: LoadModelRequest,
  hfToken: string | null,
) {
  return {
    ...wireLoadRequest(payload, hfToken),
    engine: payload.engine ?? "auto",
    engine_precision: payload.engine_precision ?? "auto",
    engine_parallelism: payload.engine_parallelism ?? "tensor",
    engine_kv_cache_dtype: payload.engine_kv_cache_dtype ?? "auto",
    tensor_parallel: payload.tensor_parallel ?? false,
    disable_vision: payload.disable_vision ?? false,
    reasoning_budget: payload.reasoning_budget ?? -1,
    reasoning_budget_message: payload.reasoning_budget_message ?? "",
  };
}
