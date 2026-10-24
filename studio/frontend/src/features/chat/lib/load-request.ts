// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The shape of a chat model load request. Every load path builds one LoadModelRequest and sends it
 *  to both /validate and /load, so the preflight judges the command the load runs. */

import type { LoadModelRequest } from "../types/api";
import {
  type ServerTuningValues,
  serverTuningLoadPayload,
} from "./server-tuning-fields";

/** The subset of a config these read; satisfied by PerModelConfig. */
export interface LlamaServerLoadSettings extends ServerTuningValues {
  reasoningBudget?: number | null;
  reasoningBudgetMessage?: string | null;
  nBatch?: number | null;
  nUbatch?: number | null;
  /** undefined: this config never read them, so the route keeps the stored list. */
  llamaExtraArgs?: string[] | null;
}

/** The llama-server knobs of a chat load, under the one rule every load path uses. GGUF only, and the
 *  reasoning budget, server tuning and pass-through args never reach the diffusion runner, which takes
 *  none of llama-server's flags. Blank knobs are omitted rather than nulled: the route reads
 *  `model_fields_set`, and a null counts as set and strips the flag from inherited extra args. */
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

/** A load request as the backend reads it: the prepared token, and the Desktop lease in its wire name. */
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

/** /validate's body is the /load request itself. The route ignores the load-only fields; a hand-kept
 *  list of the rest dropped ctx_checkpoints, spec_draft_cache_type and tensor_split. Only its
 *  non-nullable fields need their defaults. */
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
