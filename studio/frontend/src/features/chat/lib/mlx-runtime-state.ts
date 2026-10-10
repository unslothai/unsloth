// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  normalizeMlxKvQuant,
  type MlxKvQuant,
} from "@/features/model-picker/model-config/per-model-config";
import type { LoadModelResponse } from "../types/api";

type MlxRuntimeResponse = Pick<
  LoadModelResponse,
  | "is_mlx"
  | "mlx_kv_quant_requested"
  | "mlx_kv_quant_reason"
  | "chat_template_override_reason"
  | "mlx_kv_quant_note"
  | "mlx_int8_prefill_requested"
>;

/** MLX runtime state a load response establishes (KV quantization, template verdict, int8 prefill). A
 *  non-MLX response retires the verdicts but omits mlxKvQuant and mlxInt8Prefill: the settings are
 *  dormant there, not wrong, and a preset carrying the width must survive the round-trip. */
export function mlxRuntimeStateFrom(resp: MlxRuntimeResponse): {
  mlxKvQuant?: MlxKvQuant | null;
  loadedMlxKvQuantRequested: MlxKvQuant | null;
  mlxKvQuantReason: string | null;
  chatTemplateOverrideReason: string | null;
  mlxKvQuantNote: string | null;
  mlxInt8Prefill?: boolean;
  loadedMlxInt8PrefillRequested: boolean;
} {
  if (resp.is_mlx !== true) {
    return {
      loadedMlxKvQuantRequested: null,
      mlxKvQuantReason: null,
      chatTemplateOverrideReason: null,
      mlxKvQuantNote: null,
      loadedMlxInt8PrefillRequested: false,
    };
  }
  const requested = normalizeMlxKvQuant(resp.mlx_kv_quant_requested);
  return {
    mlxKvQuant: requested,
    loadedMlxKvQuantRequested: requested,
    mlxKvQuantReason: resp.mlx_kv_quant_reason ?? null,
    chatTemplateOverrideReason: resp.chat_template_override_reason ?? null,
    mlxKvQuantNote: resp.mlx_kv_quant_note ?? null,
    mlxInt8Prefill: resp.mlx_int8_prefill_requested === true,
    loadedMlxInt8PrefillRequested: resp.mlx_int8_prefill_requested === true,
  };
}
