// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { LoadModelResponse } from "../types/api";

type MlxRuntimeResponse = Pick<
  LoadModelResponse,
  | "is_mlx"
  | "mlx_kv_bits_requested"
  | "mlx_turboquant"
  | "mlx_kv_quant_reason"
  | "chat_template_override_reason"
  | "mlx_kv_quant_note"
>;

/** MLX KV-quantization state a load response establishes. A non-MLX response retires the verdict but
 *  omits mlxKvBits: the width is dormant there, not wrong, and a preset carrying it must survive
 *  the round-trip. */
export function mlxRuntimeStateFrom(resp: MlxRuntimeResponse): {
  mlxKvBits?: number | null;
  mlxTurboQuant?: boolean;
  loadedMlxKvBitsRequested: number | null;
  loadedMlxTurboQuant: boolean;
  mlxKvQuantReason: string | null;
  chatTemplateOverrideReason: string | null;
  mlxKvQuantNote: string | null;
} {
  if (resp.is_mlx !== true) {
    return {
      loadedMlxKvBitsRequested: null,
      loadedMlxTurboQuant: false,
      mlxKvQuantReason: null,
      chatTemplateOverrideReason: null,
      mlxKvQuantNote: null,
    };
  }
  return {
    // Requested, not applied: a refusal has a reason but no width.
    mlxKvBits: resp.mlx_kv_bits_requested ?? null,
    mlxTurboQuant: resp.mlx_turboquant ?? false,
    loadedMlxKvBitsRequested: resp.mlx_kv_bits_requested ?? null,
    loadedMlxTurboQuant: resp.mlx_turboquant ?? false,
    mlxKvQuantReason: resp.mlx_kv_quant_reason ?? null,
    chatTemplateOverrideReason: resp.chat_template_override_reason ?? null,
    mlxKvQuantNote: resp.mlx_kv_quant_note ?? null,
  };
}
