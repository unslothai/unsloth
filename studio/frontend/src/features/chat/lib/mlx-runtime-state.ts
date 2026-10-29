// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  normalizeMlxKvQuant,
  normalizeMlxOffload,
  type MlxKvQuant,
  type MlxOffload,
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
  | "mlx_offload"
>;

function appliedFraction(value: unknown): number | null {
  const fraction = normalizeMlxOffload(value);
  return typeof fraction === "number" ? fraction : null;
}

export function mlxOffloadStateFrom(resp: Partial<MlxRuntimeResponse>): {
  loadedMlxPleOffloadRequested: MlxOffload;
  loadedMlxExpertOffloadRequested: MlxOffload;
  loadedMlxPleOffload: number | null;
  loadedMlxExpertOffload: number | null;
  mlxPleOffloadReason: string | null;
  mlxExpertOffloadReason: string | null;
  loadedMlxOffloadDiskBytes: number | null;
} {
  const offload = resp.is_mlx === true ? resp.mlx_offload : null;
  const disk = offload?.disk_bytes;
  return {
    loadedMlxPleOffloadRequested: normalizeMlxOffload(offload?.ple?.requested),
    loadedMlxExpertOffloadRequested: normalizeMlxOffload(offload?.experts?.requested),
    loadedMlxPleOffload: appliedFraction(offload?.ple?.applied),
    loadedMlxExpertOffload: appliedFraction(offload?.experts?.applied),
    mlxPleOffloadReason: offload?.ple?.reason ?? null,
    mlxExpertOffloadReason: offload?.experts?.reason ?? null,
    loadedMlxOffloadDiskBytes:
      typeof disk === "number" && Number.isFinite(disk) && disk > 0 ? disk : null,
  };
}

/** MLX runtime state a load response establishes (KV quantization, template verdict, int8 prefill, offload). A
 *  non-MLX response retires the verdicts but omits the chosen settings: they are
 *  dormant there, not wrong, and a preset carrying the width must survive the round-trip. */
export function mlxRuntimeStateFrom(resp: MlxRuntimeResponse): {
  mlxKvQuant?: MlxKvQuant | null;
  loadedMlxKvQuantRequested: MlxKvQuant | null;
  mlxKvQuantReason: string | null;
  chatTemplateOverrideReason: string | null;
  mlxKvQuantNote: string | null;
  mlxInt8Prefill?: boolean;
  loadedMlxInt8PrefillRequested: boolean;
  mlxPleOffload?: MlxOffload;
  mlxExpertOffload?: MlxOffload;
} & ReturnType<typeof mlxOffloadStateFrom> {
  const offload = mlxOffloadStateFrom(resp);
  if (resp.is_mlx !== true) {
    return {
      loadedMlxKvQuantRequested: null,
      mlxKvQuantReason: null,
      chatTemplateOverrideReason: null,
      mlxKvQuantNote: null,
      loadedMlxInt8PrefillRequested: false,
      ...offload,
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
    mlxPleOffload: offload.loadedMlxPleOffloadRequested,
    mlxExpertOffload: offload.loadedMlxExpertOffloadRequested,
    ...offload,
  };
}
