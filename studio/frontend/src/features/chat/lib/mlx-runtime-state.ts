// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type MlxSpeculativeMode,
  normalizeMlxDraftBlockSize,
  normalizeMlxDraftModel,
  normalizeMlxSpeculativeMode,
} from "@/lib/speculative-modes";
import type { LoadModelResponse } from "../types/api";

type MlxRuntimeResponse = Pick<
  LoadModelResponse,
  | "is_mlx"
  | "mlx_kv_bits_requested"
  | "mlx_kv_quant_reason"
  | "chat_template_override_reason"
  | "mlx_kv_quant_note"
  | "mlx_speculative_mode_requested"
  | "mlx_draft_model_requested"
  | "mlx_draft_block_size_requested"
  | "mlx_speculative_reason"
>;

/** MLX runtime state a load response establishes; non-MLX retires the verdicts but keeps the requests. */
export function mlxRuntimeStateFrom(resp: MlxRuntimeResponse): {
  mlxKvBits?: number | null;
  loadedMlxKvBitsRequested: number | null;
  mlxKvQuantReason: string | null;
  chatTemplateOverrideReason: string | null;
  mlxKvQuantNote: string | null;
  mlxSpeculativeMode?: MlxSpeculativeMode;
  mlxDraftModel?: string | null;
  mlxDraftBlockSize?: number | null;
  loadedMlxSpeculativeMode: MlxSpeculativeMode | null;
  loadedMlxDraftModel: string | null;
  loadedMlxDraftBlockSize: number | null;
  mlxSpeculativeReason: string | null;
} {
  if (resp.is_mlx !== true) {
    return {
      loadedMlxKvBitsRequested: null,
      mlxKvQuantReason: null,
      chatTemplateOverrideReason: null,
      mlxKvQuantNote: null,
      loadedMlxSpeculativeMode: null,
      loadedMlxDraftModel: null,
      loadedMlxDraftBlockSize: null,
      mlxSpeculativeReason: null,
    };
  }
  // No speculation field means an older backend, not Off.
  const mode = normalizeMlxSpeculativeMode(resp.mlx_speculative_mode_requested, "auto");
  const draftModel = normalizeMlxDraftModel(resp.mlx_draft_model_requested, mode);
  const blockSize = normalizeMlxDraftBlockSize(resp.mlx_draft_block_size_requested, mode);
  return {
    // Requested, not applied: a refusal applies no width but still has a reason.
    mlxKvBits: resp.mlx_kv_bits_requested ?? null,
    loadedMlxKvBitsRequested: resp.mlx_kv_bits_requested ?? null,
    mlxKvQuantReason: resp.mlx_kv_quant_reason ?? null,
    chatTemplateOverrideReason: resp.chat_template_override_reason ?? null,
    mlxKvQuantNote: resp.mlx_kv_quant_note ?? null,
    mlxSpeculativeMode: mode,
    mlxDraftModel: draftModel,
    mlxDraftBlockSize: blockSize,
    loadedMlxSpeculativeMode: mode,
    loadedMlxDraftModel: draftModel,
    loadedMlxDraftBlockSize: blockSize,
    mlxSpeculativeReason: resp.mlx_speculative_reason ?? null,
  };
}

type MlxSpeculativeStoreState = {
  mlxSpeculativeMode: MlxSpeculativeMode;
  mlxDraftModel: string | null;
  mlxDraftBlockSize: number | null;
  loadedMlxSpeculativeMode: MlxSpeculativeMode | null;
  loadedMlxDraftModel: string | null;
  loadedMlxDraftBlockSize: number | null;
  mlxSpeculativeReason?: string | null;
};

/** The requested half is only refreshed when nothing is pending. */
export function reconcileMlxSpeculativeStatus(
  previous: MlxSpeculativeStoreState,
  response: MlxRuntimeResponse,
  hydratingNewModel: boolean,
): Partial<MlxSpeculativeStoreState> {
  const runtime = mlxRuntimeStateFrom(response);
  const fields: Partial<MlxSpeculativeStoreState> = {
    loadedMlxSpeculativeMode: runtime.loadedMlxSpeculativeMode,
    loadedMlxDraftModel: runtime.loadedMlxDraftModel,
    loadedMlxDraftBlockSize: runtime.loadedMlxDraftBlockSize,
    mlxSpeculativeReason: runtime.mlxSpeculativeReason,
  };
  const editsPending =
    previous.mlxSpeculativeMode !==
      (previous.loadedMlxSpeculativeMode ?? "auto") ||
    previous.mlxDraftModel !== previous.loadedMlxDraftModel ||
    previous.mlxDraftBlockSize !== previous.loadedMlxDraftBlockSize;
  if (
    runtime.mlxSpeculativeMode !== undefined &&
    (hydratingNewModel || !editsPending)
  ) {
    fields.mlxSpeculativeMode = runtime.mlxSpeculativeMode;
    fields.mlxDraftModel = runtime.mlxDraftModel;
    fields.mlxDraftBlockSize = runtime.mlxDraftBlockSize;
  }
  return fields;
}
