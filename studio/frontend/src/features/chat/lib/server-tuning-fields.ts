// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** llama-server load knobs always sent, committed and cleared as a group. */

export interface ServerTuningValues {
  loadMode?: string | null;
  specDraftCacheDtype?: string | null;
  ctxCheckpoints?: number | null;
  cacheRam?: number | null;
  /** Standalone Engram table precision; null keeps the quant's own table. */
  engramPrecision?: string | null;
}

export interface ServerTuningPayload {
  load_mode?: string;
  spec_draft_cache_type?: string;
  ctx_checkpoints?: number;
  cache_ram?: number;
  engram_precision?: string;
}

/** Blank knobs are omitted, not nulled: the route treats a null as set via `model_fields_set`. */
export function serverTuningLoadPayload(
  values: ServerTuningValues,
): ServerTuningPayload {
  return {
    ...(values.loadMode != null ? { load_mode: values.loadMode } : {}),
    ...(values.specDraftCacheDtype != null
      ? { spec_draft_cache_type: values.specDraftCacheDtype }
      : {}),
    ...(values.ctxCheckpoints != null
      ? { ctx_checkpoints: values.ctxCheckpoints }
      : {}),
    ...(values.cacheRam != null ? { cache_ram: values.cacheRam } : {}),
    ...(values.engramPrecision != null
      ? { engram_precision: values.engramPrecision }
      : {}),
  };
}

export interface ServerTuningState {
  loadMode: string | null;
  loadedLoadMode: string | null;
  specDraftCacheDtype: string | null;
  loadedSpecDraftCacheDtype: string | null;
  ctxCheckpoints: number | null;
  loadedCtxCheckpoints: number | null;
  cacheRam: number | null;
  loadedCacheRam: number | null;
  engramPrecision: string | null;
  loadedEngramPrecision: string | null;
}

/** Diffusion commits nothing, or a saved preset would carry values onto the next GGUF. */
export function committedServerTuningState(
  values: ServerTuningValues,
  isDiffusion = false,
): ServerTuningState {
  if (isDiffusion) {
    return clearedServerTuningState();
  }
  const loadMode = values.loadMode ?? null;
  const specDraftCacheDtype = values.specDraftCacheDtype ?? null;
  const ctxCheckpoints = values.ctxCheckpoints ?? null;
  const cacheRam = values.cacheRam ?? null;
  const engramPrecision = values.engramPrecision ?? null;
  return {
    loadMode,
    loadedLoadMode: loadMode,
    specDraftCacheDtype,
    loadedSpecDraftCacheDtype: specDraftCacheDtype,
    ctxCheckpoints,
    loadedCtxCheckpoints: ctxCheckpoints,
    cacheRam,
    loadedCacheRam: cacheRam,
    engramPrecision,
    loadedEngramPrecision: engramPrecision,
  };
}

/** Clears both halves, or a rollback re-sends the departed model's baseline. */
export function clearedServerTuningState(): ServerTuningState {
  return {
    loadMode: null,
    loadedLoadMode: null,
    specDraftCacheDtype: null,
    loadedSpecDraftCacheDtype: null,
    ctxCheckpoints: null,
    loadedCtxCheckpoints: null,
    cacheRam: null,
    loadedCacheRam: null,
    engramPrecision: null,
    loadedEngramPrecision: null,
  };
}
