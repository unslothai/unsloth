// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** llama-server tuning knobs always sent, committed and cleared as a group. */

export interface ServerTuningValues {
  loadMode?: string | null;
  specDraftCacheDtype?: string | null;
  ctxCheckpoints?: number | null;
  cacheRam?: number | null;
}

export interface ServerTuningPayload {
  load_mode?: string;
  spec_draft_cache_type?: string;
  ctx_checkpoints?: number;
  cache_ram?: number;
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
  return {
    loadMode,
    loadedLoadMode: loadMode,
    specDraftCacheDtype,
    loadedSpecDraftCacheDtype: specDraftCacheDtype,
    ctxCheckpoints,
    loadedCtxCheckpoints: ctxCheckpoints,
    cacheRam,
    loadedCacheRam: cacheRam,
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
  };
}
