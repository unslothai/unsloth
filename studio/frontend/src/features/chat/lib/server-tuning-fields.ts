// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The four llama-server tuning knobs every GGUF load carries: --load-mode, the draft context's
 *  KV cache dtype, --ctx-checkpoints and --cache-ram. One module because they are always sent,
 *  committed and cleared as a group, and every load path would otherwise repeat the same four
 *  lines with three chances to forget one. State still lives in the runtime store. */

/** The subset of a config these read; satisfied by PerModelConfig and the store. */
export interface ServerTuningValues {
  loadMode?: string | null;
  specDraftCacheDtype?: string | null;
  ctxCheckpoints?: number | null;
  cacheRam?: number | null;
}

/** The request fields, in the backend's spelling. */
export interface ServerTuningPayload {
  load_mode?: string;
  spec_draft_cache_type?: string;
  ctx_checkpoints?: number;
  cache_ram?: number;
}

/** What to send on /load, blank knobs omitted rather than nulled: the route reads
 *  `model_fields_set` to decide whether the control owns the flag, and a null counts as set,
 *  stripping the flag out of any inherited extra arguments. */
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

/** The control/baseline pairs the store keeps for these four. */
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

/** What a launch committed: click-time values, not a backend echo, like the batch sizes. Diffusion
 *  commits nothing, since it launches no llama-server and a value recorded here would ride a
 *  saved preset onto the next GGUF. */
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

/** The pairs a load that sent none of them leaves behind. Both halves, or a rollback re-sends the
 *  departed model's baseline as if this server ran it. */
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
