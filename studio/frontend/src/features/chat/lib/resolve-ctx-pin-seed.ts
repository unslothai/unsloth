// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// what a fresh /api/inference/status does to the Context Length control and its baseline

/** The subset of the pair this rule owns -- the editable Context Length (null = Auto) and the
 *  value the resident server was invoked with, as this tab last saw it -- to write back.
 *  Empty means leave the store alone. */
export interface CtxPinSeed {
  customContextLength?: number | null;
  loadedCustomContextLength?: number | null;
}

const CLEAR: CtxPinSeed = {
  customContextLength: null,
  loadedCustomContextLength: null,
};

/** Status reports `requested_n_ctx` and nothing about whether a human chose it, and the two are
 *  not the same question: `resolveLoadMaxSeqLength` sends the resolved context on a same-model
 *  reload so the reload does not resize, so an Auto reload under a custom preset reports a
 *  positive echo exactly like an explicit pin does. No backend field could fix that, since the
 *  frontend sends the same number in both cases. Two exceptions, where the echo says more than
 *  "positive": under Manual memory with Auto layers the load puts 0 on the wire for Auto
 *  without exception, so a positive echo IS a pin; and an echo contradicting the recorded pin
 *  means another client reloaded at a different context, so the baseline is dropped.
 *  Otherwise this never invents a pin, clears one only on the unambiguous 0, leaves the
 *  control alone while the model has not changed, and on a model change adopts the saved
 *  Context Length only when the running server matches it. */
export function resolveCtxPinSeed(options: {
  /** ``status.requested_context_length``; undefined on a backend that omits the field. */
  incoming: number | null | undefined;
  /** ``status.is_gguf``. A non-GGUF status reports ``incoming`` too, so this flag, not
   * its presence, is what keeps a non-GGUF load out of the pin. */
  isGguf: boolean;
  /** MLX sizes its own window, and an unpinned MLX load sends 0, so a positive echo
   *  from it is proof of a pin rather than the ambiguous resolved n_ctx GGUF reports. */
  isMlx?: boolean;
  /** No load of this tab's own is in flight (``!modelLoading``). */
  seedLoadParams: boolean;
  /** The model/variant changed underneath this tab, so nothing recorded here survives it. */
  modelChanged: boolean;
  /** This model's SAVED Context Length, or null when it saved none / has no record. */
  remembered: number | null;
  /** ``status.gpu_memory_mode`` for the resident server. */
  gpuMemoryMode?: "auto" | "manual" | null;
  /** ``status.gpu_layers`` as reported, NOT normalised: negative means Auto layers. */
  gpuLayers?: number | null;
  /** ``loadedCustomContextLength``: the n_ctx this tab last recorded for this server. */
  loadedPin?: number | null;
}): CtxPinSeed {
  const {
    incoming,
    isGguf,
    isMlx,
    seedLoadParams,
    modelChanged,
    remembered,
    gpuMemoryMode,
    gpuLayers,
    loadedPin,
  } = options;
  // While a load is in flight performLoad owns the load params. This is also the mid-load window
  // where status still answers for the OUTGOING model, so nothing here can plant a stale pin.
  if (!seedLoadParams) return {};
  // a non-GGUF has no n_ctx: its max_seq_length must not be left standing as one
  if (!isGguf) return CLEAR;
  if (incoming === undefined) {
    // An older backend saying nothing. Nothing recorded against the model that just left may
    // survive it, but a steady poll must not touch the control.
    return modelChanged ? CLEAR : {};
  }
  // 0 is the wire value for Auto and only an Auto load sends it, so this is the one echo that proves its own meaning.
  if (incoming === null || !(incoming > 0)) return CLEAR;
  // Unambiguous on MLX, so it is adopted without needing a saved config to corroborate
  // it: another tab or an API client leaving a pinned model resident would otherwise
  // read as Auto here, and the next Apply would send 0 and drop their pin.
  if (isMlx) {
    return { customContextLength: incoming, loadedCustomContextLength: incoming };
  }
  // One placement mode has no ambiguity to reason around: under Manual memory
  // with Auto layers the load sends its pin as max_seq_length through
  // resolveFitMaxSeqLength, which answers `customContextLength > 0 ? it : 0`.
  // Auto there is 0 on the wire, always, so a POSITIVE echo in this mode is
  // proof of an explicit pin rather than evidence of nothing. Read before the
  // branches below because it outranks both of them: it is better evidence than
  // a saved config, and on a model change it describes the model that arrived.
  if (gpuMemoryMode === "manual" && gpuLayers != null && gpuLayers < 0) {
    return { customContextLength: incoming, loadedCustomContextLength: incoming };
  }
  if (!modelChanged) {
    // Same model, ambiguous echo. What is in the store came from watching this tab's own load,
    // which knew what the user asked for; an echo an Auto reload produces just as readily is not
    // grounds to overwrite it. Unless it CONTRADICTS the record: another client can reload the
    // same model at a different context, and keeping the pin means the next Apply silently takes
    // back that choice. The echo still cannot say whether a human chose the new value, so the
    // stale baseline is dropped instead, leaving the control on Auto. Safe against an Auto reload
    // of our own pinned model, which reports the resolved context and so matches.
    return loadedPin != null && loadedPin !== incoming ? CLEAR : {};
  }
  // A model changed underneath this tab, so nothing in the store belongs to the one that arrived.
  // Its saved config is the only record of intent left; require the running server to match it,
  // or a stale pin would be re-pinned onto a server that is not running it.
  return remembered != null && remembered > 0 && remembered === incoming
    ? {
        customContextLength: remembered,
        loadedCustomContextLength: remembered,
      }
    : CLEAR;
}
