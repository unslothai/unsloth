// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// what a fresh /api/inference/status does to the Context Length control and its baseline

/**
 * The subset of the pair this rule owns -- the editable Context Length (null =
 * Auto) and the value the resident server was invoked with, as this tab last saw
 * it -- to write back. Empty means leave the store alone.
 */
export interface CtxPinSeed {
  customContextLength?: number | null;
  loadedCustomContextLength?: number | null;
}

const CLEAR: CtxPinSeed = {
  customContextLength: null,
  loadedCustomContextLength: null,
};

/**
 * Status reports `requested_n_ctx` (the n_ctx the running server was launched
 * with) and nothing about whether a human chose it, and the two are not the same
 * question: `resolveLoadMaxSeqLength` sends the resolved context on a same-model
 * reload so the reload does not resize, so an Auto reload under a custom or
 * modified preset reports a positive `requested_n_ctx` exactly like an explicit
 * pin does. A positive value is therefore evidence of nothing on its own, and no
 * backend field could fix that: the frontend sends the same number in both
 * cases, so the backend has no more information than this does.
 *
 * Two exceptions, both where the echo says more than "positive". Under Manual
 * memory with Auto layers the load sends its pin through
 * `resolveFitMaxSeqLength`, which puts 0 on the wire for Auto without exception,
 * so a positive echo there IS an explicit pin and is adopted. And an echo that
 * contradicts the pin this tab recorded means another client reloaded the same
 * model at a different context, so the baseline is dropped rather than resent.
 *
 * Otherwise this never invents a pin out of a positive echo. It clears one only on the
 * unambiguous signal (0, the wire value for Auto, which only an Auto load
 * sends), it leaves the control alone entirely while the model has not changed
 * -- whatever this tab's own load recorded there is better evidence than the
 * echo -- and on a model change, where nothing in the store belongs to the model
 * that arrived, it adopts that model's SAVED Context Length only when the
 * running server corroborates it by matching. That last step is the rule the
 * batch sizes next door already use for their own remembered values.
 */
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
  // while a load is in flight performLoad owns the load params. This is also the
  // mid-load window where status still answers for the OUTGOING model (refreshes
  // do run with modelLoading true), so nothing here can plant a stale pin.
  if (!seedLoadParams) return {};
  // a non-GGUF has no n_ctx: its max_seq_length must not be left standing as one
  if (!isGguf) return CLEAR;
  if (incoming === undefined) {
    // An older backend saying nothing. Nothing recorded against the model that
    // just left may survive it, but a steady poll must not touch the control.
    return modelChanged ? CLEAR : {};
  }
  // 0 is the wire value for Auto and only an Auto load sends it, so this is the
  // one echo that proves its own meaning.
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
    // Same model, ambiguous echo. What is in the store came from watching this
    // tab's own load, which knew what the user actually asked for; an echo that
    // an Auto reload produces just as readily is not grounds to overwrite it,
    // in either direction.
    //
    // Unless it CONTRADICTS the record. Another tab or an API client can reload
    // the same model at a different context, and then the pin here describes an
    // invocation that is no longer running: keeping it means the next unrelated
    // Apply resends it and silently takes back the other client's choice. The
    // echo still cannot say whether a human chose the new value, so it is not
    // adopted; the stale baseline is dropped instead, which leaves the control
    // on Auto and the next Apply agreeing with the server rather than fighting
    // it. Clearing is safe against an Auto reload of our own pinned model too:
    // that reports the resolved context, which IS the pin, so it matches here
    // and nothing moves.
    return loadedPin != null && loadedPin !== incoming ? CLEAR : {};
  }
  // A model changed underneath this tab, so nothing in the store belongs to the
  // one that arrived. Its saved config is the only record of intent left;
  // require the running server to match it, or a stale pin would be re-pinned
  // onto a server that is not running it.
  return remembered != null && remembered > 0 && remembered === incoming
    ? {
        customContextLength: remembered,
        loadedCustomContextLength: remembered,
      }
    : CLEAR;
}
