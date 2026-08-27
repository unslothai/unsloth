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
 * So this never invents a pin out of a positive echo. It clears one only on the
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
  /** ``status.is_gguf``: only a GGUF load carries an n_ctx. */
  isGguf: boolean;
  /** No load of this tab's own is in flight (``!modelLoading``). */
  seedLoadParams: boolean;
  /** The model/variant changed underneath this tab, so nothing recorded here survives it. */
  modelChanged: boolean;
  /** This model's SAVED Context Length, or null when it saved none / has no record. */
  remembered: number | null;
}): CtxPinSeed {
  const { incoming, isGguf, seedLoadParams, modelChanged, remembered } = options;
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
  if (!modelChanged) {
    // Same model, ambiguous echo. What is in the store came from watching this
    // tab's own load, which knew what the user actually asked for; an echo that
    // an Auto reload produces just as readily is not grounds to overwrite it,
    // in either direction.
    return {};
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
