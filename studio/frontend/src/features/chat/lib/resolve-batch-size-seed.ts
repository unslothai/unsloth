// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// what a fresh /api/inference/status does to one batch control/baseline pair

export interface BatchSizeSeedState {
  /** The editable control: what the next load or Apply would send. */
  value: number | null;
  /** What the resident server was invoked with, as this tab last saw it. */
  loaded: number | null;
}

/** The subset of the pair to write back; empty means leave the store alone. */
export type BatchSizeSeed = Partial<BatchSizeSeedState>;

export function resolveBatchSizeSeed(options: {
  /** ``status.requested_n_batch`` / ``_n_ubatch``; undefined on a backend that omits the field. */
  incoming: number | null | undefined;
  /** ``status.is_gguf``: a non-GGUF (or diffusion, which echoes null) never has the flag. */
  isGguf: boolean;
  previous: BatchSizeSeedState;
  /** No load of this tab's own is in flight (``!modelLoading``). */
  seedLoadParams: boolean;
  /** The model/variant changed underneath this tab. Nothing staged against the
   *  previous model may survive it, so the new model's echo is the whole truth:
   *  the control adopts it even when a pending edit would otherwise hold, and
   *  even when the new model happens to report the count the old one ran. */
  modelChanged?: boolean;
}): BatchSizeSeed {
  const {
    incoming,
    isGguf,
    previous,
    seedLoadParams,
    modelChanged = false,
  } = options;
  // while a load is in flight performLoad owns the load params
  if (!seedLoadParams) {
    return {};
  }
  // a non-gguf has no batch flags; an absent field is an older backend saying nothing
  const effective = isGguf ? incoming : null;
  if (effective === undefined) {
    // Nothing to adopt, but a control staged against the model that just left must
    // still go, or the old model's edit follows onto the new one.
    return modelChanged ? { value: null } : {};
  }
  // steady echo: an ordinary poll must not touch anything
  if (previous.loaded === effective && !modelChanged) {
    return {};
  }
  // the baseline is a fact about the running server; the control follows it only while clean
  const controlIsClean = modelChanged || previous.value === previous.loaded;
  return {
    loaded: effective,
    ...(controlIsClean ? { value: effective } : {}),
  };
}
