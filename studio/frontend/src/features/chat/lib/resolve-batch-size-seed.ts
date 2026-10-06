// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// what a fresh /api/inference/status does to one load control/baseline pair

/** Generic in the value type because the rule is about the ECHO, not about batch sizes: the string
 *  tuning controls beside them follow the same steady-echo, dirty-control and model-change logic.
 *  Defaults to number, so existing call sites read unchanged. */
export interface BatchSizeSeedState<T extends number | string = number> {
  /** The editable control: what the next load or Apply would send. */
  value: T | null;
  /** What the resident server was invoked with, as this tab last saw it. */
  loaded: T | null;
}

/** The subset of the pair to write back; empty means leave the store alone. */
export type BatchSizeSeed<T extends number | string = number> = Partial<
  BatchSizeSeedState<T>
>;

export function resolveBatchSizeSeed<T extends number | string = number>(options: {
  /** ``status.requested_n_batch`` / ``_n_ubatch``; undefined on a backend that omits the field. */
  incoming: T | null | undefined;
  /** ``status.is_gguf``: a non-GGUF (or diffusion, which echoes null) never has the flag. */
  isGguf: boolean;
  previous: BatchSizeSeedState<T>;
  /** No load of this tab's own is in flight (``!modelLoading``). */
  seedLoadParams: boolean;
  /** The model/variant changed underneath this tab. Nothing staged against the previous model may
   *  survive it, so the new model's echo is the whole truth: the control adopts it even when a
   *  pending edit would otherwise hold, and even when the new model reports the old count. */
  modelChanged?: boolean;
}): BatchSizeSeed<T> {
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
    // Nothing to adopt, and nothing staged or recorded against the model that just left may survive
    // it: the baseline goes with the control, or a later rollback resends the departed model's batch
    // as if the new server were running it.
    return modelChanged ? { value: null, loaded: null } : {};
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
