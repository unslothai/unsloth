// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What a fresh /api/inference/status does to one batch-size control/baseline pair
// (nBatch or nUbatch). Own module like resolve-chat-template-seed.ts, importing
// neither the store nor a barrel, so the rules are testable off a browser.

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
}): BatchSizeSeed {
  const { incoming, isGguf, previous, seedLoadParams } = options;
  // while a load is in flight performLoad owns the load params
  if (!seedLoadParams) {
    return {};
  }
  // a non-gguf status has no batch flags; an absent field on a gguf is an older backend saying nothing
  const effective = isGguf ? incoming : null;
  if (effective === undefined) {
    return {};
  }
  // steady echo: an ordinary poll must not touch anything
  if (previous.loaded === effective) {
    return {};
  }
  // The baseline is a fact about the running server, so it always advances (the
  // rollback resends it). The control follows only while it still sits on the old
  // non-null baseline; a pending edit or a blank "follow default" keeps its intent.
  // A move to null (a same-model reload elsewhere dropped the override) follows the
  // same rule, so the tab does not read dirty against a server back at defaults.
  const controlIsClean =
    previous.loaded !== null && previous.value === previous.loaded;
  return {
    loaded: effective,
    ...(controlIsClean ? { value: effective } : {}),
  };
}
