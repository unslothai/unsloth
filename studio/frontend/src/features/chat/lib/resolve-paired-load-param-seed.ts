// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What a fresh /api/inference/status does to one editable control / loaded-baseline pair.
// Shared by the per-model load settings group (spec mode, KV dtype, tensor parallel, draft
// count) so a same-checkpoint reload from another tab or API caller advances baselines
// without clobbering a genuinely staged edit.

export interface PairedLoadParamState<T> {
  /** The editable control: what the next load or Apply would send. */
  control: T | null;
  /** What the resident server was launched with, as this tab last saw it. */
  loaded: T | null;
}

/** The subset of the pair to write back; empty means leave the store alone. */
export type PairedLoadParamSeed<T> = Partial<PairedLoadParamState<T>>;

export function resolvePairedLoadParamSeed<T>(options: {
  /** The status echo for this field; undefined on a backend that omits it. */
  incoming: T | null | undefined;
  previous: PairedLoadParamState<T>;
  /** A different model or quant is being adopted, so both fields are stale. */
  hydratingExistingModel: boolean;
  /** No load of this tab's own is in flight (``!modelLoading``). */
  seedLoadParams: boolean;
  equals?: (a: T | null, b: T | null) => boolean;
  /** Store default for the control while ``loaded`` is still null. */
  pristineControl?: T | null;
}): PairedLoadParamSeed<T> {
  const {
    incoming,
    previous,
    hydratingExistingModel,
    seedLoadParams,
    equals = (a, b) => a === b,
    pristineControl,
  } = options;
  if (!seedLoadParams) {
    return {};
  }
  if (incoming === undefined) {
    if (hydratingExistingModel) {
      return { control: null, loaded: null };
    }
    return {};
  }
  // A null loaded baseline means this tab has not hydrated the resident server yet.
  // Non-null store defaults (tensorParallel=false, persisted speculativeType) are not
  // user edits and must adopt the status echo together with the baseline. Nullable
  // controls with a staged non-null value while loaded is null are real edits.
  const unseeded =
    previous.loaded === null &&
    (previous.control === null ||
      (pristineControl !== undefined &&
        equals(previous.control, pristineControl)));
  if (hydratingExistingModel || unseeded) {
    return { control: incoming, loaded: incoming };
  }
  if (equals(incoming, previous.loaded)) {
    return {};
  }
  const controlIsDirty = !equals(previous.control, previous.loaded);
  return {
    loaded: incoming,
    ...(controlIsDirty ? {} : { control: incoming }),
  };
}
