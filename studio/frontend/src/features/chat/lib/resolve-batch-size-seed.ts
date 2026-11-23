// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Generic because the rule is about the echo; string tuning controls share it. */
export interface BatchSizeSeedState<T extends number | string = number> {
  value: T | null;
  loaded: T | null;
}

export type BatchSizeSeed<T extends number | string = number> = Partial<
  BatchSizeSeedState<T>
>;

export function resolveBatchSizeSeed<T extends number | string = number>(options: {
  incoming: T | null | undefined;
  isGguf: boolean;
  previous: BatchSizeSeedState<T>;
  seedLoadParams: boolean;
  /** On a model change the new echo wins even over a pending edit. */
  modelChanged?: boolean;
}): BatchSizeSeed<T> {
  const {
    incoming,
    isGguf,
    previous,
    seedLoadParams,
    modelChanged = false,
  } = options;
  if (!seedLoadParams) {
    return {};
  }
  const effective = isGguf ? incoming : null;
  if (effective === undefined) {
    // Clear the baseline too, or a rollback resends the departed model's batch.
    return modelChanged ? { value: null, loaded: null } : {};
  }
  if (previous.loaded === effective && !modelChanged) {
    return {};
  }
  const controlIsClean = modelChanged || previous.value === previous.loaded;
  return {
    loaded: effective,
    ...(controlIsClean ? { value: effective } : {}),
  };
}
