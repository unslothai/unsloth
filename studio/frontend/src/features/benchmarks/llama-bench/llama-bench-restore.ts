// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface LoadedModel {
  model: string | null | undefined;
  variant: string | null | undefined;
}

const same = (a: LoadedModel, b: LoadedModel) =>
  (a.model ?? null) === (b.model ?? null) &&
  (a.variant ?? null) === (b.variant ?? null);

/** A persisted restore belongs to one run. After a restart or a later run it would load an
 * old model over whatever the user has loaded since. */
export function ownsRestore(
  pendingJobId: string | null | undefined,
  jobId: string | null | undefined,
): boolean {
  return !!pendingJobId && pendingJobId === jobId;
}

/** Put chat's model back only while the slot is empty or still holds the benched model. */
export function shouldPutBack(
  active: LoadedModel,
  original: LoadedModel,
  benched: LoadedModel,
): boolean {
  if (!active.model) return true;
  return same(active, benched) && !same(active, original);
}
