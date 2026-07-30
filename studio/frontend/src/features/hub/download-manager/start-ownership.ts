// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type JobStartOwnership = "started" | "existing" | "inactive";

export function ownershipFromStartResult(result: {
  created?: boolean;
}): Exclude<JobStartOwnership, "inactive"> {
  // Cancellation ownership requires affirmative evidence that this caller
  // created the backend job. Older backends omit this field, so fail closed.
  return result.created === true ? "started" : "existing";
}
