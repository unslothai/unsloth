// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface VisionSwitchSeedState {
  disableVision: boolean;
  loadedDisableVision: boolean | null;
  loadedVisionDisabledByUser: boolean | null;
}

/** Seed when never seeded, the model changed, or the running value moved while idle.
 *  A pending edit is detected against the OLD baseline. */
export function shouldSeedVisionSwitch(args: {
  incoming: boolean;
  previous: VisionSwitchSeedState;
  hydratingExistingModel: boolean;
}): boolean {
  const { incoming, previous, hydratingExistingModel } = args;
  if (previous.loadedVisionDisabledByUser === null) return true;
  if (hydratingExistingModel) return true;
  if (previous.loadedDisableVision === null) return false;
  if (previous.loadedDisableVision === incoming) return false;
  return previous.disableVision === previous.loadedDisableVision;
}
