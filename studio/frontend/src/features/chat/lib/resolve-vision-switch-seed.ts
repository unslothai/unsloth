// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The three store fields the Vision switch is decided from. */
export interface VisionSwitchSeedState {
  /** The editable control: what the next load or Apply will send. */
  disableVision: boolean;
  /** What the RUNNING server was loaded with; null before the first status. */
  loadedDisableVision: boolean | null;
  /** Set once the pair has been seeded at all; null means never. */
  loadedVisionDisabledByUser: boolean | null;
}

/**
 * Whether a status poll may write the incoming value into the editable control.
 *
 * Three cases say yes:
 *
 * 1. Never seeded, so the control holds a default nobody chose.
 * 2. The model or variant changed under this tab, so the control belongs to the
 *    model that just left.
 * 3. The RUNNING server's value moved while this tab was idle -- another tab or
 *    an API client reloaded the same model with the opposite setting. Without
 *    this, `loadedDisableVision` and the image gate follow the new server while
 *    the control keeps the old value, so Advanced Settings shows the opposite of
 *    the running projector and the next Apply quietly undoes the external change.
 *
 * The exception to case 3 is a genuinely pending local edit: a control that has
 * been moved away from its loaded baseline is the user's unapplied intent, and a
 * poll must not overwrite it. That is why "no pending edit" is `disableVision ===
 * loadedDisableVision` against the OLD baseline, not against the incoming value.
 */
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
