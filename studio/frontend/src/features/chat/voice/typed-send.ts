// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Whether a composer send in voice mode carries typed words. Enter while the loop
 * listens normally just ends the spoken turn, but the loop only ever delivers what
 * it heard, so text typed into the box (or anything at all while paused, with the
 * mic muted) was dropped when the session ended empty and the re-arm cleared the box.
 */
export function isTypedVoiceSubmit(opts: {
  /** The composer's text at the send. */
  box: string;
  /** What the loop itself put in the box: parked turns and the held fragment. */
  shown: string;
  paused: boolean;
}): boolean {
  const box = opts.box.trim();
  if (!box) return false;
  if (opts.paused) return true;
  return box !== opts.shown.trim();
}
