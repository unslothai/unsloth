// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** What the full voice orb draws around itself. Both default off. */
export interface VoiceOrbOverlayPrefs {
  /** The turn / filler / interrupt arcs round the orb. */
  showVoiceArcs: boolean;
  /** The decision chip under the caption ("VAD kept", "Filler ..."). */
  showVoiceCaption: boolean;
}

export const DEFAULT_VOICE_ORB_OVERLAY_PREFS: VoiceOrbOverlayPrefs = {
  showVoiceArcs: false,
  showVoiceCaption: false,
};

export function readVoiceOrbOverlayPrefs(saved: unknown): VoiceOrbOverlayPrefs {
  const s = (saved ?? {}) as Partial<Record<keyof VoiceOrbOverlayPrefs, unknown>>;
  return {
    showVoiceArcs: typeof s.showVoiceArcs === "boolean" ? s.showVoiceArcs : false,
    showVoiceCaption: typeof s.showVoiceCaption === "boolean" ? s.showVoiceCaption : false,
  };
}
