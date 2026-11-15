// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface VoiceOrbOverlayPrefs {
  showVoiceArcs: boolean;
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
