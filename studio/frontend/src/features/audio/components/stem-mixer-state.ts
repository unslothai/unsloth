// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Free of app imports so the node test runner can load it.

export interface StemMix {
  volume: number;
  muted: boolean;
  solo: boolean;
}

export interface StemMixerState {
  byStem: Readonly<Record<string, StemMix>>;
}

export type StemMixerAction =
  | { type: "toggleSolo"; role: string }
  | { type: "toggleMute"; role: string }
  | { type: "setVolume"; role: string; volume: number }
  | { type: "reset" };

export const DEFAULT_STEM_MIX: StemMix = {
  volume: 1,
  muted: false,
  solo: false,
};

export const INITIAL_STEM_MIXER_STATE: StemMixerState = { byStem: {} };

export function stemMix(state: StemMixerState, role: string): StemMix {
  return state.byStem[role] ?? DEFAULT_STEM_MIX;
}

export function anySolo(state: StemMixerState): boolean {
  return Object.values(state.byStem).some((mix) => mix.solo);
}

export function clampVolume(volume: number, fallback = 1): number {
  if (!Number.isFinite(volume)) return fallback;
  return Math.min(1, Math.max(0, volume));
}

function withStem(
  state: StemMixerState,
  role: string,
  patch: Partial<StemMix>,
): StemMixerState {
  const current = stemMix(state, role);
  const next = { ...current, ...patch };
  if (
    next.volume === current.volume &&
    next.muted === current.muted &&
    next.solo === current.solo
  )
    return state;
  return { byStem: { ...state.byStem, [role]: next } };
}

export function stemMixerReducer(
  state: StemMixerState,
  action: StemMixerAction,
): StemMixerState {
  switch (action.type) {
    case "toggleSolo":
      return withStem(state, action.role, {
        solo: !stemMix(state, action.role).solo,
      });
    case "toggleMute":
      return withStem(state, action.role, {
        muted: !stemMix(state, action.role).muted,
      });
    case "setVolume": {
      const current = stemMix(state, action.role).volume;
      return withStem(state, action.role, {
        volume: clampVolume(action.volume, current),
      });
    }
    case "reset":
      return Object.keys(state.byStem).length === 0
        ? state
        : INITIAL_STEM_MIXER_STATE;
    default:
      return state;
  }
}

/** Solo beats mute: while any stem is soloed only soloed stems sound. */
export function effectiveGain(state: StemMixerState, role: string): number {
  const mix = stemMix(state, role);
  if (anySolo(state)) return mix.solo ? mix.volume : 0;
  return mix.muted ? 0 : mix.volume;
}
