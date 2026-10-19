// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Which voice Chat may unload: one it loaded itself, never one the Audio page was already serving.

export type VoiceHold = { id: string; owned: boolean };

export function voiceToUnload(hold: VoiceHold | null): string | null {
  return hold?.owned ? hold.id : null;
}

/** A voice found already serving stays owned only if this page loaded that same voice. */
export function holdForServing(hold: VoiceHold | null, id: string): VoiceHold {
  return { id, owned: hold?.id === id && hold.owned };
}

/** A load that settles after voice mode let go (epoch moved) is unloaded instead of kept. */
export function settleVoiceLoad(
  id: string,
  startEpoch: number,
  epoch: number,
): { hold: VoiceHold | null; unload: string | null } {
  return startEpoch === epoch ? { hold: { id, owned: true }, unload: null } : { hold: null, unload: id };
}
