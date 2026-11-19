// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type VoiceReference = { inputId: string; text: string };

/** Spoken once per session, never played. Long on purpose: short fillers clone badly. */
export const VOICE_REFERENCE_TEXT =
  "Hi there. I'll keep this same voice for our whole conversation, so it's easy to follow along.";

/** Seconds: a drawn clip outside this range is broken. */
export const VOICE_REFERENCE_SECONDS = { min: 2, max: 20 } as const;

type Pin = { model: string; ready: Promise<VoiceReference | null> };
let pin: Pin | null = null;
let generation = 0;

export function clearVoicePin(): void {
  pin = null;
  generation += 1;
}

/** Bumps when the session voice drops, so audio cached in the old voice is not reused. */
export function voicePinGeneration(): number {
  return generation;
}

/** Voices with no preset speaker change on every request, so each sentence clones one clip; a failed draw speaks unpinned. */
export async function withVoicePin<T>(
  model: string,
  speak: (reference: VoiceReference | null) => Promise<T>,
  draw: () => Promise<VoiceReference | null>,
): Promise<T> {
  let held = pin?.model === model ? pin : null;
  if (!held) {
    held = { model, ready: draw().catch(() => null) };
    pin = held;
  }
  const reference = await held.ready;
  if (!reference) {
    if (pin === held) pin = null;
    return speak(null);
  }
  try {
    return await speak(reference);
  } catch (error) {
    // A refused clip is dropped, so the retry draws a new one.
    if (pin === held) pin = null;
    throw error;
  }
}

export function wavSeconds(bytes: ArrayBuffer): number | null {
  const view = new DataView(bytes);
  const tag = (at: number) =>
    String.fromCharCode(
      view.getUint8(at),
      view.getUint8(at + 1),
      view.getUint8(at + 2),
      view.getUint8(at + 3),
    );
  if (bytes.byteLength < 44 || tag(0) !== "RIFF" || tag(8) !== "WAVE")
    return null;
  let byteRate = 0;
  for (let at = 12; at + 8 <= bytes.byteLength;) {
    const id = tag(at);
    const size = view.getUint32(at + 4, true);
    if (id === "fmt ") byteRate = view.getUint32(at + 16, true);
    if (id === "data") {
      if (!byteRate) return null;
      return Math.min(size, bytes.byteLength - at - 8) / byteRate;
    }
    at += 8 + size + (size % 2);
  }
  return null;
}
