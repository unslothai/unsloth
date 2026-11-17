// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** A clip the server holds (POST /audio/inputs) and what it says, to clone the voice from. */
export type VoiceReference = { inputId: string; text: string };

/**
 * Spoken once per session, never played, and cloned by every sentence after it. A filler such as
 * "Let me see." is too short to clone from: mimo heard it garbled, and the long sentences cloned
 * from it drifted to other speakers.
 */
export const VOICE_REFERENCE_TEXT =
  "Hi there. I'll keep this same voice for our whole conversation, so it's easy to follow along.";

/** A drawn reference outside this range is a broken clip, not a voice to keep. */
export const VOICE_REFERENCE_SECONDS = { min: 2, max: 20 } as const;

type Pin = { model: string; ready: Promise<VoiceReference | null> };
let pin: Pin | null = null;
let generation = 0;

/** Forget the session's voice: the next sentence draws a new one. */
export function clearVoicePin(): void {
  pin = null;
  generation += 1;
}

/** Changes whenever the session voice is dropped, so audio cached in the old voice is not reused. */
export function voicePinGeneration(): number {
  return generation;
}

/**
 * A voice with no preset speaker (OmniVoice, Breeze, MOSS-Nano, ...) draws a new speaker on every
 * request, and voice mode speaks sentence by sentence. The first sentence for a model draws one
 * reference clip and every sentence, that one included, clones it. Sentences that arrive while it
 * is drawn wait for it. A draw that fails speaks the sentence as drawn and leaves the next one to try.
 */
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
    // The clip may be gone or refused: the caller's retry draws a new voice.
    if (pin === held) pin = null;
    throw error;
  }
}

/** Seconds of audio in a PCM WAV, or null when the header is not one. */
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
