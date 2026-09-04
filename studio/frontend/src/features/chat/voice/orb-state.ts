// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * What the voice orb shows, decided in one place.
 *
 * Seven inputs arrive from three different sources -- the loop's own mode, the
 * chat runtime store, and the TTS player -- and several are true at once for most
 * of a turn: the model is still writing while the first sentence is already
 * playing, and the mic is live over the top of both. So the orb needs a stated
 * priority order rather than whichever effect happened to run last.
 *
 * Pure and React-free so the order is unit-testable, which matters here: the loop
 * itself cannot be exercised without a microphone, a GPU and a browser.
 */

export type VoiceOrbState =
  | "listening"
  | "transcribing"
  | "generating"
  | "synthesizing"
  | "speaking"
  | "hearing"
  | "loading";

export type VoiceOrbInputs = {
  /** The loop's own mode, not the store mirror. */
  voiceMode: "off" | "configuring" | "active";
  /** The voice slot or the transcription model is still warming up. */
  voiceSlotLoading: boolean;
  /** The microphone is picking up speech right now. */
  voiceHearing: boolean;
  /** A clip is audibly playing. */
  isPlaying: boolean;
  /** A captured utterance is being turned into text. */
  voiceTranscribing: boolean;
  /** The model is generating a reply. */
  isThreadRunning: boolean;
  /** A TTS session is open, whether or not a clip is playing this instant. */
  isSpeaking: boolean;
};

/**
 * "hold" means the caller must not change the orb yet: a TTS session is open but
 * nothing is playing, which is usually just the gap between two already
 * synthesized sentences swapping the audio element. Switching straight to
 * "synthesizing" would flicker at every sentence boundary, so the caller holds the
 * current state and only shows it if the gap persists.
 */
export type VoiceOrbDecision =
  | { kind: "set"; state: VoiceOrbState | null }
  | { kind: "hold" };

const set = (state: VoiceOrbState | null): VoiceOrbDecision => ({
  kind: "set",
  state,
});

export function deriveOrbState(inputs: VoiceOrbInputs): VoiceOrbDecision {
  // Voice is not running: the orb has nothing to say.
  if (inputs.voiceMode !== "active") return set(null);
  // A model is warming up (~35s on ROCm). Grey-blue rather than the lilac
  // "generating speech", so it does not read as a ready green, and so loading
  // does not collide with TTS synthesis on the same colour.
  if (inputs.voiceSlotLoading) return set("loading");
  // Above the speaking checks, so talking over the model (a barge-in) reads as
  // hearing; and above generating, so it shows the instant you start.
  if (inputs.voiceHearing) return set("hearing");
  // Playback wins over any background work. The model writing the rest of the
  // reply and the next sentence synthesizing both run in parallel with playback,
  // but playback is what the user is actually perceiving.
  if (inputs.isPlaying) return set("speaking");
  // Nothing playing yet, so name the real phase rather than a vague "thinking".
  // Transcribing precedes generation within a turn, so it is checked first.
  if (inputs.voiceTranscribing) return set("transcribing");
  if (inputs.isThreadRunning) return set("generating");
  if (inputs.isSpeaking) return { kind: "hold" };
  return set("listening");
}
