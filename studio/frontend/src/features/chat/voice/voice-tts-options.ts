// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Which of a speech model's own options voice mode offers, and what it sends.
// Free of app imports so the node test runner can load it.

import type { AudioOptionSpec, AudioOptionValues } from "../../audio/audio-options";

/**
 * Options voice mode owns or that do nothing for one short sentence:
 * - voice: the panel's speaker list chooses it;
 * - seed: voice mode pins a session voice itself (voice-pin.ts);
 * - duration_sec: one length for every sentence would stretch or cut them;
 * - text/audio chunking: only long texts are chunked, and voice mode sends a sentence at a time.
 */
export const VOICE_MODE_HIDDEN_OPTIONS: ReadonlySet<string> = new Set([
  "voice",
  "seed",
  "duration_sec",
  "text_chunk_size",
  "text_chunk_mode",
  "audio_chunk_duration_sec",
  "audio_chunk_threshold_sec",
]);

// Shown up front: what changes how the voice sounds. Everything else waits under More options.
const VOICE_MODE_PRIMARY_OPTIONS: ReadonlySet<string> = new Set([
  "speed",
  "language",
  "instruct",
  "instruction",
]);

// Most models declare speed without a range (Kokoro: "finite and positive"), which draws a number box.
// Voice mode gives it the range speech stays clear in, so it is a slider like the rest.
const VOICE_MODE_SPEED = { min: 0.5, max: 2, default: 1 };

function forVoiceMode(spec: AudioOptionSpec): AudioOptionSpec {
  if (spec.name !== "speed" || (spec.type !== "float" && spec.type !== "int")) return spec;
  return {
    ...spec,
    type: "float",
    min: spec.min ?? VOICE_MODE_SPEED.min,
    max: spec.max ?? VOICE_MODE_SPEED.max,
    default: spec.default ?? VOICE_MODE_SPEED.default,
    // The label and the multiplier beside the slider say it all; spec text reads like an API doc.
    description: null,
  };
}

/** The options the voice panel and Settings show, split into the few up front and the rest. */
export function voiceModeOptionGroups(specs: readonly AudioOptionSpec[]): {
  primary: AudioOptionSpec[];
  more: AudioOptionSpec[];
} {
  const primary: AudioOptionSpec[] = [];
  const more: AudioOptionSpec[] = [];
  for (const raw of specs) {
    if (VOICE_MODE_HIDDEN_OPTIONS.has(raw.name)) continue;
    const spec = forVoiceMode(raw);
    // A required option with no default would refuse every sentence: never tuck it away.
    const upFront =
      VOICE_MODE_PRIMARY_OPTIONS.has(spec.name) || (spec.required === true && spec.default == null);
    (upFront ? primary : more).push(spec);
  }
  return { primary, more };
}

/** Saved values to send with each sentence, without the ones voice mode owns. The server re-validates. */
export function voiceModeRequestOptions(
  values: AudioOptionValues | undefined,
): AudioOptionValues | undefined {
  if (!values) return undefined;
  const out: AudioOptionValues = {};
  for (const [name, value] of Object.entries(values)) {
    if (!VOICE_MODE_HIDDEN_OPTIONS.has(name)) out[name] = value;
  }
  return Object.keys(out).length ? out : undefined;
}

/** How many saved values differ from the model's defaults, for the "More options" count. */
export function changedOptionCount(
  specs: readonly AudioOptionSpec[],
  values: AudioOptionValues | undefined,
): number {
  if (!values) return 0;
  return specs.filter((spec) => {
    const value = values[spec.name];
    return value !== undefined && value !== spec.default;
  }).length;
}
