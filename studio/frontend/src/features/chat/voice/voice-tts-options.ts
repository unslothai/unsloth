// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { AudioOptionSpec, AudioOptionValues } from "../../audio/audio-options";

export const VOICE_MODE_HIDDEN_OPTIONS: ReadonlySet<string> = new Set([
  "voice",
  "seed",
  "duration_sec",
  "text_chunk_size",
  "text_chunk_mode",
  "audio_chunk_duration_sec",
  "audio_chunk_threshold_sec",
]);

const VOICE_MODE_PRIMARY_OPTIONS: ReadonlySet<string> = new Set([
  "speed",
  "language",
  "instruct",
  "instruction",
]);

// Speed often has no declared range; this one makes it a slider.
const VOICE_MODE_SPEED = { min: 0.5, max: 2, default: 1 };

function forVoiceMode(spec: AudioOptionSpec): AudioOptionSpec {
  if (spec.name !== "speed" || (spec.type !== "float" && spec.type !== "int")) return spec;
  return {
    ...spec,
    type: "float",
    min: spec.min ?? VOICE_MODE_SPEED.min,
    max: spec.max ?? VOICE_MODE_SPEED.max,
    default: spec.default ?? VOICE_MODE_SPEED.default,
    description: null,
  };
}

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
