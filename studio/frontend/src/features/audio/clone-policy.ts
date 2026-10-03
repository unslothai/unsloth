// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What Clone needs before it can run, and the request rules its panels share. Free of app
// imports so the node test runner can load it directly.

import type { AudioSourceSelection } from "./audio-run-request";
import type { AudioModelContext, AudioRunPatch } from "./tools/types";

/** How the "What's said in the clip" field shows. */
export type ReferenceTextField = "required" | "optional" | "hidden";

/** The transcript field for the loaded model, after any panel that changes it (Timbre only,
 *  CosyVoice3 Cross-lingual). An unknown model gets an optional field. */
export function referenceTextField(
  ctx: Pick<AudioModelContext, "referenceTextMode">,
  patch: Pick<AudioRunPatch, "referenceTextMode">,
): ReferenceTextField {
  if (patch.referenceTextMode) return patch.referenceTextMode;
  switch (ctx.referenceTextMode) {
    case "required":
      return "required";
    case "unused":
      return "hidden";
    default:
      return "optional";
  }
}

/** The languages Qwen3-TTS names in full; Auto (empty) lets the model detect it. The backend
 *  maps ISO codes for families that need names, so these work for every clone model. */
export const QWEN3_LANGUAGE_NAMES = [
  "Chinese",
  "English",
  "Japanese",
  "Korean",
  "German",
  "French",
  "Russian",
  "Portuguese",
  "Spanish",
  "Italian",
] as const;

/** Three sentences to try a new voice on. */
export const CLONE_TEXT_EXAMPLES = [
  "Thanks for calling. How can I help you today?",
  "The quick brown fox jumps over the lazy dog near the river bank.",
  "Welcome back. Here is a short summary of what changed this week.",
] as const;

const SPEAKER_LINE = /^\s*Speaker\s*\d+\s*:/im;

/** VibeVoice reads a script of `Speaker N:` lines. Plain text becomes one speaker's line, the
 *  same rule the backend applies, so the preview shows what is sent. */
export function formatVibeVoiceScript(text: string): string {
  const trimmed = text.trim();
  if (!trimmed || SPEAKER_LINE.test(trimmed)) return trimmed;
  return `Speaker 1: ${trimmed}`;
}

/** IndexTTS2's eight emotion dimensions, in the order its vector takes them. */
export const INDEX_TTS2_EMOTIONS = [
  { key: "happy", label: "Happy" },
  { key: "angry", label: "Angry" },
  { key: "sad", label: "Sad" },
  { key: "afraid", label: "Afraid" },
  { key: "disgusted", label: "Disgusted" },
  { key: "melancholic", label: "Melancholic" },
  { key: "surprised", label: "Surprised" },
  { key: "calm", label: "Calm" },
] as const;

/** The vector as the runtime takes it: eight comma-separated numbers, two decimals at most. */
export function emotionVectorString(vector: readonly number[]): string {
  return INDEX_TTS2_EMOTIONS.map((_, index) => {
    const value = vector[index];
    const clamped =
      typeof value === "number" && Number.isFinite(value)
        ? Math.min(1, Math.max(0, value))
        : 0;
    return String(Number(clamped.toFixed(2)));
  }).join(",");
}

export type CloneBlockerKind =
  | "reference"
  | "reference-busy"
  | "reference-expired"
  | "reference-error"
  | "reference-text"
  | "text"
  | "panel";

export interface CloneBlockerInput {
  reference: AudioSourceSelection | null;
  /** The reference card is uploading or recording. */
  referenceBusy: boolean;
  referenceExpired: boolean;
  referenceError: string | null;
  referenceText: string;
  referenceTextField: ReferenceTextField;
  text: string;
  panelError: string | null;
}

/** The first page input Clone is missing, in rail order; null when it can run. Model blockers
 *  (none loaded, cannot clone) come from the host first. */
export function cloneBlocker(
  input: CloneBlockerInput,
): { kind: CloneBlockerKind; reason: string } | null {
  if (input.referenceExpired) {
    return {
      kind: "reference-expired",
      reason: "This reference expired. Add it again.",
    };
  }
  if (input.referenceBusy) {
    return {
      kind: "reference-busy",
      reason: "Waiting for the reference to finish uploading.",
    };
  }
  if (input.referenceError && !input.reference) {
    return { kind: "reference-error", reason: input.referenceError };
  }
  if (!input.reference) {
    return {
      kind: "reference",
      reason: "Needs a short clip of the voice to copy.",
    };
  }
  if (input.referenceTextField === "required" && !input.referenceText.trim()) {
    return {
      kind: "reference-text",
      reason: "This model needs what's said in the clip.",
    };
  }
  if (!input.text.trim()) {
    return { kind: "text", reason: "Type the text to speak." };
  }
  if (input.panelError) return { kind: "panel", reason: input.panelError };
  return null;
}
