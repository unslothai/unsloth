// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// No app imports: the node test runner loads this directly.

import type { AudioSourceSelection } from "./audio-run-request";
import type { AudioModelContext, AudioRunPatch } from "./tools/types";

type ReferenceTextField = "required" | "optional" | "hidden";

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

/** Qwen3-TTS spellings; the backend maps them for other clone families. */
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

export const CLONE_TEXT_EXAMPLES = [
  "Thanks for calling. How can I help you today?",
  "The quick brown fox jumps over the lazy dog near the river bank.",
  "Welcome back. Here is a short summary of what changed this week.",
] as const;

const SPEAKER_LINE = /^\s*Speaker\s*\d+\s*:/im;

export function formatVibeVoiceScript(text: string): string {
  const trimmed = text.trim();
  if (!trimmed || SPEAKER_LINE.test(trimmed)) return trimmed;
  return `Speaker 1: ${trimmed}`;
}

/** Order matters: it is the runtime's vector order. */
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

type CloneBlockerKind =
  | "reference"
  | "reference-busy"
  | "reference-expired"
  | "reference-error"
  | "reference-text"
  | "text"
  | "panel";

interface CloneBlockerInput {
  reference: AudioSourceSelection | null;
  referenceBusy: boolean;
  referenceExpired: boolean;
  referenceError: string | null;
  referenceText: string;
  referenceTextField: ReferenceTextField;
  text: string;
  panelError: string | null;
}

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
  // A failed replacement hides the kept clip, so Generate waits until it is dismissed.
  if (input.referenceError) {
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
