// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// JSX-free so the node test runner can load it; clone-panels.tsx / speak-panels.tsx add Components.

import { type AudioSourceSelection, sourceRefOf } from "../audio-run-request";
import { INDEX_TTS2_EMOTIONS, emotionVectorString } from "../clone-policy";
import type { AudioSourceStatus } from "../hooks/audio-source-state";
import type { AudioRunPatch, AudioToolPanel } from "./types";

type AudioToolPanelLogic<V> = Omit<AudioToolPanel<V>, "Component">;

const clamp = (x: number, lo: number, hi: number) =>
  Math.min(hi, Math.max(lo, x));

export interface TimbreOnlyValue {
  timbreOnly: boolean;
}

export const qwen3TimbreLogic: AudioToolPanelLogic<TimbreOnlyValue> = {
  id: "qwen3-timbre",
  families: ["qwen3_tts"],
  workflows: ["clone"],
  title: "Timbre only",
  claims: ["x_vector_only_mode"],
  initial: () => ({ timbreOnly: false }),
  toRequest: (value) =>
    value.timbreOnly
      ? {
          options: { x_vector_only_mode: true },
          referenceTextMode: "hidden",
        }
      : {},
};

export type EmotionMode = "text" | "audio" | "mixer";

export interface EmotionValue {
  mode: EmotionMode;
  text: string;
  vector: number[];
  alpha: number;
  source: AudioSourceSelection | null;
  /** Why the emotion clip cannot be sent yet (uploading, failed, expired); set by its input. */
  sourceProblem?: string | null;
}

export function emotionSourceProblem(status: AudioSourceStatus): string | null {
  if (status.phase === "uploading" || status.phase === "recording") {
    return "Waiting for the emotion clip to finish uploading.";
  }
  if (status.phase === "error") return status.message;
  if (status.phase === "expired") return "The emotion clip expired. Add it again.";
  return null;
}

export const EMOTION_AUDIO_MISSING =
  "Add the clip whose emotion to copy, or switch Emotion to From text.";

export const indexTts2EmotionLogic: AudioToolPanelLogic<EmotionValue> = {
  id: "index-emotion",
  families: ["index_tts2"],
  workflows: ["clone"],
  title: "Emotion",
  claims: [
    "emotion_vector",
    "emotion_alpha",
    "use_emotion_text",
    "emotion_text",
  ],
  initial: () => ({
    mode: "text",
    text: "",
    vector: INDEX_TTS2_EMOTIONS.map(() => 0),
    alpha: 0.7,
    source: null,
  }),
  toRequest: (value): AudioRunPatch => {
    const alpha = clamp(value.alpha, 0, 1);
    if (value.mode === "text") {
      const text = value.text.trim();
      return text
        ? {
            options: {
              use_emotion_text: true,
              emotion_text: text,
              emotion_alpha: alpha,
            },
          }
        : {};
    }
    if (value.mode === "audio") {
      return value.source
        ? {
            inputs: { emotion: sourceRefOf(value.source) },
            options: { emotion_alpha: alpha },
          }
        : {};
    }
    return value.vector.some((weight) => weight > 0)
      ? {
          options: {
            emotion_vector: emotionVectorString(value.vector),
            emotion_alpha: alpha,
          },
        }
      : {};
  },
  validate: (value) =>
    value.mode !== "audio"
      ? null
      : (value.sourceProblem ?? (value.source ? null : EMOTION_AUDIO_MISSING)),
};

export interface ExpressivenessValue {
  exaggeration: number;
  guidance: number;
}

export const chatterboxExpressivenessLogic: AudioToolPanelLogic<ExpressivenessValue> =
  {
    id: "chatterbox-expressiveness",
    families: ["chatterbox"],
    workflows: ["clone"],
    title: "Expressiveness",
    claims: ["exaggeration", "guidance_scale"],
    initial: () => ({ exaggeration: 0.5, guidance: 0.5 }),
    toRequest: (value) => ({
      options: {
        exaggeration: clamp(value.exaggeration, 0, 2),
        guidance_scale: clamp(value.guidance, 0, 5),
      },
    }),
  };

export type CosyVoiceMode = "zero_shot" | "cross_lingual" | "instruct";

export interface CosyVoiceModeValue {
  mode: CosyVoiceMode;
  instruction: string;
}

export const COSYVOICE_INSTRUCTION_MISSING =
  "Write the instruction for Instruct, or pick another mode.";

export const cosyVoiceModeLogic: AudioToolPanelLogic<CosyVoiceModeValue> = {
  id: "cosyvoice-mode",
  families: ["cosyvoice3"],
  workflows: ["clone"],
  title: "Mode",
  claims: ["template_name", "instruction"],
  initial: () => ({ mode: "zero_shot", instruction: "" }),
  toRequest: (value) => {
    if (value.mode === "cross_lingual") {
      return {
        options: { template_name: "cross_lingual" },
        referenceTextMode: "hidden",
      };
    }
    if (value.mode === "instruct") {
      const instruction = value.instruction.trim();
      return {
        options: { template_name: "instruct" },
        ...(instruction ? { instructions: instruction } : {}),
        referenceTextMode: "optional",
      };
    }
    return { options: { template_name: "zero_shot" } };
  },
  validate: (value) =>
    value.mode === "instruct" && !value.instruction.trim()
      ? COSYVOICE_INSTRUCTION_MISSING
      : null,
};

export interface SpeedDialectValue {
  speed: number;
  dialect: string;
}

export const F5_DIALECTS: readonly { value: string; label: string }[] = [
  { value: "UNK", label: "Auto" },
  { value: "MSA", label: "Modern Standard Arabic" },
  { value: "SAU", label: "Saudi" },
  { value: "UAE", label: "Emirati" },
  { value: "ALG", label: "Algerian" },
  { value: "IRQ", label: "Iraqi" },
  { value: "EGY", label: "Egyptian" },
  { value: "MAR", label: "Moroccan" },
  { value: "OMN", label: "Omani" },
  { value: "TUN", label: "Tunisian" },
  { value: "LEV", label: "Levantine" },
  { value: "SDN", label: "Sudanese" },
  { value: "LBY", label: "Libyan" },
];

export const f5SpeedDialectLogic: AudioToolPanelLogic<SpeedDialectValue> = {
  id: "f5-speed-dialect",
  families: ["f5_tts"],
  workflows: ["clone"],
  title: "Speed and dialect",
  claims: ["speed", "dialect"],
  initial: () => ({ speed: 1, dialect: "UNK" }),
  toRequest: (value) => ({
    ...(Math.abs(value.speed - 1) > 1e-6
      ? { speed: clamp(value.speed, 0.5, 2) }
      : {}),
    ...(value.dialect && value.dialect !== "UNK"
      ? { options: { dialect: value.dialect } }
      : {}),
  }),
};

export interface SpeakVoiceValue {
  source: "builtin" | "saved";
  voiceId: string | null;
}

export const SAVED_VOICE_MISSING =
  "Pick a saved voice, or switch Voice to Built-in.";
export const SAVED_VOICE_DELETED =
  "That saved voice was deleted. Pick another one, or switch Voice to Built-in.";

export const speakVoiceLogic: AudioToolPanelLogic<SpeakVoiceValue> = {
  id: "speak-voice",
  families: [],
  workflows: ["speak"],
  title: "Voice",
  claims: [],
  appliesTo: (ctx) =>
    Boolean(
      ctx.audioWorkflows?.includes("speak") &&
        ctx.audioWorkflows.includes("clone"),
    ),
  initial: () => ({ source: "builtin", voiceId: null }),
  toRequest: (value) =>
    value.source === "saved" && value.voiceId
      ? { inputs: { reference: { voice_id: value.voiceId } } }
      : {},
  validate: (value, _core, ctx) => {
    if (value.source !== "saved") return null;
    if (!value.voiceId) return SAVED_VOICE_MISSING;
    // Deleting a voice on Clone does not reach the choice kept here.
    return ctx.savedVoiceIds && !ctx.savedVoiceIds.includes(value.voiceId)
      ? SAVED_VOICE_DELETED
      : null;
  },
};

// The backend formats plain text into `Speaker N:` lines, so nothing is sent.
export const vibeVoiceDialogueLogic: AudioToolPanelLogic<null> = {
  id: "vibevoice-dialogue",
  families: ["vibevoice"],
  workflows: ["speak"],
  title: "Dialogue",
  claims: [],
  initial: () => null,
  toRequest: () => ({}),
};

export const CLONE_PANEL_LOGIC = [
  qwen3TimbreLogic,
  indexTts2EmotionLogic,
  chatterboxExpressivenessLogic,
  cosyVoiceModeLogic,
  f5SpeedDialectLogic,
] as const;
