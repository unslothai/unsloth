// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What each Clone and Speak model tool sends and when it holds Generate back, without its
// controls. Free of JSX so the node test runner can load it; clone-panels.tsx and
// speak-panels.tsx add the Components.

import { type AudioSourceSelection, sourceRefOf } from "../audio-run-request";
import { INDEX_TTS2_EMOTIONS, emotionVectorString } from "../clone-policy";
import type { AudioRunPatch, AudioToolPanel } from "./types";

export type AudioToolPanelLogic<V> = Omit<AudioToolPanel<V>, "Component">;

// ---- Qwen3-TTS Base: Timbre only -------------------------------------------------------------

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

// ---- IndexTTS2: Emotion ----------------------------------------------------------------------

export type EmotionMode = "text" | "audio" | "mixer";

export interface EmotionValue {
  mode: EmotionMode;
  text: string;
  /** Eight weights 0..1 in INDEX_TTS2_EMOTIONS order. */
  vector: number[];
  /** How strongly the emotion applies, 0..1. */
  alpha: number;
  source: AudioSourceSelection | null;
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
    const alpha = Math.min(1, Math.max(0, value.alpha));
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
    value.mode === "audio" && !value.source ? EMOTION_AUDIO_MISSING : null,
};

// ---- Chatterbox: Expressiveness --------------------------------------------------------------

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
        exaggeration: Math.min(2, Math.max(0, value.exaggeration)),
        guidance_scale: Math.min(5, Math.max(0, value.guidance)),
      },
    }),
  };

// ---- CosyVoice3: Mode ------------------------------------------------------------------------

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
      // Cross-lingual reads only the voice, not the clip's words.
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

// ---- F5-TTS: Speed and dialect ---------------------------------------------------------------

export interface SpeedDialectValue {
  speed: number;
  dialect: string;
}

/** Habibi's dialect tokens with their names; UNK lets the model decide. */
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
      ? { speed: Math.min(2, Math.max(0.5, value.speed)) }
      : {}),
    ...(value.dialect && value.dialect !== "UNK"
      ? { options: { dialect: value.dialect } }
      : {}),
  }),
};

// ---- Speak: saved voice, VibeVoice dialogue --------------------------------------------------

export interface SpeakVoiceValue {
  source: "builtin" | "saved";
  voiceId: string | null;
}

export const SAVED_VOICE_MISSING =
  "Pick a saved voice, or switch Voice to Built-in.";

/** Models that both speak and clone (VoxCPM2) can read the text in a saved voice. */
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
  validate: (value) =>
    value.source === "saved" && !value.voiceId ? SAVED_VOICE_MISSING : null,
};

/** VibeVoice reads `Speaker N:` lines; the backend formats plain text, so nothing is sent. */
export const vibeVoiceDialogueLogic: AudioToolPanelLogic<null> = {
  id: "vibevoice-dialogue",
  families: ["vibevoice"],
  workflows: ["speak"],
  title: "Dialogue",
  claims: [],
  initial: () => null,
  toRequest: () => ({}),
};

/** Every panel's logic in rail order, mirroring AUDIO_TOOL_PANELS without the instruction fields. */
export const CLONE_PANEL_LOGIC = [
  qwen3TimbreLogic,
  indexTts2EmotionLogic,
  chatterboxExpressivenessLogic,
  cosyVoiceModeLogic,
  f5SpeedDialectLogic,
] as const;
