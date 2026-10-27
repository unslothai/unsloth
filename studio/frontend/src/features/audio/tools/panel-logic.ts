// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// JSX-free so the node test runner can load it; clone-panels.tsx / speak-panels.tsx add Components.

import type { AudioOptionSpec } from "../audio-options";
import { type AudioSourceSelection, sourceRefOf } from "../audio-run-request";
import { INDEX_TTS2_EMOTIONS, emotionVectorString } from "../clone-policy";
import type { AudioSourceStatus } from "../hooks/audio-source-state";
import type { AudioModelContext, AudioRunPatch, AudioToolPanel } from "./types";

type AudioToolPanelLogic<V> = Omit<AudioToolPanel<V>, "Component">;

const clamp = (x: number, lo: number, hi: number) =>
  Math.min(hi, Math.max(lo, x));

// A panel starts where the server would: the option's declared default, which it fills when unset.
function numberDefault(
  specs: readonly AudioOptionSpec[],
  name: string,
  fallback: number,
): number {
  const value = specs.find((spec) => spec.name === name)?.default;
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function boolDefault(
  specs: readonly AudioOptionSpec[],
  name: string,
  fallback: boolean,
): boolean {
  const value = specs.find((spec) => spec.name === name)?.default;
  return typeof value === "boolean" ? value : fallback;
}

export interface TimbreOnlyValue {
  timbreOnly: boolean;
}

export const qwen3TimbreLogic: AudioToolPanelLogic<TimbreOnlyValue> = {
  id: "qwen3-timbre",
  families: ["qwen3_tts"],
  workflows: ["clone"],
  title: "Timbre only",
  claims: ["x_vector_only_mode"],
  initial: (specs) => ({
    timbreOnly: boolDefault(specs, "x_vector_only_mode", false),
  }),
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
  initial: (specs) => ({
    mode: "text",
    text: "",
    vector: INDEX_TTS2_EMOTIONS.map(() => 0),
    alpha: numberDefault(specs, "emotion_alpha", 0.7),
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
    initial: (specs) => ({
      exaggeration: numberDefault(specs, "exaggeration", 0.5),
      guidance: numberDefault(specs, "guidance_scale", 0.5),
    }),
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
  /** The built-in voice; null leaves the model's default. */
  builtin: string | number | boolean | null;
}

export const SAVED_VOICE_MISSING =
  "Pick a saved voice, or switch Voice to Built-in.";
export const SAVED_VOICE_DELETED =
  "That saved voice was deleted. Pick another one, or switch Voice to Built-in.";
export const SAVED_VOICE_NO_TRANSCRIPT =
  "This voice has no transcript, and this model needs one. Add it on Clone, or pick another voice.";

const VOICE_OPTION_NAMES = ["voice", "voice_id", "speaker"] as const;

/** Saved voices clone, so a model that cannot clone always uses its built-in voice. */
export const usesSavedVoice = (value: SpeakVoiceValue, ctx?: AudioModelContext) =>
  value.source === "saved" && ctx?.audioWorkflows?.includes("clone") === true;

/** The option that picks a built-in voice: the first choice list among voice, voice_id and
 *  speaker, else the first of them the model declares at all. */
export function voiceOptionName(
  specs: readonly AudioOptionSpec[],
): string | null {
  const declared = VOICE_OPTION_NAMES.map((name) =>
    specs.find((spec) => spec.name === name),
  ).filter((spec) => spec !== undefined);
  return (
    (declared.find((spec) => spec.type === "enum") ?? declared[0])?.name ??
    null
  );
}

/** The kept picker value; one saved before the picker held built-in voices takes the voice
 *  chosen in Advanced, which the picker now claims. */
export function speakVoiceStored(
  stored: unknown,
  optionValues: Readonly<Record<string, unknown>>,
  voiceOption: string | null | undefined,
): unknown {
  const saved = voiceOption ? optionValues[voiceOption] : undefined;
  if (saved === undefined || saved === null) return stored;
  if (stored === undefined) return { builtin: saved };
  if (typeof stored === "object" && stored !== null && !("builtin" in stored))
    return { ...stored, builtin: saved };
  return stored;
}

export const speakVoiceLogic: AudioToolPanelLogic<SpeakVoiceValue> = {
  id: "speak-voice",
  families: [],
  workflows: ["speak"],
  title: "Voice",
  claims: VOICE_OPTION_NAMES,
  appliesTo: (ctx) =>
    Boolean(
      ctx.audioWorkflows?.includes("speak") &&
        (ctx.voiceOption || ctx.audioWorkflows.includes("clone")),
    ),
  initial: () => ({ source: "builtin", voiceId: null, builtin: null }),
  toRequest: (value, ctx) => {
    if (usesSavedVoice(value, ctx)) {
      return value.voiceId
        ? { inputs: { reference: { voice_id: value.voiceId } } }
        : {};
    }
    return ctx?.voiceOption && value.builtin !== null
      ? { options: { [ctx.voiceOption]: value.builtin } }
      : {};
  },
  validate: (value, _core, ctx) => {
    if (!usesSavedVoice(value, ctx)) return null;
    if (!value.voiceId) return SAVED_VOICE_MISSING;
    // Deleting a voice on Clone does not reach the choice kept here.
    if (ctx.savedVoiceIds && !ctx.savedVoiceIds.includes(value.voiceId))
      return SAVED_VOICE_DELETED;
    // The server falls back to the voice's transcript, so only an empty one blocks.
    return ctx.referenceTextMode === "required" &&
      ctx.voicesWithoutTranscript?.includes(value.voiceId)
      ? SAVED_VOICE_NO_TRANSCRIPT
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

// audio.cpp model_specs/supertonic.json; anything else fails its tokenizer.
export const SUPERTONIC_LANGUAGES: readonly string[] = (
  "en ko ja ar bg cs da de el es et fi fr hi hr hu id it lt lv nl pl pt ro ru " +
  "sk sl sv tr uk vi"
).split(" ");

/** Language code, "" for the model's default (English). */
export const supertonicLanguageLogic: AudioToolPanelLogic<string> = {
  id: "supertonic-language",
  families: ["supertonic"],
  workflows: ["speak"],
  title: "Language",
  claims: [],
  initial: () => "",
  toRequest: (value) =>
    typeof value === "string" && SUPERTONIC_LANGUAGES.includes(value)
      ? { language: value }
      : {},
};

export const CLONE_PANEL_LOGIC = [
  qwen3TimbreLogic,
  indexTts2EmotionLogic,
  chatterboxExpressivenessLogic,
  cosyVoiceModeLogic,
  f5SpeedDialectLogic,
] as const;
