// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The instruction and language fields that used to sit inline in the Audio rail, unchanged.

import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import type { NativeAudioInstructionsKind } from "../audio-page-policy";
import { Field } from "../components/field";

/** The free-text instruction a model takes beside its text: a music description, a Higgs scene,
 *  a voice design, or MOSS Local style guidance. */
export function InstructionsField({
  instructionsKind,
  musicNeedsDescription,
  voiceRequired = false,
  audioInstructions,
  setAudioInstructions,
}: {
  instructionsKind: NativeAudioInstructionsKind;
  musicNeedsDescription: boolean;
  /** Maya1: the model cannot speak without a voice description. */
  voiceRequired?: boolean;
  audioInstructions: string;
  setAudioInstructions: (value: string) => void;
}) {
  return (
    <Field
      label={
        instructionsKind === "music"
          ? "Music description"
          : instructionsKind === "scene"
            ? "Scene description"
            : instructionsKind === "voice"
              ? "Voice or style description"
              : "Style instructions"
      }
      hint={
        instructionsKind === "music"
          ? musicNeedsDescription
            ? "Describe genre, tempo, mood, vocals, and arrangement. This model requires it separately from the lyrics."
            : "Optional genre, tempo, mood, vocals, and arrangement. Without it, the text above is used as the prompt."
          : instructionsKind === "scene"
            ? "Optional Higgs TTS 2 scene guidance such as room acoustics, recording conditions, or background ambience."
            : instructionsKind === "voice"
              ? voiceRequired
                ? "Required by this model. Describe the voice, accent, age and delivery."
                : "Optional. Used by Qwen3-TTS VoiceDesign, Qwen3-TTS CustomVoice and VoxCPM2; other models ignore it."
              : "Optional MOSS Local guidance such as speaking style, emotion, pace, or delivery."
      }
      htmlFor="audio-instructions"
    >
      <Textarea
        id="audio-instructions"
        value={audioInstructions}
        onChange={(event) =>
          setAudioInstructions(event.target.value)
        }
        placeholder={
          instructionsKind === "music"
            ? "Acoustic pop, 96 BPM, warm female lead, fingerpicked guitar and soft piano…"
            : instructionsKind === "scene"
              ? "Close-mic studio recording in a quiet, softly treated room…"
              : instructionsKind === "voice"
                ? "A warm, low female voice, speaking slowly and calmly…"
                : "Warm, measured delivery with a calm conversational tone…"
        }
        className="min-h-24"
      />
    </Field>
  );
}

/** MOSS Local's optional language tag. */
export function MossLanguageField({
  audioLanguage,
  setAudioLanguage,
}: {
  audioLanguage: string;
  setAudioLanguage: (value: string) => void;
}) {
  return (
    <Field
      label="Language"
      htmlFor="audio-language"
      hint="Optional, but MOSS Local v1.5 recommends a language tag when known (for example English, Arabic, or French)."
    >
      <Input
        id="audio-language"
        value={audioLanguage}
        onChange={(event) => setAudioLanguage(event.target.value)}
        placeholder="English"
      />
    </Field>
  );
}
