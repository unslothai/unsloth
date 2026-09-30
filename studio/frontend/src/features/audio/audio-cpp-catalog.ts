// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Curated audio.cpp models, mirrored from the backend's audio_cpp_models.py. audio.cpp publishes
// every package as a subfolder of one Hub repo, so a Studio id is that repo plus the subfolder
// ("audio-cpp/audio.cpp-gguf/Kokoro-82M-GGUF"). Three segments never collide with an owner/name
// repo. Speech-to-text entries are also addressed by a short key, which is what the dictation
// settings store. Free of app imports so the node test runner can load it directly.

export const AUDIO_CPP_REPO = "audio-cpp/audio.cpp-gguf";

/** The audio_type the backend reports for a loaded audio.cpp speech or music model. */
export const AUDIO_CPP_TTS_AUDIO_TYPE = "audiocpp_tts";
export const AUDIO_CPP_MUSIC_AUDIO_TYPE = "audiocpp_music";
export const AUDIO_CPP_AUDIO_TYPES: ReadonlySet<string> = new Set([
  AUDIO_CPP_TTS_AUDIO_TYPE,
  AUDIO_CPP_MUSIC_AUDIO_TYPE,
]);

export type AudioCppTask = "tts" | "music" | "asr";

export interface AudioCppModel {
  /** Studio id: `${AUDIO_CPP_REPO}/<folder>`. */
  id: string;
  key: string;
  displayName: string;
  task: AudioCppTask;
  /** Download size, matching the backend's size_bytes. */
  sizeBytes: number;
  /** ASR only: the primary language codes the model transcribes. Absent = multilingual. */
  languages?: readonly string[];
  /** Phonemizes with eSpeak-ng, which upstream audio.cpp bundles are built without (backend
   *  needs_espeak). */
  needsEspeak?: boolean;
}

/** The `audio_cpp_runtime` block of /api/inference/audio/stt/status. */
export interface AudioCppRuntimeStatus {
  available: boolean;
  espeak: boolean;
  backend: string | null;
  release_tag: string | null;
}

const MB = 1024 * 1024;

const m = <const K extends string, const T extends AudioCppTask>(
  folder: string,
  key: K,
  displayName: string,
  task: T,
  sizeMb: number,
  languages?: readonly string[],
) => ({
  id: `${AUDIO_CPP_REPO}/${folder}`,
  key,
  displayName,
  task,
  sizeBytes: Math.trunc(sizeMb * MB),
  ...(languages ? { languages } : {}),
});

const espeak = <const T extends object>(model: T) => ({ ...model, needsEspeak: true as const });

const ENGLISH = ["en"] as const;

export const AUDIO_CPP_MODELS = [
  // Text to speech.
  espeak(m("Kokoro-82M-GGUF", "audiocpp-kokoro-82m", "Kokoro 82M", "tts", 180.8)),
  espeak(m("KittenTTS-GGUF", "audiocpp-kitten-tts-mini", "KittenTTS Mini 0.8", "tts", 288.2)),
  espeak(m("Piper-TTS-GGUF", "audiocpp-piper-en-us-lessac", "Piper (en-US Lessac)", "tts", 59.8)),
  espeak(m("Inflect-Micro-v2-GGUF", "audiocpp-inflect-micro-v2", "Inflect Micro v2", "tts", 68.7)),
  m("PocketTTS-GGUF/english", "audiocpp-pocket-tts-en", "PocketTTS (English)", "tts", 280.0),
  m("MOSS-TTS-Nano-100M-GGUF", "audiocpp-moss-tts-nano", "MOSS-TTS Nano 100M", "tts", 184.4),
  m("Supertonic-3-GGUF", "audiocpp-supertonic-3", "Supertonic 3", "tts", 298.3),
  m("Chatterbox-Turbo-GGUF", "audiocpp-chatterbox-turbo", "Chatterbox Turbo", "tts", 666.7),
  m("VoxCPM2-GGUF", "audiocpp-voxcpm2", "VoxCPM2 2B", "tts", 2818.1),
  m(
    "Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF",
    "audiocpp-qwen3-tts-1.7b-customvoice",
    "Qwen3-TTS 1.7B CustomVoice",
    "tts",
    2686.5,
  ),
  m(
    "Qwen3-TTS-12Hz-1.7B-VoiceDesign-GGUF",
    "audiocpp-qwen3-tts-1.7b-voicedesign",
    "Qwen3-TTS 1.7B VoiceDesign",
    "tts",
    2686.5,
  ),
  // Music generation.
  m("ACE-Step1.5-GGUF/turbo", "audiocpp-ace-step-1.5-turbo", "ACE-Step 1.5 Turbo", "music", 5898.9),
  m("ACE-Step1.5-GGUF/base", "audiocpp-ace-step-1.5-base", "ACE-Step 1.5 Base", "music", 5898.9),
  m(
    "Stable-Audio-3-Small-Music-GGUF",
    "audiocpp-stable-audio-3-small-music",
    "Stable Audio 3 Small (Music)",
    "music",
    1605.6,
  ),
  // Speech to text.
  m("Qwen3-ASR-0.6B-GGUF", "audiocpp-qwen3-asr-0.6b", "Qwen3-ASR 0.6B", "asr", 1097.9),
  m("Qwen3-ASR-1.7B-GGUF", "audiocpp-qwen3-asr-1.7b", "Qwen3-ASR 1.7B", "asr", 2358.4),
  m(
    "Parakeet-TDT-0.6B-v3-GGUF",
    "audiocpp-parakeet-tdt-0.6b-v3",
    "Parakeet TDT 0.6B v3",
    "asr",
    873.3,
  ),
  m("Canary-180M-Flash-GGUF", "audiocpp-canary-180m-flash", "Canary 180M Flash", "asr", 237.9, [
    "en",
    "de",
    "es",
    "fr",
  ]),
  m(
    "Moonshine-Streaming-GGUF/tiny",
    "audiocpp-moonshine-tiny",
    "Moonshine Tiny",
    "asr",
    57.6,
    ENGLISH,
  ),
  m(
    "Moonshine-Streaming-GGUF/small",
    "audiocpp-moonshine-small",
    "Moonshine Small",
    "asr",
    286.7,
    ENGLISH,
  ),
  m(
    "Nemotron-3.5-ASR-Streaming-0.6B-GGUF",
    "audiocpp-nemotron-3.5-asr-0.6b",
    "Nemotron 3.5 ASR 0.6B",
    "asr",
    887.5,
    ENGLISH,
  ),
] as const satisfies readonly AudioCppModel[];

export type AudioCppSttKey = Extract<
  (typeof AUDIO_CPP_MODELS)[number],
  { task: "asr" }
>["key"];

const BY_ID = new Map<string, AudioCppModel>(
  AUDIO_CPP_MODELS.map((model) => [model.id.toLowerCase(), model]),
);
const BY_KEY = new Map<string, AudioCppModel>(
  AUDIO_CPP_MODELS.map((model) => [model.key, model]),
);

/** Whether `id` is a curated audio.cpp Studio id. Its folder names end in "-GGUF", but only
 *  audiocpp_server reads them: every llama.cpp GGUF heuristic has to skip these. */
export function isAudioCppModelId(id: string | null | undefined): boolean {
  const text = id?.trim().toLowerCase().replace(/\/+$/, "");
  return Boolean(text && BY_ID.has(text));
}

/** The curated model for a Studio id or short key, else null. */
export function audioCppModelFor(
  identifier: string | null | undefined,
): AudioCppModel | null {
  const text = identifier?.trim();
  if (!text) return null;
  return (
    BY_KEY.get(text) ?? BY_ID.get(text.toLowerCase().replace(/\/+$/, "")) ?? null
  );
}

export function audioCppModelsForTask(task: AudioCppTask): AudioCppModel[] {
  return AUDIO_CPP_MODELS.filter((model) => model.task === task);
}

export const AUDIO_CPP_STT_KEYS = AUDIO_CPP_MODELS.filter(
  (model): model is Extract<(typeof AUDIO_CPP_MODELS)[number], { task: "asr" }> =>
    model.task === "asr",
).map((model) => model.key);

/** Music length the backend generates, whatever max_tokens asks for (audio_cpp_backend.py). The
 *  frame rate is the MiniMax Music 3 convention the Audio page already sends. */
export const AUDIO_CPP_MUSIC_MIN_SECONDS = 5;
export const AUDIO_CPP_MUSIC_MAX_SECONDS = 240;

/** "181 MB" / "2.8 GB", in the style of the dictation model sizes. */
export function audioCppSizeLabel(sizeBytes: number): string {
  const mb = sizeBytes / MB;
  return mb < 1000 ? `${Math.round(mb)} MB` : `${(mb / 1024).toFixed(1)} GB`;
}
