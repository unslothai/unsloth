// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Seeds the Audio pickers and dictation settings; any GGUF the backend recognises works too.
// Free of app imports so the node test runner can load it directly.

export const AUDIO_CPP_REPO = "audio-cpp/audio.cpp-gguf";

export const AUDIO_CPP_TTS_AUDIO_TYPE = "audiocpp_tts";
export const AUDIO_CPP_MUSIC_AUDIO_TYPE = "audiocpp_music";
export const AUDIO_CPP_AUDIO_TYPES: ReadonlySet<string> = new Set([
  AUDIO_CPP_TTS_AUDIO_TYPE,
  AUDIO_CPP_MUSIC_AUDIO_TYPE,
]);

export type AudioCppTask = "tts" | "music" | "asr";

/** Mirrors AudioWorkflowId; spelled out to keep this file import-free. */
export type AudioCppWorkflow = "speak" | "clone" | "edit" | "music" | "transcribe";

export interface AudioCppModel {
  /** Hub repo id, or `${AUDIO_CPP_REPO}/<folder>` for a package in the shared repo. */
  id: string;
  task: AudioCppTask;
  workflows?: readonly AudioCppWorkflow[];
  /** ASR only: the primary language codes the model transcribes. Absent = multilingual. */
  languages?: readonly string[];
  /** Phonemizes with eSpeak-ng, which upstream runtime bundles lack (backend needs_espeak). */
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

const folder = (name: string) => `${AUDIO_CPP_REPO}/${name}`;
const ENGLISH = ["en"] as const;

export const AUDIO_CPP_MODELS: readonly AudioCppModel[] = [
  { id: folder("Kokoro-82M-GGUF"), task: "tts", needsEspeak: true },
  { id: folder("KittenTTS-GGUF"), task: "tts", needsEspeak: true },
  { id: folder("Piper-TTS-GGUF"), task: "tts", needsEspeak: true },
  { id: folder("Inflect-Micro-v2-GGUF"), task: "tts", needsEspeak: true },
  { id: folder("PocketTTS-GGUF"), task: "tts" },
  { id: folder("MOSS-TTS-Nano-100M-GGUF"), task: "tts" },
  { id: folder("Supertonic-3-GGUF"), task: "tts" },
  { id: folder("Chatterbox-Turbo-GGUF"), task: "tts" },
  { id: folder("VoxCPM2-GGUF"), task: "tts", workflows: ["speak", "clone"] },
  { id: folder("Qwen3-TTS-12Hz-0.6B-Base-GGUF"), task: "tts", workflows: ["clone"] },
  { id: folder("Chatterbox-GGUF"), task: "tts", workflows: ["clone"] },
  { id: folder("IndexTTS2-GGUF"), task: "tts", workflows: ["clone"] },
  { id: folder("CosyVoice3-GGUF"), task: "tts", workflows: ["clone"] },
  { id: folder("Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF"), task: "tts" },
  { id: folder("Qwen3-TTS-12Hz-1.7B-VoiceDesign-GGUF"), task: "tts" },
  { id: folder("DotTTS-Edit-GGUF"), task: "tts", workflows: ["speak", "edit"] },
  { id: folder("Vevo2-GGUF"), task: "tts", workflows: ["clone", "edit"] },
  { id: folder("FireRedAudio-GGUF"), task: "tts", workflows: ["clone", "edit"] },
  { id: "audio-cpp/MiniMax-Music3-GGUF", task: "music" },
  { id: "audio-cpp/Yue2-3B-GGUF", task: "music" },
  { id: folder("ACE-Step1.5-GGUF"), task: "music" },
  { id: folder("Stable-Audio-3-Small-Music-GGUF"), task: "music" },
  { id: folder("Qwen3-ASR-0.6B-GGUF"), task: "asr" },
  { id: folder("Qwen3-ASR-1.7B-GGUF"), task: "asr" },
  { id: folder("Parakeet-TDT-0.6B-v3-GGUF"), task: "asr" },
  { id: folder("Canary-180M-Flash-GGUF"), task: "asr", languages: ["en", "de", "es", "fr"] },
  { id: folder("Moonshine-Streaming-GGUF"), task: "asr", languages: ENGLISH },
  { id: folder("Nemotron-3.5-ASR-Streaming-0.6B-GGUF"), task: "asr", languages: ENGLISH },
];

/** Legacy Settings > Voice keys; the backend still maps each to its folder id (and variant). */
export interface AudioCppDictationModel {
  key: string;
  id: string;
  /** The folder's sub-package when it ships several (Moonshine sizes). */
  variant?: string;
  sizeBytes: number;
}

const dictation = <const K extends string>(
  key: K,
  name: string,
  sizeMb: number,
  variant?: string,
): AudioCppDictationModel & { key: K } => ({
  key,
  id: folder(name),
  sizeBytes: Math.trunc(sizeMb * MB),
  ...(variant ? { variant } : {}),
});

export const AUDIO_CPP_DICTATION_MODELS = [
  dictation("audiocpp-qwen3-asr-0.6b", "Qwen3-ASR-0.6B-GGUF", 1097.9),
  dictation("audiocpp-qwen3-asr-1.7b", "Qwen3-ASR-1.7B-GGUF", 2358.4),
  dictation("audiocpp-parakeet-tdt-0.6b-v3", "Parakeet-TDT-0.6B-v3-GGUF", 873.3),
  dictation("audiocpp-canary-180m-flash", "Canary-180M-Flash-GGUF", 237.9),
  dictation("audiocpp-moonshine-tiny", "Moonshine-Streaming-GGUF", 57.6, "tiny"),
  dictation("audiocpp-moonshine-small", "Moonshine-Streaming-GGUF", 286.7, "small"),
  dictation(
    "audiocpp-nemotron-3.5-asr-0.6b",
    "Nemotron-3.5-ASR-Streaming-0.6B-GGUF",
    887.5,
  ),
] as const satisfies readonly AudioCppDictationModel[];

export type AudioCppSttKey = (typeof AUDIO_CPP_DICTATION_MODELS)[number]["key"];

export const AUDIO_CPP_STT_KEYS: readonly AudioCppSttKey[] = AUDIO_CPP_DICTATION_MODELS.map(
  (model) => model.key,
);

function normalizedId(id: string | null | undefined): string {
  return id?.trim().toLowerCase().replace(/\/+$/, "") ?? "";
}

const BY_ID = new Map<string, AudioCppModel>(
  AUDIO_CPP_MODELS.map((model) => [model.id.toLowerCase(), model]),
);
const DICTATION_BY_KEY = new Map<string, AudioCppDictationModel>(
  AUDIO_CPP_DICTATION_MODELS.map((model) => [model.key, model]),
);

export function isAudioCppFolderId(id: string | null | undefined): boolean {
  const text = normalizedId(id);
  return (
    text.startsWith(`${AUDIO_CPP_REPO.toLowerCase()}/`) &&
    text.length > AUDIO_CPP_REPO.length + 1
  );
}

export function audioCppDisplayName(id: string): string {
  const trimmed = id.trim().replace(/\/+$/, "");
  if (isAudioCppFolderId(trimmed)) {
    return trimmed.slice(AUDIO_CPP_REPO.length + 1).split("/")[0] || trimmed;
  }
  return trimmed.split("/").pop() || trimmed;
}

export function audioCppModelFor(id: string | null | undefined): AudioCppModel | null {
  return BY_ID.get(normalizedId(id)) ?? null;
}

export function audioCppDictationModelFor(
  key: string | null | undefined,
): AudioCppDictationModel | null {
  const text = key?.trim();
  return text ? (DICTATION_BY_KEY.get(text) ?? null) : null;
}

/** Served by the audio runtime rather than llama-server, as far as the picker can tell. */
export function isAudioRuntimeGguf(
  id: string | null | undefined,
  audioType?: string | null,
): boolean {
  return (
    AUDIO_CPP_AUDIO_TYPES.has(audioType ?? "") ||
    audioCppModelFor(id) !== null ||
    isAudioCppFolderId(id)
  );
}

const TASK_WORKFLOW: Record<AudioCppTask, AudioCppWorkflow> = {
  tts: "speak",
  music: "music",
  asr: "transcribe",
};

export function audioCppWorkflowsFor(
  model: AudioCppModel,
): readonly AudioCppWorkflow[] {
  return model.workflows ?? [TASK_WORKFLOW[model.task]];
}

export function audioCppModelSpeaks(id: string | null | undefined): boolean {
  const model = audioCppModelFor(id);
  return !model || audioCppWorkflowsFor(model).includes("speak");
}

export function audioCppModelsForTask(task: AudioCppTask): AudioCppModel[] {
  return AUDIO_CPP_MODELS.filter((model) => model.task === task);
}

/** Music length the backend clamps to, whatever max_tokens asks for. */
export const AUDIO_CPP_MUSIC_MIN_SECONDS = 5;
export const AUDIO_CPP_MUSIC_MAX_SECONDS = 240;

export function audioCppSizeLabel(sizeBytes: number): string {
  const mb = sizeBytes / MB;
  return mb < 1000 ? `${Math.round(mb)} MB` : `${(mb / 1024).toFixed(1)} GB`;
}
