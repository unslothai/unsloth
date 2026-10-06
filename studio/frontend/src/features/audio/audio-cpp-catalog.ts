// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Seeds the Audio pickers and dictation settings; any GGUF the backend recognises works too.
// Free of app imports so the node test runner can load it directly.

export const AUDIO_CPP_REPO = "audio-cpp/audio.cpp-gguf";

export const AUDIO_CPP_TTS_AUDIO_TYPE = "audiocpp_tts";
export const AUDIO_CPP_MUSIC_AUDIO_TYPE = "audiocpp_music";
/** Source separation (HTDemucs, RoFormers). Mirrors AUDIO_CPP_SEP_AUDIO_TYPE in audio_cpp_models.py. */
export const AUDIO_CPP_SEP_AUDIO_TYPE = "audiocpp_sep";
export const AUDIO_CPP_AUDIO_TYPES: ReadonlySet<string> = new Set([
  AUDIO_CPP_TTS_AUDIO_TYPE,
  AUDIO_CPP_MUSIC_AUDIO_TYPE,
  AUDIO_CPP_SEP_AUDIO_TYPE,
]);

export type AudioCppTask = "tts" | "music" | "asr" | "sep";

/** Mirrors AudioWorkflowId; spelled out to keep this file import-free. */
export type AudioCppWorkflow =
  | "speak"
  | "clone"
  | "edit"
  | "convert"
  | "music"
  | "separate"
  | "transcribe";

export interface AudioCppModel {
  /** Hub repo id, or `${AUDIO_CPP_REPO}/<folder>` for a package in the shared repo. */
  id: string;
  task: AudioCppTask;
  workflows?: readonly AudioCppWorkflow[];
  /** ASR only: primary language codes; absent means multilingual. */
  languages?: readonly string[];
  /** uses eSpeak-ng phonemization, absent from upstream runtime bundles. */
  needsEspeak?: boolean;
  stems?: readonly string[];
}

/** the `audio_cpp_runtime` block of `/api/inference/audio/stt/status`. */
export interface AudioCppRuntimeStatus {
  available: boolean;
  espeak: boolean;
  backend: string | null;
  release_tag: string | null;
  /** update target tag; null for unmanaged or unknown runtimes. */
  expected_tag?: string | null;
  /** managed runtime differs from the target; absent on older servers. */
  outdated?: boolean;
}

const MB = 1024 * 1024;

const folder = (name: string) => `${AUDIO_CPP_REPO}/${name}`;
const ENGLISH = ["en"] as const;

export const AUDIO_CPP_MODELS: readonly AudioCppModel[] = [
  { id: folder("Kokoro-82M-GGUF"), task: "tts", needsEspeak: true },
  { id: folder("KittenTTS-GGUF"), task: "tts", needsEspeak: true },
  { id: folder("Piper-TTS-GGUF"), task: "tts", needsEspeak: true },
  { id: folder("Inflect-Micro-v2-GGUF"), task: "tts", needsEspeak: true },
  { id: folder("PocketTTS-GGUF"), task: "tts", workflows: ["speak", "clone"] },
  { id: folder("MOSS-TTS-Nano-100M-GGUF"), task: "tts", workflows: ["speak", "clone"] },
  { id: folder("Supertonic-3-GGUF"), task: "tts" },
  { id: folder("Chatterbox-Turbo-GGUF"), task: "tts" },
  { id: folder("VoxCPM2-GGUF"), task: "tts", workflows: ["speak", "clone"] },
  { id: folder("Qwen3-TTS-12Hz-0.6B-Base-GGUF"), task: "tts", workflows: ["clone"] },
  { id: folder("Chatterbox-GGUF"), task: "tts", workflows: ["clone", "convert"] },
  { id: folder("IndexTTS2-GGUF"), task: "tts", workflows: ["clone"] },
  { id: folder("CosyVoice3-GGUF"), task: "tts", workflows: ["clone"] },
  { id: folder("Qwen3-TTS-12Hz-1.7B-CustomVoice-GGUF"), task: "tts" },
  { id: folder("Qwen3-TTS-12Hz-1.7B-VoiceDesign-GGUF"), task: "tts" },
  { id: folder("DotTTS-Edit-GGUF"), task: "tts", workflows: ["speak", "clone", "edit"] },
  { id: folder("Vevo2-GGUF"), task: "tts", workflows: ["clone", "edit", "convert"] },
  { id: folder("FireRedAudio-GGUF"), task: "tts", workflows: ["clone", "edit"] },
  { id: folder("SeedVC-MLX-GGUF"), task: "tts", workflows: ["convert"] },
  { id: folder("RVC-GGUF"), task: "tts", workflows: ["convert"] },
  { id: folder("MeanVC2-GGUF"), task: "tts", workflows: ["convert"] },
  { id: folder("Tone-Color-VC-GGUF"), task: "tts", workflows: ["convert"] },
  { id: folder("Breeze-TTS-2-GGUF"), task: "tts", workflows: ["speak", "clone"] },
  { id: folder("DotTTS-MF-GGUF"), task: "tts", workflows: ["speak", "clone"] },
  { id: folder("DotTTS-SOAR-GGUF"), task: "tts", workflows: ["speak", "clone"] },
  // Speak only: its spec lists clone, but from a short reference clip it may not keep the voice.
  { id: folder("DramaBox-GGUF"), task: "tts" },
  { id: folder("Higgs-Audio-v3-TTS-4B-GGUF"), task: "tts", workflows: ["speak", "clone"] },
  { id: folder("Irodori-TTS-500M-v3-GGUF"), task: "tts", workflows: ["speak", "clone"] },
  { id: folder("Irodori-TTS-600M-v3-VoiceDesign-GGUF"), task: "tts", workflows: ["speak", "clone"] },
  { id: folder("Irodori-TTS-v4-Small-GGUF"), task: "tts", workflows: ["speak", "clone"] },
  { id: folder("MOSS-TTS-Local-v1.5-GGUF"), task: "tts", workflows: ["speak", "clone"] },
  { id: folder("MOSS-VoiceGenerator-GGUF"), task: "tts" },
  { id: folder("MagpieTTS-Multilingual-357M-GGUF"), task: "tts" },
  { id: folder("Maya1-GGUF"), task: "tts" },
  { id: folder("NeuTTS-2E-GGUF"), task: "tts" },
  { id: folder("OmniVoice-GGUF"), task: "tts", workflows: ["speak", "clone"] },
  { id: folder("VibeVoice-1.5B-GGUF"), task: "tts" },
  { id: folder("VoxCPM1-GGUF"), task: "tts", workflows: ["speak", "clone"] },
  { id: folder("Fish-Audio-S2-Pro-GGUF"), task: "tts", workflows: ["speak", "clone"] },
  { id: folder("Confucius4-TTS-GGUF"), task: "tts", workflows: ["clone"] },
  { id: folder("FireRedTTS3-Base-GGUF"), task: "tts", workflows: ["clone"] },
  { id: folder("FireRedTTS3-Instruct-GGUF"), task: "tts", workflows: ["clone"] },
  { id: folder("IndexTTS2.5-GGUF"), task: "tts", workflows: ["clone"] },
  { id: folder("MioTTS-1.7B-GGUF"), task: "tts", workflows: ["clone"] },
  { id: folder("Qwen3-TTS-12Hz-1.7B-Base-GGUF"), task: "tts", workflows: ["clone"] },
  { id: "audio-cpp/MiniMax-Music3-GGUF", task: "music" },
  { id: "audio-cpp/Yue2-3B-GGUF", task: "music" },
  { id: folder("ACE-Step1.5-GGUF"), task: "music" },
  { id: folder("Stable-Audio-3-Small-Music-GGUF"), task: "music" },
  { id: folder("Stable-Audio-3-Small-SFX-GGUF"), task: "music" },
  { id: folder("ControlFoley-GGUF"), task: "music" },
  { id: folder("HeartMuLa-GGUF"), task: "music" },
  { id: folder("MiDashengLM-Gen-GGUF"), task: "music" },
  { id: folder("Stable-Audio-3-Medium-GGUF"), task: "music" },
  {
    id: folder("HTDemucs-GGUF"),
    task: "sep",
    workflows: ["separate"],
    stems: ["vocals", "drums", "bass", "other"],
  },
  {
    id: folder("BS-RoFormer-ep368-GGUF"),
    task: "sep",
    workflows: ["separate"],
    stems: ["vocals", "instrumental"],
  },
  {
    id: folder("HTDemucs-6stems-GGUF"),
    task: "sep",
    workflows: ["separate"],
    stems: ["vocals", "drums", "bass", "guitar", "piano", "other"],
  },
  {
    id: folder("Mel-Band-RoFormer-GGUF"),
    task: "sep",
    workflows: ["separate"],
    stems: ["vocals", "instrumental"],
  },
  { id: folder("Qwen3-ASR-0.6B-GGUF"), task: "asr" },
  { id: folder("Qwen3-ASR-1.7B-GGUF"), task: "asr" },
  { id: folder("Parakeet-TDT-0.6B-v3-GGUF"), task: "asr" },
  { id: folder("Canary-180M-Flash-GGUF"), task: "asr", languages: ["en", "de", "es", "fr"] },
  { id: folder("Moonshine-Streaming-GGUF"), task: "asr", languages: ENGLISH },
  { id: folder("Nemotron-3.5-ASR-Streaming-0.6B-GGUF"), task: "asr", languages: ENGLISH },
  // Diarizes; Transcribe offers it for the Speakers switch.
  { id: folder("MOSS-Transcribe-Diarize-GGUF"), task: "asr" },
  { id: folder("Citrinet-ASR-GGUF"), task: "asr" },
  { id: folder("Cohere-Transcribe-GGUF"), task: "asr" },
  { id: folder("Fun-ASR-Nano-2512-GGUF"), task: "asr" },
  { id: folder("GigaAM-ASR-GGUF"), task: "asr" },
  { id: folder("Granite-Speech-5.0-470M-TurboCTC-GGUF"), task: "asr", languages: ENGLISH },
  { id: folder("Higgs-Audio-v3-STT-GGUF"), task: "asr" },
  { id: folder("Hviske-v5.3-GGUF"), task: "asr", languages: ["da"] },
  { id: folder("Kroko-ASR-GGUF"), task: "asr", languages: ENGLISH },
  { id: folder("Niagara-ASR-GGUF"), task: "asr", languages: ENGLISH },
  { id: folder("VibeVoice-ASR-GGUF"), task: "asr" },
  { id: folder("Voxtral-Mini-4B-Realtime-2602-GGUF"), task: "asr" },
];

/** Shared-repo folders the pickers leave out, and why; the nightly catalog check fails on an unclassified folder. */
export const AUDIO_CPP_UNOFFERED_FOLDERS: Readonly<Record<string, string>> = {
  "CrisperWhisper2.0-GGUF": "needs a newer audio runtime",
  "KugelAudio-0-Open-GGUF": "needs a newer audio runtime",
  "OWSM-CTC-GGUF": "needs a newer audio runtime",
  "OWSM-GGUF": "needs a newer audio runtime",
  "Sidon-GGUF": "needs a newer audio runtime",
  "Smart-Turn-v3-GGUF": "needs a newer audio runtime",
  "MioCodec-25Hz-44.1kHz-v2-GGUF": "the codec MioTTS downloads and loads with",
  "Qwen3-ForcedAligner-0.6B-GGUF": "the aligner Transcribe uses for Qwen3-ASR timestamps",
  "MiniMax-H3-Q4-GGUF": "an audio and video package Studio cannot load yet",
  "Apollo-GGUF": "audio restoration has no page",
  "AudioSR-GGUF": "audio super-resolution has no page",
  "UniverSR-GGUF": "audio super-resolution has no page",
  "PersonaPlex-GGUF": "duplex speech chat has no page",
  "MMS-Forced-Aligner-GGUF": "forced alignment has no page",
  "Sortformer-Diar-4spk-v1-GGUF": "speaker diarization has no page",
  "PulseVAD-GGUF": "voice activity detection has no page",
  "MuScriptor-Small-GGUF": "music transcription has no page",
  "Samsone-GGUF": "describes audio rather than transcribing it",
};

// Families the backend marks speaks=False (audio_cpp_models.FAMILIES), by repo name: Hub rows
// carry no backend workflows before download. Chatterbox-Turbo is its own family and speaks.
const CLONE_ONLY_FAMILY_HINT =
  /miotts|vevo-?2|fireredtts-?3|firered-?audio|indextts-?2|cosyvoice-?3|confucius-?4|echo-?tts|f5-?tts|chatterbox(?!-?turbo)|qwen3-?tts[^/]*-base/i;

export function isCloneOnlyFamilyId(id: string | null | undefined): boolean {
  return CLONE_ONLY_FAMILY_HINT.test(id ?? "");
}

// Families that both speak and clone (fish_audio), likewise by repo name.
const SPEAK_AND_CLONE_FAMILY_HINT = /fish-?(audio|speech)|openaudio/i;

export function isSpeakAndCloneFamilyId(id: string | null | undefined): boolean {
  return SPEAK_AND_CLONE_FAMILY_HINT.test(id ?? "");
}

// Families that clone and convert (chatterbox, vevo2) and the convert-only ones (rvc, seed_vc,
// meanvc2, tone_color_vc), by repo name the way the backend's family_from_names reads it.
const CLONE_AND_CONVERT_FAMILY_HINT = /vevo[-_]?2|chatterbox(?![-_]?turbo)/i;
const CONVERT_ONLY_FAMILY_HINT = /rvc|seed[-_]?vc|meanvc[-_]?2|tone[-_]?color/i;

export function isCloneAndConvertFamilyId(id: string | null | undefined): boolean {
  return CLONE_AND_CONVERT_FAMILY_HINT.test(id ?? "");
}

export function isConvertOnlyFamilyId(id: string | null | undefined): boolean {
  return CONVERT_ONLY_FAMILY_HINT.test(id ?? "");
}

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
  sep: "separate",
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

/** Music length the backend clamps to, whatever max_tokens asks for. */
export const AUDIO_CPP_MUSIC_MIN_SECONDS = 5;
export const AUDIO_CPP_MUSIC_MAX_SECONDS = 240;

export function audioCppSizeLabel(sizeBytes: number): string {
  const mb = sizeBytes / MB;
  return mb < 1000 ? `${Math.round(mb)} MB` : `${(mb / 1024).toFixed(1)} GB`;
}
