// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isTauri } from "@/lib/api-base";
import { isAudioCppFolderId } from "../../audio/audio-cpp-catalog";
import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import {
  DEFAULT_STT_MODEL,
  type DefaultSttModel,
  STT_MODELS,
  STT_MODEL_LANGUAGES,
  STT_MODEL_REPOS,
  type SttModel,
  migrateVoiceSettings,
} from "./stt-model-catalog";

export * from "./stt-model-catalog";

// Adapters read these at call time, so changes apply without reloading the chat runtime.

export interface RecentDictation {
  id: string;
  text: string;
  at: number;
  chatId?: string;
  audioDataUrl?: string;
}

// pagination keeps full history; quota failures trim it to QUOTA_TRIM_KEEP.
const QUOTA_TRIM_KEEP = 50;
// transcript caps prevent synchronous localStorage quota failures.
const MAX_RECENT_DICTATION_LENGTH = 2000;
export const MAX_DICTIONARY_ENTRIES = 300;
const MAX_DICTIONARY_ENTRY_LENGTH = 120;

export const RECENT_DICTATION_LIMIT_OPTIONS = [0, 5, 10, 25, 50] as const;
export const DEFAULT_MAX_RECENT_DICTATIONS = 0;

export const STT_KEEP_ALIVE_OPTIONS = [300, 900, 1800, 3600, 0] as const;
export const DEFAULT_STT_KEEP_ALIVE_SECONDS = 300;

/**
 * 80 curated default vocabulary words (not including the user's name).
 * Words from the user's profile name are automatically prepended on top of these 80 words.
 */
export const DEFAULT_DICTATION_DICTIONARY_WORDS: readonly string[] = [
  "Unsloth",
  "QLoRA",
  "LoRA",
  "GGUF",
  "GGML",
  "llama.cpp",
  "whisper.cpp",
  "safetensors",
  "PyTorch",
  "Transformers",
  "Hugging Face",
  "bfloat16",
  "float16",
  "FlashAttention",
  "xFormers",
  "Triton",
  "vLLM",
  "SGLang",
  "Ollama",
  "LM Studio",
  "DeepSeek",
  "Qwen",
  "Mistral",
  "Mixtral",
  "Llama",
  "Gemma",
  "Phi",
  "Cohere",
  "Anthropic",
  "Claude",
  "OpenAI",
  "ChatGPT",
  "Gemini",
  "Parakeet",
  "Voxtral",
  "Whisper",
  "Kokoro",
  "Orpheus",
  "GRPO",
  "RLHF",
  "DPO",
  "ORPO",
  "SFT",
  "PEFT",
  "TRL",
  "BitsAndBytes",
  "DeepSpeed",
  "FSDP",
  "Megatron",
  "KV Cache",
  "RoPE",
  "SwiGLU",
  "RMSNorm",
  "CUDA",
  "cuDNN",
  "NVLink",
  "TensorRT",
  "ROCm",
  "ONNX",
  "Vulkan",
  "OpenCL",
  "CoreML",
  "MLX",
  "TypeScript",
  "JavaScript",
  "Python",
  "Rust",
  "Tauri",
  "React",
  "Vite",
  "Zustand",
  "Tailwind",
  "FastAPI",
  "Pydantic",
  "Uvicorn",
  "PostgreSQL",
  "SQLite",
  "Docker",
  "Kubernetes",
  "GitHub",
] as const;

function readStoredUserProfileNames(): { displayName: string; nickname: string } {
  if (typeof localStorage === "undefined") {
    return { displayName: "", nickname: "" };
  }
  try {
    const raw = localStorage.getItem("unsloth_user_profile");
    if (!raw) return { displayName: "", nickname: "" };
    const parsed = JSON.parse(raw) as {
      state?: { displayName?: unknown; nickname?: unknown };
    };
    return {
      displayName:
        typeof parsed?.state?.displayName === "string"
          ? parsed.state.displayName
          : "",
      nickname:
        typeof parsed?.state?.nickname === "string"
          ? parsed.state.nickname
          : "",
    };
  } catch {
    return { displayName: "", nickname: "" };
  }
}

/**
 * Extracts individual words from the user's configured display name and nickname.
 */
export function extractUserNameWords(
  displayName?: string,
  nickname?: string,
): string[] {
  const stored =
    displayName === undefined && nickname === undefined
      ? readStoredUserProfileNames()
      : { displayName: displayName ?? "", nickname: nickname ?? "" };
  const combined = `${stored.displayName} ${stored.nickname}`.trim();
  if (!combined) return [];
  const words: string[] = [];
  const seen = new Set<string>();
  for (const token of combined.split(/[\s,;/|]+/)) {
    const cleaned = token
      .replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}._-]+$/gu, "")
      .trim()
      .slice(0, MAX_DICTIONARY_ENTRY_LENGTH);
    if (!cleaned) continue;
    const key = cleaned.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    words.push(cleaned);
  }
  return words;
}

export function buildDefaultDictationDictionary(
  displayName?: string,
  nickname?: string,
): string[] {
  const userWords = extractUserNameWords(displayName, nickname);
  const result: string[] = [];
  const seen = new Set<string>();
  for (const word of [...userWords, ...DEFAULT_DICTATION_DICTIONARY_WORDS]) {
    const trimmed = word.trim().slice(0, MAX_DICTIONARY_ENTRY_LENGTH);
    if (!trimmed) continue;
    const key = trimmed.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    result.push(trimmed);
  }
  return result.slice(0, MAX_DICTIONARY_ENTRIES);
}

const HF_REPO_ID =
  /^[A-Za-z0-9][A-Za-z0-9._-]{0,95}\/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$/;

export function isSttModelId(value: string): boolean {
  const normalized = value.trim();
  return (
    (STT_MODELS as readonly string[]).includes(normalized) ||
    HF_REPO_ID.test(normalized) ||
    isAudioCppFolderId(normalized)
  );
}

export function normalizeSttModel(value: unknown): SttModel {
  if (typeof value !== "string") {
    return DEFAULT_STT_MODEL;
  }
  const normalized = value.trim();
  return isSttModelId(normalized) ? normalized : DEFAULT_STT_MODEL;
}

export function getSttModelRepo(model: SttModel): string {
  return STT_MODEL_REPOS[model as DefaultSttModel] ?? normalizeSttModel(model);
}

// Custom `.en` checkpoints are treated as English-only.

export function isSttModelLanguageCompatible(
  model: SttModel,
  language: string,
): boolean {
  const allowed =
    STT_MODEL_LANGUAGES.get(model) ??
    (getSttModelRepo(model).toLowerCase().endsWith(".en") ? ["en"] : null);
  if (!allowed) {
    return true;
  }
  const normalized = language.trim().replaceAll("_", "-").toLowerCase();
  // Auto sends no forced language, which every checkpoint accepts.
  return normalized === "auto" || allowed.includes(normalized.split("-", 1)[0]);
}

export type DictationEngine = "browser" | "model" | "custom";

/** Mirrors the backend `SttLoadRequest.device`; "gpu" is not offered since it adds nothing. */
export type SttDevice = "auto" | "cpu";

export const DEFAULT_STT_DEVICE: SttDevice = "auto";

function normalizeSttDevice(value: unknown): SttDevice {
  return value === "cpu" ? "cpu" : DEFAULT_STT_DEVICE;
}

export type TtsEngine = "system" | "studio" | "custom";

/** Custom repos run safetensors on Transformers. */
export function isCuratedSttModel(model: SttModel): boolean {
  return (STT_MODELS as readonly string[]).includes(model.trim());
}

export interface VoiceSettingsState {
  /** input device; "default" selects the system microphone. */
  micDeviceId: string;
  setMicDeviceId: (value: string) => void;

  /** "browser" uses Web Speech API; "model" chooses the local backend from the model. */
  dictationEngine: DictationEngine;
  setDictationEngine: (value: DictationEngine) => void;

  /** STT model used when dictationEngine is "model". */
  sttModel: SttModel;
  setSttModel: (value: SttModel) => void;

  /** quant for a package folder `sttModel`; "" uses its resident or default quant. */
  sttGgufVariant: string;
  setSttGgufVariant: (value: string) => void;

  /** "cpu" keeps the model in system RAM; loads and transcriptions apply the current value. */
  sttDevice: SttDevice;
  setSttDevice: (value: SttDevice) => void;

  sttProviderId: string;
  setSttProviderId: (value: string) => void;
  sttProviderModel: string;
  setSttProviderModel: (value: string) => void;

  /** BCP 47 tag, or "auto" for engine-specific language detection. */
  dictationLanguage: string;
  setDictationLanguage: (value: string) => void;

  /** exact spellings applied to matching transcript words and phrases. */
  dictionary: string[];
  dictionarySeeded: boolean;
  addDictionaryEntry: (value: string) => void;
  updateDictionaryEntry: (index: number, value: string) => void;
  commitDictionaryEntry: (index: number) => void;
  removeDictionaryEntry: (index: number) => void;
  syncUserNameToDictionary: (displayName?: string, nickname?: string) => void;
  resetDictionaryToDefaults: () => void;

  /**Seconds before an idle local STT model is automatically unloaded (0 = never). */
  sttKeepAliveSeconds: number;
  setSttKeepAliveSeconds: (value: number) => void;

  /** Maximum number of recent dictations to retain (0 = off). */
  maxRecentDictations: number;
  setMaxRecentDictations: (value: number) => void;

  recentDictations: RecentDictation[];
  addRecentDictation: (
    text: string,
    chatId?: string,
    audioDataUrl?: string,
  ) => void;
  removeRecentDictation: (id: string) => void;
  clearRecentDictations: () => void;

  ttsEnabled: boolean;
  setTtsEnabled: (value: boolean) => void;

  /** Speak each finished assistant reply without a click. Off by default. */
  autoReadAloud: boolean;
  setAutoReadAloud: (value: boolean) => void;

  /** "system": speechSynthesis voices. "studio": the loaded TTS audio model.
   * "custom": a saved connection's /audio/speech endpoint. */
  ttsEngine: TtsEngine;
  setTtsEngine: (value: TtsEngine) => void;

  ttsProviderId: string;
  setTtsProviderId: (value: string) => void;
  ttsProviderModel: string;
  setTtsProviderModel: (value: string) => void;
  /** Blank defaults to alloy. */
  ttsProviderVoice: string;
  setTtsProviderVoice: (value: string) => void;

  /** Saved Audio voice id the "studio" engine speaks in; "" for the model's own voice. */
  ttsStudioVoiceId: string;
  setTtsStudioVoiceId: (value: string) => void;

  /** speechSynthesis voiceURI, or "default" for the system voice. */
  ttsVoiceURI: string;
  setTtsVoiceURI: (value: string) => void;

  ttsRate: number;
  setTtsRate: (value: number) => void;
  ttsPitch: number;
  setTtsPitch: (value: number) => void;
  ttsVolume: number;
  setTtsVolume: (value: number) => void;
}

/** On quota errors drop the oldest dictations instead of losing the whole save. */
const quotaSafeLocalStorage = {
  getItem: (key: string) => localStorage.getItem(key),
  removeItem: (key: string) => localStorage.removeItem(key),
  setItem: (key: string, value: string) => {
    try {
      localStorage.setItem(key, value);
      return;
    } catch {
      // Quota exceeded: trim dictation history, oldest first, and retry.
    }
    let parsed: { state?: { recentDictations?: RecentDictation[] } };
    try {
      parsed = JSON.parse(value) as typeof parsed;
    } catch {
      return;
    }
    const state = parsed.state;
    const recents = state?.recentDictations;
    if (!state || !Array.isArray(recents)) return;
    // Halve down to empty, since another store may already have consumed the quota.
    let keep = Math.min(QUOTA_TRIM_KEEP, recents.length);
    for (;;) {
      keep = keep >= recents.length ? Math.floor(recents.length / 2) : keep;
      try {
        state.recentDictations = recents.slice(0, keep);
        localStorage.setItem(key, JSON.stringify(parsed));
        return;
      } catch {
        // Still over quota: trim harder.
      }
      if (keep === 0) return;
      keep = Math.floor(keep / 2);
    }
  },
};

export const useVoiceSettingsStore = create<VoiceSettingsState>()(
  persist(
    (set) => ({
      micDeviceId: "default",
      setMicDeviceId: (micDeviceId) => set({ micDeviceId }),

      dictationEngine: isTauri ? "model" : "browser",
      setDictationEngine: (dictationEngine) => set({ dictationEngine }),

      sttModel: DEFAULT_STT_MODEL,
      setSttModel: (value) =>
        set((state) => {
          const normalized = normalizeSttModel(value);
          const sttModel = isSttModelLanguageCompatible(
            normalized,
            state.dictationLanguage,
          )
            ? normalized
            : DEFAULT_STT_MODEL;
          // a quant belongs to the model it was selected for.
          return sttModel === state.sttModel
            ? { sttModel }
            : { sttModel, sttGgufVariant: "" };
        }),

      sttGgufVariant: "",
      setSttGgufVariant: (sttGgufVariant) => set({ sttGgufVariant }),

      sttDevice: DEFAULT_STT_DEVICE,
      setSttDevice: (value) => set({ sttDevice: normalizeSttDevice(value) }),

      sttProviderId: "",
      setSttProviderId: (sttProviderId) => set({ sttProviderId }),
      sttProviderModel: "",
      setSttProviderModel: (sttProviderModel) => set({ sttProviderModel }),

      dictationLanguage: "auto",
      setDictationLanguage: (dictationLanguage) =>
        set((state) =>
          isSttModelLanguageCompatible(state.sttModel, dictationLanguage)
            ? { dictationLanguage }
            : {
                dictationLanguage,
                sttModel: DEFAULT_STT_MODEL,
                sttGgufVariant: "",
              },
        ),

      dictionary: buildDefaultDictationDictionary(),
      dictionarySeeded: true,
      addDictionaryEntry: (value) =>
        set((state) => {
          const trimmed = value.trim().slice(0, MAX_DICTIONARY_ENTRY_LENGTH);
          if (!trimmed) return state;
          if (state.dictionary.length >= MAX_DICTIONARY_ENTRIES) return state;
          if (
            state.dictionary.some(
              (entry) => entry.toLowerCase() === trimmed.toLowerCase(),
            )
          ) {
            return state;
          }
          return {
            dictionary: [...state.dictionary, trimmed],
            dictionarySeeded: true,
          };
        }),
      // Raw value while editing; commitDictionaryEntry finalizes on blur.
      updateDictionaryEntry: (index, value) =>
        set((state) => {
          const dictionary = [...state.dictionary];
          if (index < 0 || index >= dictionary.length) return state;
          dictionary[index] = value.slice(0, MAX_DICTIONARY_ENTRY_LENGTH);
          return { dictionary, dictionarySeeded: true };
        }),
      commitDictionaryEntry: (index) =>
        set((state) => {
          const dictionary = [...state.dictionary];
          if (index < 0 || index >= dictionary.length) return state;
          const trimmed = dictionary[index]?.trim() ?? "";
          if (trimmed) {
            dictionary[index] = trimmed;
          } else {
            dictionary.splice(index, 1);
          }
          return { dictionary, dictionarySeeded: true };
        }),
      removeDictionaryEntry: (index) =>
        set((state) => ({
          dictionary: state.dictionary.filter((_, i) => i !== index),
          dictionarySeeded: true,
        })),
      syncUserNameToDictionary: (displayName, nickname) =>
        set((state) => {
          const userWords = extractUserNameWords(displayName, nickname);
          if (userWords.length === 0) return state;
          const existingLower = new Set(
            state.dictionary.map((entry) => entry.toLowerCase()),
          );
          const additions = userWords.filter(
            (word) => !existingLower.has(word.toLowerCase()),
          );
          if (additions.length === 0) return state;
          return {
            dictionary: [...additions, ...state.dictionary].slice(
              0,
              MAX_DICTIONARY_ENTRIES,
            ),
            dictionarySeeded: true,
          };
        }),
      resetDictionaryToDefaults: () =>
        set({
          dictionary: buildDefaultDictationDictionary(),
          dictionarySeeded: true,
        }),

      sttKeepAliveSeconds: DEFAULT_STT_KEEP_ALIVE_SECONDS,
      setSttKeepAliveSeconds: (value) =>
        set({
          sttKeepAliveSeconds: normalizeSttKeepAliveSeconds(value),
        }),

      maxRecentDictations: DEFAULT_MAX_RECENT_DICTATIONS,
      setMaxRecentDictations: (value) =>
        set((state) => {
          const maxRecentDictations = normalizeMaxRecentDictations(value);
          return {
            maxRecentDictations,
            recentDictations:
              maxRecentDictations <= 0
                ? []
                : state.recentDictations.slice(0, maxRecentDictations),
          };
        }),

      recentDictations: [],
      addRecentDictation: (text, chatId, audioDataUrl) =>
        set((state) => {
          if (state.maxRecentDictations <= 0) {
            return state;
          }
          const trimmed = text.trim().slice(0, MAX_RECENT_DICTATION_LENGTH);
          if (!trimmed) {
            return state;
          }
          const at = Date.now();
          return {
            recentDictations: [
              {
                id: `${at}-${Math.random().toString(36).slice(2, 10)}`,
                text: trimmed,
                at,
                ...(chatId ? { chatId } : {}),
                ...(audioDataUrl ? { audioDataUrl } : {}),
              },
              ...state.recentDictations,
            ].slice(0, state.maxRecentDictations),
          };
        }),
      removeRecentDictation: (id) =>
        set((state) => ({
          recentDictations: state.recentDictations.filter(
            (dictation) => dictation.id !== id,
          ),
        })),
      clearRecentDictations: () => set({ recentDictations: [] }),

      ttsEnabled: true,
      setTtsEnabled: (ttsEnabled) => set({ ttsEnabled }),
      autoReadAloud: false,
      setAutoReadAloud: (autoReadAloud) => set({ autoReadAloud }),

      ttsEngine: "system",
      setTtsEngine: (ttsEngine) => set({ ttsEngine }),

      ttsProviderId: "",
      setTtsProviderId: (ttsProviderId) => set({ ttsProviderId }),
      ttsProviderModel: "",
      setTtsProviderModel: (ttsProviderModel) => set({ ttsProviderModel }),
      ttsProviderVoice: "",
      setTtsProviderVoice: (ttsProviderVoice) => set({ ttsProviderVoice }),

      ttsStudioVoiceId: "",
      setTtsStudioVoiceId: (ttsStudioVoiceId) => set({ ttsStudioVoiceId }),

      ttsVoiceURI: "default",
      setTtsVoiceURI: (ttsVoiceURI) => set({ ttsVoiceURI }),

      ttsRate: 1,
      setTtsRate: (ttsRate) => set({ ttsRate }),
      ttsPitch: 1,
      setTtsPitch: (ttsPitch) => set({ ttsPitch }),
      ttsVolume: 1,
      setTtsVolume: (ttsVolume) => set({ ttsVolume }),
    }),
    {
      name: "unsloth_voice_settings",
      storage: createJSONStorage(() => quotaSafeLocalStorage),
      version: 1,
      migrate: migrateVoiceSettings,
      merge: (persisted, current) => {
        const saved = persisted as Partial<VoiceSettingsState> | undefined;
        const dictationLanguage = asString(saved?.dictationLanguage, "auto");
        // "gguf" was a short-lived separate engine; both local backends now live under "model".
        const savedEngine = saved?.dictationEngine as string | undefined;
        const dictationEngine: DictationEngine =
          savedEngine === "custom"
            ? "custom"
            : isTauri || savedEngine === "model" || savedEngine === "gguf"
              ? "model"
              : "browser";
        const savedSttModel = normalizeSttModel(saved?.sttModel);
        const sttModel = isSttModelLanguageCompatible(
          savedSttModel,
          dictationLanguage,
        )
          ? savedSttModel
          : DEFAULT_STT_MODEL;
        return {
          ...current,
          micDeviceId: asString(saved?.micDeviceId, "default"),
          dictationEngine,
          sttModel,
          sttGgufVariant:
            sttModel === savedSttModel
              ? asString(saved?.sttGgufVariant, "")
              : "",
          sttDevice: normalizeSttDevice(saved?.sttDevice),
          sttProviderId: asString(saved?.sttProviderId, ""),
          sttProviderModel: asString(saved?.sttProviderModel, ""),
          dictationLanguage,
          dictionary: normalizeSavedDictionary(
            saved?.dictionary,
            saved?.dictionarySeeded === true,
          ),
          dictionarySeeded: true,
          sttKeepAliveSeconds: normalizeSttKeepAliveSeconds(
            saved?.sttKeepAliveSeconds,
          ),
          maxRecentDictations: normalizeMaxRecentDictations(
            saved?.maxRecentDictations,
          ),
          recentDictations: normalizeRecentDictations(
            saved?.recentDictations,
          ).slice(0, normalizeMaxRecentDictations(saved?.maxRecentDictations)),
          ttsEnabled:
            typeof saved?.ttsEnabled === "boolean" ? saved.ttsEnabled : true,
          autoReadAloud: saved?.autoReadAloud === true,
          ttsEngine:
            saved?.ttsEngine === "studio" || saved?.ttsEngine === "custom"
              ? saved.ttsEngine
              : "system",
          ttsProviderId: asString(saved?.ttsProviderId, ""),
          ttsProviderModel: asString(saved?.ttsProviderModel, ""),
          ttsProviderVoice: asString(saved?.ttsProviderVoice, ""),
          ttsStudioVoiceId: asString(saved?.ttsStudioVoiceId, ""),
          ttsVoiceURI: asString(saved?.ttsVoiceURI, "default"),
          ttsRate: clampNumber(saved?.ttsRate, 0.5, 2, 1),
          ttsPitch: clampNumber(saved?.ttsPitch, 0, 2, 1),
          ttsVolume: clampNumber(saved?.ttsVolume, 0, 1, 1),
        };
      },
    },
  ),
);

function asString(value: unknown, fallback: string): string {
  return typeof value === "string" && value ? value : fallback;
}

function normalizeSttKeepAliveSeconds(value: unknown): number {
  if (
    typeof value === "number" &&
    (STT_KEEP_ALIVE_OPTIONS as readonly number[]).includes(value)
  ) {
    return value;
  }
  return DEFAULT_STT_KEEP_ALIVE_SECONDS;
}

function normalizeMaxRecentDictations(value: unknown): number {
  if (
    typeof value === "number" &&
    (RECENT_DICTATION_LIMIT_OPTIONS as readonly number[]).includes(value)
  ) {
    return value;
  }
  return DEFAULT_MAX_RECENT_DICTATIONS;
}

function normalizeSavedDictionary(
  value: unknown,
  alreadySeeded: boolean,
): string[] {
  const cleaned = Array.isArray(value)
    ? value
        .filter((v): v is string => typeof v === "string" && !!v.trim())
        .map((v) => v.trim().slice(0, MAX_DICTIONARY_ENTRY_LENGTH))
    : [];
  const defaults = buildDefaultDictationDictionary();
  if (!alreadySeeded) {
    const merged: string[] = [];
    const seen = new Set<string>();
    for (const word of [...cleaned, ...defaults]) {
      const key = word.toLowerCase();
      if (seen.has(key)) continue;
      seen.add(key);
      merged.push(word);
    }
    return merged.slice(0, MAX_DICTIONARY_ENTRIES);
  }
  // Even when already seeded, ensure any current user profile name words are present.
  const userWords = extractUserNameWords();
  const seen = new Set(cleaned.map((entry) => entry.toLowerCase()));
  const additions = userWords.filter((word) => !seen.has(word.toLowerCase()));
  return [...additions, ...cleaned].slice(0, MAX_DICTIONARY_ENTRIES);
}

function normalizeRecentDictations(value: unknown): RecentDictation[] {
  if (!Array.isArray(value)) {
    return [];
  }

  const normalized: RecentDictation[] = [];
  for (const [index, entry] of value.entries()) {
    if (!entry || typeof entry !== "object") {
      continue;
    }
    const candidate = entry as Partial<RecentDictation>;
    if (
      typeof candidate.text !== "string" ||
      !candidate.text.trim() ||
      typeof candidate.at !== "number" ||
      !Number.isFinite(candidate.at)
    ) {
      continue;
    }
    normalized.push({
      id:
        typeof candidate.id === "string" && candidate.id
          ? candidate.id
          : `legacy-${candidate.at}-${index}`,
      text: candidate.text.trim().slice(0, MAX_RECENT_DICTATION_LENGTH),
      at: candidate.at,
      ...(typeof candidate.chatId === "string" && candidate.chatId
        ? { chatId: candidate.chatId }
        : {}),
      ...(typeof candidate.audioDataUrl === "string" &&
      candidate.audioDataUrl.startsWith("data:audio/")
        ? { audioDataUrl: candidate.audioDataUrl }
        : {}),
    });
  }
  return normalized;
}

function clampNumber(
  value: unknown,
  min: number,
  max: number,
  fallback: number,
): number {
  if (typeof value !== "number" || Number.isNaN(value)) return fallback;
  return Math.min(max, Math.max(min, value));
}

export function resolveDictationLanguage(setting?: string): string {
  const value = setting ?? useVoiceSettingsStore.getState().dictationLanguage;
  if (value && value !== "auto") return value;
  return typeof navigator !== "undefined" && navigator.language
    ? navigator.language
    : "en-US";
}

// transformers `LANGUAGES` plus stt_sidecar.py `_WHISPER_LANGUAGE_ALIASES`. Keep in sync with
// `_known_whisper_languages()`.
const WHISPER_DICTATION_LANGUAGES = new Set([
  "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo", "br", "bs",
  "ca", "cmn", "cs", "cy", "da", "de", "el", "en", "es", "et", "eu", "fa",
  "fi", "fil", "fo", "fr", "gl", "gu", "ha", "haw", "he", "hi", "hr", "ht",
  "hu", "hy", "id", "in", "is", "it", "iw", "ja", "ji", "jw", "ka", "kk",
  "km", "kn", "ko", "la", "lb", "ln", "lo", "lt", "lv", "mg", "mi", "mk",
  "ml", "mn", "mr", "ms", "mt", "my", "nb", "ne", "nl", "nn", "no", "oc",
  "pa", "pl", "ps", "pt", "ro", "ru", "sa", "sd", "si", "sk", "sl", "sn",
  "so", "sq", "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tk", "tl",
  "tr", "tt", "uk", "ur", "uz", "vi", "yi", "yo", "yue", "zh",
]);

/** Only literal "auto" resolves, and only to a language both the model and Whisper honor;
 * otherwise it stays auto-detect rather than 422ing every dictation. */
export function resolveModelDictationLanguage(
  model: SttModel,
  requested: string,
): string {
  if (requested !== "auto") return requested;
  const resolved = resolveDictationLanguage(requested);
  const primary = resolved
    .trim()
    .replaceAll("_", "-")
    .toLowerCase()
    .split("-", 1)[0];
  return isSttModelLanguageCompatible(model, resolved) &&
    WHISPER_DICTATION_LANGUAGES.has(primary)
    ? resolved
    : requested;
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * Case-insensitive, word-bounded dictionary and snippet replacement:
 * - Plain entries ("Jane Doe", "Unsloth") enforce exact casing.
 * - Alias/snippet entries ("my email -> team@unsloth.ai" or "k8s => Kubernetes")
 *   expand spoken triggers into their target text.
 * - Longer triggers run first so multi-word phrases win over single-word prefixes.
 */
export function applyDictationDictionary(
  transcript: string,
  dictionary?: string[],
): string {
  const entries =
    dictionary ?? [
      ...extractUserNameWords(),
      ...useVoiceSettingsStore.getState().dictionary,
    ];
  if (!transcript || entries.length === 0) return transcript;

  const rules: { trigger: string; replacement: string }[] = [];
  for (const entry of entries) {
    const trimmed = entry.trim();
    if (!trimmed) continue;
    const arrowMatch = /^(.+?)\s*(?:->|=>)\s*(.+)$/.exec(trimmed);
    if (arrowMatch) {
      const trigger = arrowMatch[1].trim();
      const replacement = arrowMatch[2].trim();
      if (trigger && replacement) {
        rules.push({ trigger, replacement });
      }
    } else {
      rules.push({ trigger: trimmed, replacement: trimmed });
    }
  }
  rules.sort((a, b) => b.trigger.length - a.trigger.length);

  let result = transcript;
  for (const { trigger, replacement } of rules) {
    const pattern = trigger.split(/\s+/).map(escapeRegExp).join("\\s+");
    try {
      // Capture the boundary instead of a lookbehind, which Safari < 16.4 cannot compile.
      const regex = new RegExp(
        `(^|[^\\p{L}\\p{N}])(${pattern})(?![\\p{L}\\p{N}])`,
        "giu",
      );
      // Callback form avoids $-pattern expansion.
      result = result.replace(
        regex,
        (_match, prefix: string) => `${prefix}${replacement}`,
      );
    } catch {
      // Skip entries that produce an invalid pattern.
    }
  }
  return result;
}

export function recordRecentDictation(
  text: string,
  chatId?: string,
  audioDataUrl?: string,
): void {
  useVoiceSettingsStore
    .getState()
    .addRecentDictation(text, chatId, audioDataUrl);
}
