// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { useVoiceSettingsStore } from "@/features/settings/stores/voice-settings-store";
import { toast } from "@/lib/toast";
import type { SpeechSynthesisAdapter } from "@assistant-ui/react";
import { encryptProviderApiKey } from "../api/providers-api";
import { getExternalProviderApiKey } from "../external-providers";
import { stripSearchImageTokens } from "../search-images/search-images";
import { useExternalProvidersStore } from "../stores/external-providers-store";

/** Voice for a stored voiceURI. "default" resolves to the voice the platform marks as its
 *  default, so the "System default" choice means what it says instead of falling back to a
 *  curated pick. Undefined lets the browser pick. */
export function findTtsVoice(
  voiceURI: string,
): SpeechSynthesisVoice | undefined {
  if (typeof window === "undefined" || !window.speechSynthesis) {
    return undefined;
  }
  if (!voiceURI) return undefined;
  const voices = window.speechSynthesis.getVoices();
  if (voiceURI === "default") return voices.find((voice) => voice.default);
  return voices.find((voice) => voice.voiceURI === voiceURI);
}

// macOS novelty and legacy Eloquence voices that sound robotic and flood the picker.
const LOW_QUALITY_VOICE_NAMES = new Set([
  "albert",
  "bad news",
  "bahh",
  "bells",
  "boing",
  "bubbles",
  "cellos",
  "deranged",
  "eddy",
  "flo",
  "fred",
  "good news",
  "grandma",
  "grandpa",
  "hysterical",
  "jester",
  "junior",
  "kathy",
  "organ",
  "princess",
  "ralph",
  "reed",
  "rocko",
  "sandy",
  "shelley",
  "superstar",
  "trinoids",
  "whisper",
  "wobble",
  "zarvox",
]);

function voiceBaseName(voice: SpeechSynthesisVoice): string {
  // "Eddy (English (US))" -> "eddy"; "Bad News" -> "bad news"
  const name = voice.name.split("(")[0]?.trim().toLowerCase() ?? "";
  return name;
}

// Well-known natural English voices, best first. Breaks ties when the name carries no quality
// hint, so basic voices are not just alphabetical.
const PREFERRED_VOICE_NAMES = [
  "samantha",
  "alex",
  "ava",
  "allison",
  "susan",
  "tom",
  "daniel",
  "serena",
  "karen",
  "moira",
  "tessa",
  "fiona",
];

// Quality tier from vendor hints in the voice name.
function voiceQualityScore(voice: SpeechSynthesisVoice): number {
  const name = voice.name.toLowerCase();
  let score = 0;
  if (name.includes("premium")) score += 100;
  if (name.includes("siri")) score += 90;
  if (name.includes("enhanced")) score += 80;
  if (name.includes("natural") || name.includes("neural")) score += 70;
  if (name.includes("google")) score += 40;
  if (name.includes("microsoft")) score += 30;
  return score;
}

// Higher for voices in the user's exact region, then the same language.
function voiceLocaleScore(voice: SpeechSynthesisVoice): number {
  const navLang =
    typeof navigator !== "undefined" && navigator.language
      ? navigator.language.toLowerCase()
      : "en-us";
  const lang = voice.lang.toLowerCase().replace("_", "-");
  if (lang === navLang) return 2;
  if (langBase(lang) === langBase(navLang)) return 1;
  return 0;
}

// Rank in the preferred list, best first; 0 when the voice is not listed.
function voicePreferredRank(voice: SpeechSynthesisVoice): number {
  const index = PREFERRED_VOICE_NAMES.indexOf(voiceBaseName(voice));
  return index === -1 ? 0 : PREFERRED_VOICE_NAMES.length - index;
}

function langBase(tag: string): string {
  return tag.toLowerCase().split(/[-_]/)[0] ?? "";
}

const MAX_CURATED_VOICES = 20;

/** Keep the best, most relevant voices: drop low-quality ones, keep English, the browser
 *  language and the dictation language, rank by quality hints, and cap the list. The selected
 *  voice is always kept. */
export function curateSystemVoices(
  voices: SpeechSynthesisVoice[],
  selectedVoiceURI?: string,
  dictationLanguage = useVoiceSettingsStore.getState().dictationLanguage,
): SpeechSynthesisVoice[] {
  const wantedLangs = new Set<string>(["en"]);
  if (typeof navigator !== "undefined" && navigator.language) {
    wantedLangs.add(langBase(navigator.language));
  }
  if (dictationLanguage && dictationLanguage !== "auto") {
    wantedLangs.add(langBase(dictationLanguage));
  }

  // WebKit and Linux engines report voices with empty or duplicate voiceURIs; drop them so the
  // Radix Select never gets an empty or colliding value.
  const seenVoiceURIs = new Set<string>();
  const kept = voices.filter((voice) => {
    if (!voice.voiceURI || seenVoiceURIs.has(voice.voiceURI)) return false;
    seenVoiceURIs.add(voice.voiceURI);
    if (LOW_QUALITY_VOICE_NAMES.has(voiceBaseName(voice))) return false;
    return wantedLangs.has(langBase(voice.lang));
  });

  kept.sort((a, b) => {
    const quality = voiceQualityScore(b) - voiceQualityScore(a);
    if (quality !== 0) return quality;
    const locale = voiceLocaleScore(b) - voiceLocaleScore(a);
    if (locale !== 0) return locale;
    const preferred = voicePreferredRank(b) - voicePreferredRank(a);
    if (preferred !== 0) return preferred;
    const byDefault = Number(b.default) - Number(a.default);
    if (byDefault !== 0) return byDefault;
    return a.name.localeCompare(b.name);
  });

  // macOS reports some voices twice (compact + enhanced) under one name. Keep one per name and
  // language, preferring the selected voice then the best ranked.
  const keyOf = (voice: SpeechSynthesisVoice) =>
    `${voiceBaseName(voice)}|${voice.lang.toLowerCase()}`;
  const winners = new Map<string, string>();
  for (const voice of kept) {
    const key = keyOf(voice);
    if (!winners.has(key)) winners.set(key, voice.voiceURI);
    if (voice.voiceURI === selectedVoiceURI) winners.set(key, voice.voiceURI);
  }
  const deduped = kept.filter(
    (voice) => winners.get(keyOf(voice)) === voice.voiceURI,
  );

  const curated = deduped.slice(0, MAX_CURATED_VOICES);
  if (
    selectedVoiceURI &&
    selectedVoiceURI !== "default" &&
    !curated.some((voice) => voice.voiceURI === selectedVoiceURI)
  ) {
    const selected = voices.find(
      (voice) => voice.voiceURI === selectedVoiceURI,
    );
    if (selected) curated.push(selected);
  }
  return curated;
}

/** Best voice when none is chosen. The browser default on macOS is often a robotic legacy
 *  voice, so fall back to the top curated voice instead. */
function defaultTtsVoice(): SpeechSynthesisVoice | undefined {
  if (typeof window === "undefined" || !window.speechSynthesis) {
    return undefined;
  }
  const voices = window.speechSynthesis.getVoices();
  return (
    curateSystemVoices(voices)[0] ??
    voices.find((voice) => voice.default) ??
    voices[0]
  );
}

/** Build an utterance from the current Voice settings. */
export function createConfiguredUtterance(
  text: string,
): SpeechSynthesisUtterance {
  const { ttsVoiceURI, ttsRate, ttsPitch, ttsVolume } =
    useVoiceSettingsStore.getState();
  const utterance = new SpeechSynthesisUtterance(text);
  const voice = findTtsVoice(ttsVoiceURI) ?? defaultTtsVoice();
  if (voice) {
    utterance.voice = voice;
    utterance.lang = voice.lang;
  }
  utterance.rate = ttsRate;
  utterance.pitch = ttsPitch;
  utterance.volume = ttsVolume;
  return utterance;
}

/** Generate speech via the loaded TTS audio model; returns a WAV data URL. */
export async function generateStudioTtsAudio(
  text: string,
  signal?: AbortSignal,
): Promise<string> {
  const response = await authFetch("/api/inference/audio/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages: [{ role: "user", content: text }],
      stream: false,
    }),
    signal,
  });
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as {
      detail?: string;
    } | null;
    const detail = body?.detail ?? `HTTP ${response.status}`;
    if (/no model loaded|not an audio model/i.test(detail)) {
      throw new Error(
        "No TTS model is loaded. Load an audio model (e.g. Orpheus TTS) from the model selector, then try again.",
      );
    }
    throw new Error(detail);
  }
  const data = (await response.json()) as { audio?: { data?: string } };
  if (!data.audio?.data) {
    throw new Error("The TTS model returned no audio.");
  }
  return `data:audio/wav;base64,${data.audio.data}`;
}

/** Speech via a saved connection's /audio/speech. Returns an object URL to release. */
export async function generateCustomTtsAudio(
  text: string,
  signal?: AbortSignal,
): Promise<string> {
  const providersState = useExternalProvidersStore.getState();
  if (!providersState.connectionsEnabled) {
    throw new Error(
      "Connections are disabled. Turn on Enable connections in Settings → Connections to use a custom TTS endpoint.",
    );
  }
  const { ttsProviderId, ttsProviderModel, ttsProviderVoice } =
    useVoiceSettingsStore.getState();
  const model = ttsProviderModel.trim();
  const voice = ttsProviderVoice.trim() || "alloy";
  if (!ttsProviderId || !model) {
    throw new Error(
      "Custom TTS is not configured. Pick a connection and model in Settings → Voice.",
    );
  }
  // A browser whose key migration failed keeps the connection selectable on a retained legacy
  // key; send it like chat and STT do, or this call is unauthenticated.
  const provider = providersState.providers.find(
    (candidate) => candidate.id === ttsProviderId,
  );
  if (!provider) {
    useVoiceSettingsStore.getState().setTtsProviderId("");
    throw new Error(
      "The custom TTS connection no longer exists. Pick another connection in Settings → Voice.",
    );
  }
  const legacyApiKey = provider.hasApiKey
    ? ""
    : getExternalProviderApiKey(ttsProviderId).trim();
  const encryptedApiKey = legacyApiKey
    ? await encryptProviderApiKey(legacyApiKey)
    : "";

  // Encryption and auth refresh both yield. Reuse this check before the first request and every
  // authFetch retry so neither path can release assistant text or a retained key after the
  // frontend-only connection policy changes.
  const assertConnectionSnapshot = () => {
    const currentProvidersState = useExternalProvidersStore.getState();
    if (!currentProvidersState.connectionsEnabled) {
      throw new Error(
        "Connections are disabled. Turn on Enable connections in Settings → Connections to use a custom TTS endpoint.",
      );
    }
    const currentProvider = currentProvidersState.providers.find(
      (candidate) => candidate.id === ttsProviderId,
    );
    if (!currentProvider) {
      useVoiceSettingsStore.getState().setTtsProviderId("");
      throw new Error(
        "The custom TTS connection no longer exists. Pick another connection in Settings → Voice.",
      );
    }
    if (
      currentProvider.baseUrl !== provider.baseUrl ||
      currentProvider.providerType !== provider.providerType ||
      currentProvider.backendProviderType !== provider.backendProviderType ||
      currentProvider.hasApiKey !== provider.hasApiKey ||
      currentProvider.updatedAt !== provider.updatedAt
    ) {
      throw new Error(
        "The custom TTS connection changed while the request was starting. Try again.",
      );
    }
  };
  assertConnectionSnapshot();
  const response = await authFetch(
    "/api/inference/audio/speech",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        input: text,
        provider_id: ttsProviderId,
        provider_base_url: provider.baseUrl,
        model,
        voice,
        ...(encryptedApiKey ? { encrypted_api_key: encryptedApiKey } : {}),
      }),
      signal,
    },
    { beforeRetry: assertConnectionSnapshot },
  );
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as {
      detail?: string;
    } | null;
    throw new Error(body?.detail ?? `HTTP ${response.status}`);
  }
  const bytes = await response.arrayBuffer();
  const contentType =
    response.headers.get("content-type")?.split(";")[0]?.trim() || "audio/wav";
  return URL.createObjectURL(new Blob([bytes], { type: contentType }));
}

/** Release a URL returned by the generate helpers (data URLs need nothing). */
export function releaseTtsAudioUrl(url: string): void {
  if (url.startsWith("blob:")) URL.revokeObjectURL(url);
}

function speakWithBackendAudio(
  generate: (text: string, signal?: AbortSignal) => Promise<string>,
  text: string,
  handleEnd: (
    reason: "finished" | "error" | "cancelled",
    error?: unknown,
  ) => void,
  markRunning: () => void,
): { cancel: () => void } {
  const { ttsRate, ttsVolume } = useVoiceSettingsStore.getState();
  const controller = new AbortController();
  let audio: HTMLAudioElement | null = null;
  let audioUrl: string | null = null;
  let cancelled = false;

  // Release the element and its multi-MB WAV data URL as soon as playback ends.
  const cleanup = () => {
    if (audio) {
      audio.pause();
      audio.removeAttribute("src");
      audio = null;
    }
    if (audioUrl) {
      releaseTtsAudioUrl(audioUrl);
      audioUrl = null;
    }
  };

  void (async () => {
    try {
      const url = await generate(text, controller.signal);
      if (cancelled) {
        releaseTtsAudioUrl(url);
        return;
      }
      audioUrl = url;
      audio = new Audio(url);
      audio.playbackRate = ttsRate;
      audio.volume = ttsVolume;
      // Some browsers reset playbackRate to 1 once the source loads; reapply it on loadedmetadata so
      // the speed setting reliably takes effect.
      audio.addEventListener("loadedmetadata", () => {
        if (audio) audio.playbackRate = ttsRate;
      });
      audio.addEventListener("ended", () => {
        cleanup();
        handleEnd("finished");
      });
      audio.addEventListener("error", () => {
        if (cancelled) return;
        cleanup();
        handleEnd("error", new Error("Audio playback failed."));
      });
      markRunning();
      await audio.play();
    } catch (error) {
      if (cancelled || controller.signal.aborted) return;
      cleanup();
      handleEnd("error", error);
    }
  })();

  return {
    cancel: () => {
      cancelled = true;
      controller.abort();
      cleanup();
      handleEnd("cancelled");
    },
  };
}

/** Text-to-speech for assistant messages. Reads Voice settings at speak time. Engines:
 *  "system" (speechSynthesis), "studio" (local TTS model), "custom" (a connection). */
export class StudioSpeechSynthesisAdapter implements SpeechSynthesisAdapter {
  /** Web Speech synthesis, used by the "system" engine. */
  static systemVoicesSupported(): boolean {
    return (
      typeof window !== "undefined" &&
      "speechSynthesis" in window &&
      typeof window.SpeechSynthesisUtterance !== "undefined"
    );
  }

  // The "studio" engine only needs fetch plus Audio playback, so a WebView without Web Speech
  // synthesis can still read aloud through the backend.
  static isSupported(): boolean {
    return (
      StudioSpeechSynthesisAdapter.systemVoicesSupported() ||
      (typeof window !== "undefined" && typeof window.Audio !== "undefined")
    );
  }

  speak(spokenText: string): SpeechSynthesisAdapter.Utterance {
    // Renderer markup: without this the reader says the token id out loud.
    const text = stripSearchImageTokens(spokenText);
    const subscribers = new Set<() => void>();

    const handleEnd = (
      reason: "finished" | "error" | "cancelled",
      error?: unknown,
    ) => {
      if (res.status.type === "ended") return;
      // Surface genuine read-aloud failures; a cancelled or interrupted utterance is a normal stop,
      // not an error, and must not toast.
      if (
        reason === "error" &&
        error !== "interrupted" &&
        error !== "canceled"
      ) {
        toast.error(
          error instanceof Error ? error.message : "Read aloud failed.",
        );
      }
      res.status = { type: "ended", reason, error };
      for (const handler of subscribers) handler();
    };

    let cancelImpl: () => void;
    const { ttsEngine } = useVoiceSettingsStore.getState();

    const res: SpeechSynthesisAdapter.Utterance = {
      status: { type: "starting" },
      cancel: () => cancelImpl(),
      subscribe: (callback) => {
        if (res.status.type === "ended") {
          let cancelled = false;
          queueMicrotask(() => {
            if (!cancelled) callback();
          });
          return () => {
            cancelled = true;
          };
        }
        subscribers.add(callback);
        return () => {
          subscribers.delete(callback);
        };
      },
    };

    // Fall back to the backend model when the runtime lacks Web Speech synthesis, so read-aloud
    // still works.
    if (
      ttsEngine !== "system" ||
      !StudioSpeechSynthesisAdapter.systemVoicesSupported()
    ) {
      const generate =
        ttsEngine === "custom"
          ? generateCustomTtsAudio
          : generateStudioTtsAudio;
      const session = speakWithBackendAudio(generate, text, handleEnd, () => {
        if (res.status.type === "ended") return;
        // Notify subscribers of the async starting -> running transition; the adapter contract drives
        // UI state off these subscribe callbacks.
        res.status = { type: "running" };
        for (const handler of subscribers) handler();
      });
      cancelImpl = session.cancel;
      return res;
    }

    const utterance = createConfiguredUtterance(text);
    utterance.addEventListener("end", () => handleEnd("finished"));
    utterance.addEventListener("error", (e) => handleEnd("error", e.error));

    // Chrome silently drops speak() while another utterance is queued from a cancelled run;
    // clearing first keeps read-aloud deterministic.
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);
    res.status = { type: "running" };

    cancelImpl = () => {
      window.speechSynthesis.cancel();
      handleEnd("cancelled");
    };
    return res;
  }
}
