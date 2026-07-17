// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { useVoiceSettingsStore } from "@/features/settings/stores/voice-settings-store";
import { toast } from "@/lib/toast";
import type { SpeechSynthesisAdapter } from "@assistant-ui/react";

/** Voice for a stored voiceURI; undefined lets the browser pick. */
export function findTtsVoice(
  voiceURI: string,
): SpeechSynthesisVoice | undefined {
  if (typeof window === "undefined" || !window.speechSynthesis) {
    return undefined;
  }
  if (!voiceURI || voiceURI === "default") return undefined;
  return window.speechSynthesis
    .getVoices()
    .find((voice) => voice.voiceURI === voiceURI);
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

function voiceQualityScore(voice: SpeechSynthesisVoice): number {
  const name = voice.name.toLowerCase();
  let score = 0;
  if (name.includes("premium")) score += 8;
  if (name.includes("enhanced")) score += 7;
  if (name.includes("natural") || name.includes("neural")) score += 6;
  if (name.includes("siri")) score += 6;
  if (name.includes("google")) score += 5;
  if (name.includes("microsoft")) score += 4;
  if (voice.default) score += 3;
  return score;
}

function langBase(tag: string): string {
  return tag.toLowerCase().split(/[-_]/)[0] ?? "";
}

const MAX_CURATED_VOICES = 20;

/**
 * Keep the best, most relevant voices: drop low-quality ones, keep English,
 * the browser language, and the dictation language, rank by quality hints,
 * and cap the list. The selected voice is always kept.
 */
export function curateSystemVoices(
  voices: SpeechSynthesisVoice[],
  selectedVoiceURI?: string,
): SpeechSynthesisVoice[] {
  const { dictationLanguage } = useVoiceSettingsStore.getState();
  const wantedLangs = new Set<string>(["en"]);
  if (typeof navigator !== "undefined" && navigator.language) {
    wantedLangs.add(langBase(navigator.language));
  }
  if (dictationLanguage && dictationLanguage !== "auto") {
    wantedLangs.add(langBase(dictationLanguage));
  }

  // WebKit and Linux engines report voices with empty or duplicate voiceURIs;
  // drop them so the Radix Select never gets an empty or colliding value.
  const seenVoiceURIs = new Set<string>();
  const kept = voices.filter((voice) => {
    if (!voice.voiceURI || seenVoiceURIs.has(voice.voiceURI)) return false;
    seenVoiceURIs.add(voice.voiceURI);
    if (LOW_QUALITY_VOICE_NAMES.has(voiceBaseName(voice))) return false;
    return wantedLangs.has(langBase(voice.lang));
  });

  kept.sort((a, b) => {
    const scoreDiff = voiceQualityScore(b) - voiceQualityScore(a);
    if (scoreDiff !== 0) return scoreDiff;
    return a.name.localeCompare(b.name);
  });

  const curated = kept.slice(0, MAX_CURATED_VOICES);
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

/** Build an utterance from the current Voice settings. */
export function createConfiguredUtterance(
  text: string,
): SpeechSynthesisUtterance {
  const { ttsVoiceURI, ttsRate, ttsPitch, ttsVolume } =
    useVoiceSettingsStore.getState();
  const utterance = new SpeechSynthesisUtterance(text);
  const voice = findTtsVoice(ttsVoiceURI);
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

function speakWithStudioModel(
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
  let cancelled = false;

  // Release the element and its multi-MB WAV data URL as soon as playback ends.
  const cleanup = () => {
    if (audio) {
      audio.pause();
      audio.removeAttribute("src");
      audio = null;
    }
  };

  void (async () => {
    try {
      const url = await generateStudioTtsAudio(text, controller.signal);
      if (cancelled) return;
      audio = new Audio(url);
      audio.playbackRate = ttsRate;
      audio.volume = ttsVolume;
      // Some browsers reset playbackRate to 1 once the source loads; reapply
      // it on loadedmetadata so the speed setting reliably takes effect.
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

/**
 * Text-to-speech for assistant messages. Reads Voice settings at speak time.
 * Engines: "system" (speechSynthesis) or "studio" (loaded TTS audio model).
 */
export class StudioSpeechSynthesisAdapter implements SpeechSynthesisAdapter {
  /** Web Speech synthesis, used by the "system" engine. */
  static systemVoicesSupported(): boolean {
    return (
      typeof window !== "undefined" &&
      "speechSynthesis" in window &&
      typeof window.SpeechSynthesisUtterance !== "undefined"
    );
  }

  // The "studio" engine only needs fetch + Audio playback, so a WebView
  // without Web Speech synthesis can still read aloud through the backend.
  static isSupported(): boolean {
    return (
      StudioSpeechSynthesisAdapter.systemVoicesSupported() ||
      (typeof window !== "undefined" && typeof window.Audio !== "undefined")
    );
  }

  speak(text: string): SpeechSynthesisAdapter.Utterance {
    const subscribers = new Set<() => void>();

    const handleEnd = (
      reason: "finished" | "error" | "cancelled",
      error?: unknown,
    ) => {
      if (res.status.type === "ended") return;
      // Surface genuine read-aloud failures; a cancelled/interrupted utterance
      // is a normal stop, not an error, and must not toast.
      if (reason === "error" && error !== "interrupted" && error !== "canceled") {
        toast.error(error instanceof Error ? error.message : "Read aloud failed.");
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

    // Fall back to the backend model when the runtime lacks Web Speech
    // synthesis (e.g. an audio-only WebView), so read-aloud still works.
    if (
      ttsEngine === "studio" ||
      !StudioSpeechSynthesisAdapter.systemVoicesSupported()
    ) {
      const session = speakWithStudioModel(text, handleEnd, () => {
        if (res.status.type === "ended") return;
        // Notify subscribers of the async starting -> running transition;
        // the adapter contract drives UI state off these subscribe callbacks.
        res.status = { type: "running" };
        for (const handler of subscribers) handler();
      });
      cancelImpl = session.cancel;
      return res;
    }

    const utterance = createConfiguredUtterance(text);
    utterance.addEventListener("end", () => handleEnd("finished"));
    utterance.addEventListener("error", (e) => handleEnd("error", e.error));

    // Chrome silently drops speak() while another utterance is queued from a
    // cancelled run; clearing first keeps read-aloud deterministic.
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
