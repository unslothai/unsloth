// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useSettingsDialogStore } from "@/features/settings/stores/settings-dialog-store";
import {
  applyDictationDictionary,
  recordRecentDictation,
  resolveDictationLanguage,
  useVoiceSettingsStore,
} from "@/features/settings/stores/voice-settings-store";
import type { DictationAdapter } from "@assistant-ui/react";
import { toast } from "sonner";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import { startDictationLevelMeter } from "./dictation-level";

/** Chat open while dictating, so the saved dictation can link back to it. */
export function activeDictationChatId(): string | undefined {
  return useChatRuntimeStore.getState().activeThreadId ?? undefined;
}

/**
 * Resolve the chat a saved dictation links to. undefined falls back to the
 * active single chat; null (composers without one, e.g. Compare) means none.
 */
export function resolveDictationChatId(
  chatId: string | null | undefined,
): string | undefined {
  if (chatId === undefined) return activeDictationChatId();
  return chatId ?? undefined;
}

// Reused id so repeated network failures replace, not stack, the same toast.
const NETWORK_TOAST_ID = "dictation-network-offline";

// Give the browser a short chance to turn its latest interim hypothesis into a
// final result. If it does not, promote that already-produced text instead of
// leaving the composer behind a long service-finalization spinner.
const STOP_FINALIZATION_GRACE_MS = 350;

/**
 * A dictation session with an extra onEnd hook (not part of the assistant-ui
 * interface) so non-runtime callers can reset when the session ends by itself.
 */
export type StudioDictationSession = DictationAdapter.Session & {
  onEnd?: (callback: () => void) => () => void;
};

const getSpeechRecognitionAPI = ():
  | SpeechRecognitionConstructor
  | undefined => {
  if (typeof window === "undefined") return undefined;
  return window.SpeechRecognition ?? window.webkitSpeechRecognition;
};

const stopStream = (stream: MediaStream | null) => {
  for (const track of stream?.getTracks() ?? []) {
    track.stop();
  }
};

const mediaErrorName = (error: unknown): unknown =>
  error && typeof error === "object" && "name" in error
    ? (error as { name?: unknown }).name
    : undefined;

/** True for getUserMedia errors meaning the requested device is gone. */
export const isMissingDeviceError = (error: unknown): boolean => {
  const name = mediaErrorName(error);
  return name === "OverconstrainedError" || name === "NotFoundError";
};

export const describeMediaError = (error: unknown): string => {
  const name = mediaErrorName(error);
  if (name === "NotAllowedError" || name === "SecurityError") {
    return "Microphone access is blocked. Allow microphone access for this Unsloth page, then try again.";
  }
  if (name === "NotFoundError" || name === "OverconstrainedError") {
    return "No microphone was found for dictation.";
  }
  if (name === "NotReadableError" || name === "AbortError") {
    return "The microphone is already in use or unavailable.";
  }
  return error instanceof Error && error.message
    ? error.message
    : "Dictation could not access the microphone.";
};

export const describeSpeechError = (
  error: string,
  message?: string,
): string => {
  if (error === "not-allowed") {
    return "Speech recognition was blocked by the browser. Check microphone permissions for this Unsloth page.";
  }
  if (error === "service-not-allowed") {
    return "Speech recognition is blocked by the browser speech service.";
  }
  if (error === "network") {
    return "Speech recognition could not reach the browser speech service.";
  }
  if (error === "language-not-supported") {
    return "Speech recognition does not support the current language.";
  }
  return message || `Speech recognition failed: ${error}`;
};

export class StudioWebSpeechDictationAdapter implements DictationAdapter {
  private readonly language: string | undefined;
  private readonly continuous: boolean;
  private readonly interimResults: boolean;
  private readonly chatId: string | null | undefined;

  constructor(
    options: {
      language?: string;
      continuous?: boolean;
      interimResults?: boolean;
      chatId?: string | null;
    } = {},
  ) {
    // Resolved from Voice settings at listen() time unless overridden.
    this.language = options.language;
    this.continuous = options.continuous ?? true;
    this.interimResults = options.interimResults ?? true;
    this.chatId = options.chatId;
  }

  static isSupported(): boolean {
    return (
      typeof window !== "undefined" &&
      window.isSecureContext &&
      getSpeechRecognitionAPI() !== undefined &&
      navigator.mediaDevices?.getUserMedia !== undefined
    );
  }

  listen(): DictationAdapter.Session {
    const SpeechRecognitionAPI = getSpeechRecognitionAPI();
    if (!SpeechRecognitionAPI || !navigator.mediaDevices?.getUserMedia) {
      throw new Error("Speech recognition is not supported in this browser.");
    }

    const recognition = new SpeechRecognitionAPI();
    recognition.lang = this.language ?? resolveDictationLanguage();
    recognition.continuous = this.continuous;
    recognition.interimResults = this.interimResults;

    const speechStartCallbacks = new Set<() => void>();
    const speechEndCallbacks = new Set<
      (result: DictationAdapter.Result) => void
    >();
    const speechCallbacks = new Set<
      (result: DictationAdapter.Result) => void
    >();
    const endCallbacks = new Set<() => void>();

    let stream: MediaStream | null = null;
    let finalTranscript = "";
    let ended = false;
    let started = false;
    let stopping = false;
    let stopRequestedAt = 0;
    let stopFallbackTimer = 0;
    let stopLevelMeter = () => {
      // Replaced after microphone access succeeds.
    };
    const interimParts = new Map<number, string>();
    let resolveEnded: (() => void) | null = null;
    const endedPromise = new Promise<void>((resolve) => {
      resolveEnded = resolve;
    });

    const session: StudioDictationSession = {
      status: { type: "starting" },

      stop: async () => {
        if (!ended && started) {
          stopping = true;
          stopRequestedAt = performance.now();
          recognition.stop();
          // Ending the supplied track immediately gives the browser an audio
          // endpoint to finalize and releases the microphone without waiting
          // for its remote speech service.
          stopLevelMeter();
          stopStream(stream);
          stream = null;
          scheduleStopFallback();
        } else if (!ended) {
          finish("stopped");
        }
        await endedPromise;
      },

      cancel: () => {
        if (ended) return;
        if (started) recognition.abort();
        finish("cancelled");
      },

      onSpeechStart: (callback) => {
        speechStartCallbacks.add(callback);
        return () => {
          speechStartCallbacks.delete(callback);
        };
      },

      onSpeechEnd: (callback) => {
        speechEndCallbacks.add(callback);
        return () => {
          speechEndCallbacks.delete(callback);
        };
      },

      onSpeech: (callback) => {
        speechCallbacks.add(callback);
        return () => {
          speechCallbacks.delete(callback);
        };
      },

      // Extra to the DictationAdapter interface: lets callers reset UI when the
      // session ends on its own (silence, error), not just via stop().
      onEnd: (callback: () => void) => {
        endCallbacks.add(callback);
        return () => {
          endCallbacks.delete(callback);
        };
      },
    };

    const currentInterimTranscript = () =>
      [...interimParts.entries()]
        .sort(([a], [b]) => a - b)
        .map(([, transcript]) => transcript.trim())
        .filter(Boolean)
        .join(" ");

    const promoteInterim = () => {
      const interim = currentInterimTranscript();
      interimParts.clear();
      if (!interim) return false;
      const corrected = applyDictationDictionary(interim).trim();
      if (!corrected) return false;
      finalTranscript = finalTranscript
        ? `${finalTranscript} ${corrected}`
        : corrected;
      return true;
    };

    function scheduleStopFallback(): void {
      if (!stopping || ended || stopFallbackTimer) return;
      const elapsed = performance.now() - stopRequestedAt;
      const delay = Math.max(0, STOP_FINALIZATION_GRACE_MS - elapsed);
      stopFallbackTimer = window.setTimeout(() => {
        stopFallbackTimer = 0;
        if (!ended) {
          promoteInterim();
          finish("stopped");
          recognition.abort();
        }
      }, delay);
    }

    const finish = (reason: "stopped" | "cancelled" | "error") => {
      if (ended) return;
      ended = true;
      stopping = false;
      if (stopFallbackTimer) window.clearTimeout(stopFallbackTimer);
      stopFallbackTimer = 0;
      session.status = { type: "ended", reason };
      stopLevelMeter();
      stopStream(stream);
      stream = null;
      const transcript = reason === "cancelled" ? "" : finalTranscript;
      if (transcript) {
        for (const callback of speechCallbacks) {
          callback({ transcript, isFinal: true });
        }
        recordRecentDictation(transcript, resolveDictationChatId(this.chatId));
      }
      // assistant-ui uses this standard lifecycle callback to leave dictation
      // mode. It is required even for silence and cancelled recordings.
      for (const callback of speechEndCallbacks) {
        callback({ transcript });
      }
      finalTranscript = "";
      interimParts.clear();
      for (const callback of endCallbacks) callback();
      resolveEnded?.();
    };

    recognition.addEventListener("start", () => {
      session.status = { type: "running" };
    });

    recognition.addEventListener("speechstart", () => {
      for (const callback of speechStartCallbacks) callback();
    });

    recognition.addEventListener("result", (event) => {
      if (ended) return;
      const speechEvent = event as SpeechRecognitionEvent;
      for (
        let i = speechEvent.resultIndex;
        i < speechEvent.results.length;
        i++
      ) {
        const result = speechEvent.results[i];
        if (!result) continue;
        const transcript = result[0]?.transcript ?? "";
        if (result.isFinal) {
          interimParts.delete(i);
          const corrected = applyDictationDictionary(transcript);
          // Join final chunks with a single space so recorded transcripts do
          // not merge words when a browser omits leading whitespace.
          const trimmed = corrected.trim();
          if (trimmed) {
            finalTranscript = finalTranscript
              ? `${finalTranscript} ${trimmed}`
              : trimmed;
          }
        } else {
          interimParts.set(i, transcript);
        }
      }
      const interim = currentInterimTranscript();
      if (interim) {
        scheduleStopFallback();
      }
    });

    recognition.addEventListener("end", () => {
      if (ended) {
        return;
      }
      promoteInterim();
      finish("stopped");
    });

    recognition.addEventListener("error", (event) => {
      if (ended) return;
      const errorEvent = event as SpeechRecognitionErrorEvent;
      if (errorEvent.error === "aborted") {
        if (stopping) {
          promoteInterim();
          finish("stopped");
        } else {
          finish("cancelled");
        }
        return;
      }
      const description = describeSpeechError(
        errorEvent.error,
        errorEvent.message,
      );
      console.error("Dictation error:", errorEvent.error, errorEvent.message);
      if (errorEvent.error === "network") {
        // Browser dictation can't reach the online speech service. Point the
        // user to the offline local engine; the toast opens Voice settings.
        toast.error("No internet connection", {
          id: NETWORK_TOAST_ID,
          description:
            "Browser dictation needs the online speech service. Switch to Local in Voice settings and pick a local STT model to dictate offline.",
          action: {
            label: "Open Voice settings",
            onClick: () =>
              useSettingsDialogStore.getState().openDialog("voice"),
          },
        });
      } else {
        toast.error(description);
      }
      finish("error");
    });

    void (async () => {
      try {
        const { micDeviceId } = useVoiceSettingsStore.getState();
        const baseAudio: MediaTrackConstraints = {
          echoCancellation: true,
          noiseSuppression: true,
        };
        try {
          stream = await navigator.mediaDevices.getUserMedia({
            audio:
              micDeviceId && micDeviceId !== "default"
                ? { ...baseAudio, deviceId: { exact: micDeviceId } }
                : baseAudio,
          });
        } catch (error) {
          // Saved mic may be unplugged; fall back to the default device.
          // Firefox and WebKit throw OverconstrainedError objects that are
          // not DOMException instances, so match on the error name.
          if (micDeviceId !== "default" && isMissingDeviceError(error)) {
            stream = await navigator.mediaDevices.getUserMedia({
              audio: baseAudio,
            });
          } else {
            throw error;
          }
        }
        if (ended) {
          stopStream(stream);
          stream = null;
          return;
        }
        const audioTrack = stream.getAudioTracks()[0];
        if (!audioTrack || audioTrack.readyState !== "live") {
          throw new DOMException(
            "No live microphone track is available.",
            "NotFoundError",
          );
        }
        stopLevelMeter = startDictationLevelMeter(stream);
        try {
          recognition.start(audioTrack);
        } catch (error) {
          // Older engines expose only start(); retry without the experimental
          // track overload. Recognition then captures from the default device,
          // so release the selected-device stream instead of holding it open.
          console.debug(
            "Dictation start(audioTrack) failed; retrying start().",
            error,
          );
          stopLevelMeter();
          stopStream(stream);
          stream = null;
          recognition.start();
        }
        started = true;
      } catch (error) {
        const description = describeMediaError(error);
        console.error("Dictation microphone error:", error);
        toast.error(description);
        finish("error");
      }
    })();

    return session;
  }
}
