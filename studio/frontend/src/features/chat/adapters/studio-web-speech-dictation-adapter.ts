// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  applyDictationDictionary,
  recordRecentDictation,
  resolveDictationLanguage,
  useVoiceSettingsStore,
} from "@/features/settings/stores/voice-settings-store";
import type { DictationAdapter } from "@assistant-ui/react";
import { toast } from "sonner";

const getSpeechRecognitionAPI = ():
  | SpeechRecognitionConstructor
  | undefined => {
  if (typeof window === "undefined") return undefined;
  return window.SpeechRecognition ?? window.webkitSpeechRecognition;
};

const stopStream = (stream: MediaStream | null) => {
  stream?.getTracks().forEach((track) => track.stop());
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

export const describeSpeechError = (error: string, message?: string): string => {
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

  constructor(
    options: {
      language?: string;
      continuous?: boolean;
      interimResults?: boolean;
    } = {},
  ) {
    // Resolved from Voice settings at listen() time unless overridden.
    this.language = options.language;
    this.continuous = options.continuous ?? true;
    this.interimResults = options.interimResults ?? true;
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

    let stream: MediaStream | null = null;
    let finalTranscript = "";
    let ended = false;
    let started = false;
    let resolveEnded: (() => void) | null = null;
    const endedPromise = new Promise<void>((resolve) => {
      resolveEnded = resolve;
    });

    const session: DictationAdapter.Session = {
      status: { type: "starting" },

      stop: async () => {
        if (!ended && started) {
          recognition.stop();
        } else if (!ended) {
          finish("stopped");
        }
        await endedPromise;
      },

      cancel: () => {
        if (!ended && started) {
          recognition.abort();
        } else if (!ended) {
          finish("cancelled");
        }
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
    };

    const finish = (reason: "stopped" | "cancelled" | "error") => {
      if (ended) return;
      ended = true;
      session.status = { type: "ended", reason };
      stopStream(stream);
      stream = null;
      if (finalTranscript) {
        for (const callback of speechEndCallbacks) {
          callback({ transcript: finalTranscript });
        }
        if (reason !== "cancelled") {
          recordRecentDictation(finalTranscript);
        }
        finalTranscript = "";
      }
      resolveEnded?.();
    };

    recognition.addEventListener("start", () => {
      session.status = { type: "running" };
    });

    recognition.addEventListener("speechstart", () => {
      for (const callback of speechStartCallbacks) callback();
    });

    recognition.addEventListener("result", (event) => {
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
          const corrected = applyDictationDictionary(transcript);
          // Join final chunks with a single space so recorded transcripts do
          // not merge words when a browser omits leading whitespace.
          const trimmed = corrected.trim();
          if (trimmed) {
            finalTranscript = finalTranscript
              ? `${finalTranscript} ${trimmed}`
              : trimmed;
          }
          for (const callback of speechCallbacks) {
            callback({ transcript: corrected, isFinal: true });
          }
        } else {
          for (const callback of speechCallbacks) {
            callback({ transcript, isFinal: false });
          }
        }
      }
    });

    recognition.addEventListener("end", () => {
      finish("stopped");
    });

    recognition.addEventListener("error", (event) => {
      const errorEvent = event as SpeechRecognitionErrorEvent;
      if (errorEvent.error === "aborted") {
        finish("cancelled");
        return;
      }
      const description = describeSpeechError(
        errorEvent.error,
        errorEvent.message,
      );
      console.error("Dictation error:", errorEvent.error, errorEvent.message);
      toast.error(description);
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
