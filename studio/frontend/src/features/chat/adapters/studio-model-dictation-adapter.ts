// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import {
  applyDictationDictionary,
  recordRecentDictation,
  useVoiceSettingsStore,
} from "@/features/settings/stores/voice-settings-store";
import type { DictationAdapter } from "@assistant-ui/react";
import { toast } from "sonner";
import {
  type StudioDictationSession,
  isMissingDeviceError,
} from "./studio-web-speech-dictation-adapter";

// MediaRecorder emits a chunk this often; the growing buffer is re-transcribed
// for a live preview so text appears while the user is still speaking.
const STREAM_TIMESLICE_MS = 1000;
// Do not start a new interim pass more often than this, and only one at a time.
const MIN_INTERIM_INTERVAL_MS = 900;

// Prefer Opus (small, widely supported); fall back to whatever the browser
// records. The backend decodes any of these with PyAV.
const PREFERRED_MIME_TYPES = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/ogg;codecs=opus",
  "audio/mp4",
];

function pickMimeType(): string | undefined {
  if (typeof MediaRecorder === "undefined") return undefined;
  for (const type of PREFERRED_MIME_TYPES) {
    if (MediaRecorder.isTypeSupported(type)) return type;
  }
  return undefined;
}

const stopStream = (stream: MediaStream | null) => {
  stream?.getTracks().forEach((track) => track.stop());
};

async function blobToBase64(blob: Blob): Promise<string> {
  const buffer = await blob.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  let binary = "";
  const chunk = 0x8000; // avoid call-stack limits on large recordings
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

/** POST audio to the STT sidecar and return the transcript. */
export async function transcribeAudioBlob(
  blob: Blob,
  signal?: AbortSignal,
  interim = false,
): Promise<string> {
  const { sttModel, dictationLanguage } = useVoiceSettingsStore.getState();
  const audio = await blobToBase64(blob);
  const response = await authFetch("/api/inference/audio/transcribe", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      audio,
      model: sttModel,
      language: dictationLanguage,
      interim,
    }),
    signal,
  });
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as {
      detail?: string;
    } | null;
    const detail = body?.detail ?? `HTTP ${response.status}`;
    if (response.status === 501) {
      throw new Error(
        "Speech-to-text is not available on this server. Run `unsloth studio update` to install it.",
      );
    }
    throw new Error(detail);
  }
  const data = (await response.json()) as { text?: string };
  return (data.text ?? "").trim();
}

export interface SttStatus {
  available: boolean;
  loaded_model: string | null;
  loading: boolean;
  device: string | null;
  default_model: string;
  models: string[];
}

/** Report whether STT is installed and which model, if any, is warm. */
export async function fetchSttStatus(): Promise<SttStatus> {
  const response = await authFetch("/api/inference/audio/stt/status");
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return (await response.json()) as SttStatus;
}

/** Warm the STT model so the first dictation is not delayed by a load. */
export async function loadSttModel(model: string): Promise<void> {
  const response = await authFetch("/api/inference/audio/stt/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model }),
  });
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as {
      detail?: string;
    } | null;
    throw new Error(body?.detail ?? `HTTP ${response.status}`);
  }
}

/**
 * Dictation via a loaded STT model. Records the microphone, then transcribes
 * the whole clip through the backend when the user stops. Works in any browser
 * with MediaRecorder, including Firefox, which has no Web Speech recognition.
 */
export class StudioModelDictationAdapter implements DictationAdapter {
  static isSupported(): boolean {
    return (
      typeof window !== "undefined" &&
      window.isSecureContext &&
      typeof MediaRecorder !== "undefined" &&
      navigator.mediaDevices?.getUserMedia !== undefined
    );
  }

  listen(): DictationAdapter.Session {
    if (!StudioModelDictationAdapter.isSupported()) {
      throw new Error("Recording is not supported in this browser.");
    }

    const speechStartCallbacks = new Set<() => void>();
    const speechEndCallbacks = new Set<
      (result: DictationAdapter.Result) => void
    >();
    const speechCallbacks = new Set<
      (result: DictationAdapter.Result) => void
    >();
    const endCallbacks = new Set<() => void>();

    let stream: MediaStream | null = null;
    let recorder: MediaRecorder | null = null;
    const chunks: Blob[] = [];
    let ended = false;
    let cancelled = false;
    // Final transcription: aborted only when the whole session is cancelled.
    const abortController = new AbortController();
    // Live-preview transcriptions: also aborted when recording stops.
    const interimAbort = new AbortController();
    let interimBusy = false;
    let lastInterimAt = 0;
    let resolveEnded: (() => void) | null = null;
    const endedPromise = new Promise<void>((resolve) => {
      resolveEnded = resolve;
    });

    const session: StudioDictationSession = {
      status: { type: "starting" },
      stop: async () => {
        if (!ended && recorder && recorder.state !== "inactive") {
          recorder.stop();
        } else if (!ended) {
          finish("stopped");
        }
        await endedPromise;
      },
      cancel: () => {
        cancelled = true;
        // Abort any in-flight transcription so a late response is discarded.
        interimAbort.abort();
        abortController.abort();
        if (!ended && recorder && recorder.state !== "inactive") {
          recorder.stop();
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
      onEnd: (callback: () => void) => {
        endCallbacks.add(callback);
        return () => {
          endCallbacks.delete(callback);
        };
      },
    };

    const finish = (reason: "stopped" | "cancelled" | "error") => {
      if (ended) return;
      ended = true;
      session.status = { type: "ended", reason };
      stopStream(stream);
      stream = null;
      for (const callback of endCallbacks) callback();
      resolveEnded?.();
    };

    const emitTranscript = (transcript: string) => {
      const corrected = applyDictationDictionary(transcript);
      if (!corrected) return;
      for (const callback of speechCallbacks) {
        callback({ transcript: corrected, isFinal: true });
      }
      for (const callback of speechEndCallbacks) {
        callback({ transcript: corrected });
      }
      recordRecentDictation(corrected);
    };

    // Re-transcribe the audio so far (fast pass) and emit it as a live preview.
    // Partial recordings decode fine; Whisper refines earlier words as more
    // audio arrives, and the final pass replaces the preview on stop.
    const runInterim = () => {
      if (ended || cancelled || interimBusy || chunks.length === 0) return;
      const now = Date.now();
      if (now - lastInterimAt < MIN_INTERIM_INTERVAL_MS) return;
      interimBusy = true;
      lastInterimAt = now;
      const type = recorder?.mimeType || "audio/webm";
      const blob = new Blob(chunks, { type });
      void (async () => {
        try {
          const transcript = await transcribeAudioBlob(
            blob,
            interimAbort.signal,
            true,
          );
          if (ended || cancelled || !transcript) return;
          const corrected = applyDictationDictionary(transcript);
          for (const callback of speechCallbacks) {
            callback({ transcript: corrected, isFinal: false });
          }
        } catch {
          // Interim failures (aborts, blips) are ignored; the final pass runs.
        } finally {
          interimBusy = false;
        }
      })();
    };

    const handleRecorderStop = () => {
      // Stop live previews; the accurate final pass follows.
      interimAbort.abort();
      if (cancelled) {
        finish("cancelled");
        return;
      }
      const type = recorder?.mimeType || "audio/webm";
      const blob = new Blob(chunks, { type });
      if (blob.size === 0) {
        finish("stopped");
        return;
      }
      void (async () => {
        try {
          const transcript = await transcribeAudioBlob(
            blob,
            abortController.signal,
          );
          if (cancelled) return;
          if (transcript) emitTranscript(transcript);
          finish("stopped");
        } catch (error) {
          // A cancel aborts the fetch; swallow that without an error toast.
          if (cancelled || abortController.signal.aborted) return;
          const message =
            error instanceof Error ? error.message : "Transcription failed.";
          console.error("STT transcription error:", error);
          toast.error(message);
          finish("error");
        }
      })();
    };

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
        const mimeType = pickMimeType();
        recorder = new MediaRecorder(
          stream,
          mimeType ? { mimeType } : undefined,
        );
        recorder.addEventListener("dataavailable", (event) => {
          if (event.data.size > 0) chunks.push(event.data);
          runInterim();
        });
        recorder.addEventListener("stop", handleRecorderStop);
        // Timeslice so chunks arrive during recording, enabling live previews.
        recorder.start(STREAM_TIMESLICE_MS);
        session.status = { type: "running" };
        for (const callback of speechStartCallbacks) callback();
      } catch (error) {
        const message = isMissingDeviceError(error)
          ? "No microphone was found for dictation."
          : "Dictation could not access the microphone.";
        console.error("STT microphone error:", error);
        toast.error(message);
        finish("error");
      }
    })();

    return session;
  }
}
