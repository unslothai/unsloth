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

// Dictation is recorded in short, independent clips. Each clip is transcribed
// once and appended, so text appears while speaking without re-transcribing a
// growing buffer (which gets slower and slower and can stall the backend).
// The first clip is shorter so the first words show up sooner.
const SEGMENT_MS = 4000;
const FIRST_SEGMENT_MS = 2000;
// Safety net: stop always ends the session within this window, even if a
// transcription stalls, so the stop button can never get stuck.
const STOP_TIMEOUT_MS = 6000;

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
    let segmentTimer: ReturnType<typeof setTimeout> | null = null;
    let stopTimer: ReturnType<typeof setTimeout> | null = null;
    let segmentCount = 0;
    const queue: Blob[] = [];
    let transcribing = false;
    // Full dictation so far, for Recent dictations.
    let committed = "";
    let ended = false;
    let cancelled = false;
    let finalizing = false;
    const abortController = new AbortController();
    const mimeType = pickMimeType();
    let resolveEnded: (() => void) | null = null;
    const endedPromise = new Promise<void>((resolve) => {
      resolveEnded = resolve;
    });

    const clearTimers = () => {
      if (segmentTimer) clearTimeout(segmentTimer);
      if (stopTimer) clearTimeout(stopTimer);
      segmentTimer = null;
      stopTimer = null;
    };

    const finishSession = (reason: "stopped" | "cancelled" | "error") => {
      if (ended) return;
      ended = true;
      clearTimers();
      session.status = { type: "ended", reason };
      stopStream(stream);
      stream = null;
      if (reason !== "cancelled") {
        for (const callback of speechEndCallbacks) {
          callback({ transcript: committed });
        }
        if (committed) recordRecentDictation(committed);
      }
      for (const callback of endCallbacks) callback();
      resolveEnded?.();
    };

    // Finish once the last recorded segment has been transcribed.
    const maybeFinish = () => {
      if (
        finalizing &&
        !ended &&
        !transcribing &&
        queue.length === 0 &&
        (!recorder || recorder.state === "inactive")
      ) {
        finishSession("stopped");
      }
    };

    // Transcribe queued segments one at a time and append each result. One in
    // flight at a time keeps the backend from being flooded.
    const processQueue = () => {
      if (transcribing || ended) return;
      const blob = queue.shift();
      if (!blob) {
        maybeFinish();
        return;
      }
      transcribing = true;
      void (async () => {
        try {
          const text = await transcribeAudioBlob(blob, abortController.signal);
          if (!cancelled && text) {
            const corrected = applyDictationDictionary(text);
            committed = committed ? `${committed} ${corrected}` : corrected;
            for (const callback of speechCallbacks) {
              callback({ transcript: corrected, isFinal: true });
            }
          }
        } catch {
          // Drop a failed segment rather than stalling the whole session.
        } finally {
          transcribing = false;
          processQueue();
        }
      })();
    };

    // Record one clip, then chain the next until the user stops.
    const startSegment = () => {
      if (ended || cancelled || finalizing || !stream) return;
      const parts: Blob[] = [];
      const rec = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      recorder = rec;
      rec.addEventListener("dataavailable", (event) => {
        if (event.data.size > 0) parts.push(event.data);
      });
      rec.addEventListener("stop", () => {
        if (parts.length > 0) {
          queue.push(new Blob(parts, { type: rec.mimeType || "audio/webm" }));
          processQueue();
        }
        if (!ended && !cancelled && !finalizing && stream) {
          startSegment();
        } else {
          maybeFinish();
        }
      });
      rec.start();
      const duration = segmentCount === 0 ? FIRST_SEGMENT_MS : SEGMENT_MS;
      segmentCount += 1;
      segmentTimer = setTimeout(() => {
        if (rec.state !== "inactive") rec.stop();
      }, duration);
    };

    const session: StudioDictationSession = {
      status: { type: "starting" },
      stop: async () => {
        if (!ended && !finalizing) {
          finalizing = true;
          if (segmentTimer) clearTimeout(segmentTimer);
          segmentTimer = null;
          // Never let stop hang: end the session even if a segment stalls.
          stopTimer = setTimeout(() => finishSession("stopped"), STOP_TIMEOUT_MS);
          if (recorder && recorder.state !== "inactive") {
            recorder.stop();
          } else {
            maybeFinish();
          }
        }
        await endedPromise;
      },
      cancel: () => {
        if (ended) return;
        cancelled = true;
        finalizing = true;
        abortController.abort();
        if (recorder && recorder.state !== "inactive") {
          try {
            recorder.stop();
          } catch {
            // ignore
          }
        }
        finishSession("cancelled");
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
        if (ended || cancelled) {
          stopStream(stream);
          stream = null;
          return;
        }
        session.status = { type: "running" };
        for (const callback of speechStartCallbacks) callback();
        startSegment();
      } catch (error) {
        const message = isMissingDeviceError(error)
          ? "No microphone was found for dictation."
          : "Dictation could not access the microphone.";
        console.error("STT microphone error:", error);
        toast.error(message);
        finishSession("error");
      }
    })();

    return session;
  }
}
