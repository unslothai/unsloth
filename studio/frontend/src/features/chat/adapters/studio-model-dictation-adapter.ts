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

// MediaRecorder emits a chunk this often so the buffer is ready to send the
// moment the user stops.
const RECORD_TIMESLICE_MS = 1000;

// Live microphone level (0..1), published while recording so the composer can
// draw a waveform. One recording is active at a time, so a single module-level
// emitter is enough.
type LevelListener = (level: number) => void;
const levelListeners = new Set<LevelListener>();

/** Subscribe to the live mic level (0..1) during model dictation. */
export function subscribeDictationLevel(listener: LevelListener): () => void {
  levelListeners.add(listener);
  return () => {
    levelListeners.delete(listener);
  };
}

function emitLevel(level: number): void {
  for (const listener of levelListeners) listener(level);
}

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
 * Dictation via a loaded STT model. Records the microphone the whole time and
 * transcribes the clip once when the user stops, like the ChatGPT dictation
 * flow. Stopping releases the mic immediately. Works in any browser with
 * MediaRecorder, including Firefox, which has no Web Speech recognition.
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
    let finalizing = false;
    let transcribed = false;
    const abortController = new AbortController();
    const mimeType = pickMimeType();
    // Web Audio graph for the waveform; torn down with the session.
    let audioContext: AudioContext | null = null;
    let levelRaf = 0;

    const stopLevelMeter = () => {
      if (levelRaf) cancelAnimationFrame(levelRaf);
      levelRaf = 0;
      audioContext?.close().catch(() => {});
      audioContext = null;
      emitLevel(0);
    };

    // Tap the mic stream with an analyser and publish a smoothed RMS level so
    // the composer can animate a waveform while recording.
    const startLevelMeter = (source: MediaStream) => {
      try {
        const Ctx =
          window.AudioContext ||
          (window as unknown as { webkitAudioContext?: typeof AudioContext })
            .webkitAudioContext;
        if (!Ctx) return;
        audioContext = new Ctx();
        // Browsers can create the context suspended; resume so the analyser
        // actually receives samples.
        void audioContext.resume().catch(() => {});
        const node = audioContext.createMediaStreamSource(source);
        const analyser = audioContext.createAnalyser();
        analyser.fftSize = 512;
        node.connect(analyser);
        const data = new Uint8Array(analyser.frequencyBinCount);
        const tick = () => {
          analyser.getByteTimeDomainData(data);
          let sum = 0;
          for (let i = 0; i < data.length; i++) {
            const v = (data[i] - 128) / 128;
            sum += v * v;
          }
          const rms = Math.sqrt(sum / data.length);
          // Perceptual boost so quiet speech still moves the bars.
          emitLevel(Math.min(1, rms * 3.2));
          levelRaf = requestAnimationFrame(tick);
        };
        levelRaf = requestAnimationFrame(tick);
      } catch {
        // Waveform is cosmetic; ignore any Web Audio failure.
      }
    };
    let resolveEnded: (() => void) | null = null;
    const endedPromise = new Promise<void>((resolve) => {
      resolveEnded = resolve;
    });

    const finishSession = (
      reason: "stopped" | "cancelled" | "error",
      transcript?: string,
    ) => {
      if (ended) return;
      ended = true;
      stopLevelMeter();
      session.status = { type: "ended", reason };
      stopStream(stream);
      stream = null;
      const corrected = transcript ? applyDictationDictionary(transcript) : "";
      if (reason !== "cancelled" && corrected) {
        for (const callback of speechCallbacks) {
          callback({ transcript: corrected, isFinal: true });
        }
        recordRecentDictation(corrected);
      }
      for (const callback of speechEndCallbacks) {
        callback({ transcript: corrected });
      }
      for (const callback of endCallbacks) callback();
      resolveEnded?.();
    };

    // Transcribe the whole recording once, then end the session.
    const transcribeAndFinish = () => {
      if (transcribed || cancelled || ended) return;
      transcribed = true;
      const blob = new Blob(chunks, {
        type: recorder?.mimeType || "audio/webm",
      });
      if (blob.size === 0) {
        finishSession("stopped");
        return;
      }
      void (async () => {
        try {
          const text = await transcribeAudioBlob(blob, abortController.signal);
          if (cancelled) return;
          finishSession("stopped", text);
        } catch (error) {
          if (cancelled || abortController.signal.aborted) return;
          const message =
            error instanceof Error ? error.message : "Transcription failed.";
          console.error("STT transcription error:", error);
          toast.error(message);
          finishSession("error");
        }
      })();
    };

    const session: StudioDictationSession = {
      status: { type: "starting" },
      stop: async () => {
        if (!ended && !finalizing) {
          finalizing = true;
          const rec = recorder;
          // Flush the recording, then release the mic immediately so recording
          // visibly stops on the first click. The buffered audio survives.
          if (rec && rec.state !== "inactive") {
            rec.addEventListener("stop", transcribeAndFinish, { once: true });
            try {
              rec.stop();
            } catch {
              transcribeAndFinish();
            }
          } else {
            transcribeAndFinish();
          }
          stopStream(stream);
          stream = null;
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
        recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
        recorder.addEventListener("dataavailable", (event) => {
          if (event.data.size > 0) chunks.push(event.data);
        });
        recorder.start(RECORD_TIMESLICE_MS);
        startLevelMeter(stream);
        session.status = { type: "running" };
        for (const callback of speechStartCallbacks) callback();
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
