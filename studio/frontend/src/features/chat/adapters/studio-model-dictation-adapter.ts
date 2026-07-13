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

// Finer timeslice inside a segment so the buffer is ready the moment a segment
// is cut or the user stops.
const SEGMENT_TIMESLICE_MS = 250;
// Segment sizing for the background transcription pipeline. Segments are cut at
// natural pauses so word boundaries are preserved, but bounded so no single
// clip is too short to be accurate or too long to keep the tick snappy.
const MIN_SEGMENT_MS = 2500;
const MAX_SEGMENT_MS = 6000;
const SILENCE_CUT_MS = 350;
// Raw RMS (0..1) above which a frame counts as speech. Noise suppression keeps
// the room floor well below this.
const VOICE_RMS = 0.015;

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
 * Dictation via a loaded STT model, ChatGPT style. While the user talks the
 * audio is split at natural pauses and each chunk is transcribed in the
 * background, so when they confirm only the final short tail is left to
 * transcribe and the text appears almost immediately. Confirming keeps the
 * text; discarding throws it away. Stopping releases the mic immediately.
 * Works in any browser with MediaRecorder, including Firefox, which has no
 * Web Speech recognition.
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

    // Warm the model now, in parallel with recording, so the first transcription
    // is not delayed by a cold load. A no-op if it is already resident.
    void loadSttModel(useVoiceSettingsStore.getState().sttModel).catch(() => {});

    const speechStartCallbacks = new Set<() => void>();
    const speechEndCallbacks = new Set<
      (result: DictationAdapter.Result) => void
    >();
    const speechCallbacks = new Set<
      (result: DictationAdapter.Result) => void
    >();
    const endCallbacks = new Set<() => void>();

    let stream: MediaStream | null = null;
    let ended = false;
    let cancelled = false;
    let finalizing = false;
    const abortController = new AbortController();
    const mimeType = pickMimeType();
    // Web Audio graph for the waveform + voice-activity detection; torn down
    // with the session. onAudioFrame is wired to the VAD once segments start.
    let audioContext: AudioContext | null = null;
    let levelRaf = 0;
    let onAudioFrame: (rawRms: number, now: number) => void = () => {};

    const stopLevelMeter = () => {
      if (levelRaf) cancelAnimationFrame(levelRaf);
      levelRaf = 0;
      audioContext?.close().catch(() => {});
      audioContext = null;
      emitLevel(0);
    };

    // Tap the mic stream with an analyser: publish a smoothed RMS level for the
    // waveform and feed the raw level to the pause detector.
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
          onAudioFrame(rms, performance.now());
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

    // --- Background transcription pipeline ---------------------------------
    // Each recorded segment is a self-contained clip transcribed on its own.
    // Results are stored by segment index so the final text keeps its order.
    type Segment = {
      index: number;
      chunks: Blob[];
      startedAt: number;
      voiced: boolean;
      recorder: MediaRecorder;
    };
    const results: string[] = [];
    const queue: { index: number; blob: Blob }[] = [];
    let worker = false;
    let currentSeg: Segment | null = null;
    let segCounter = 0;
    let silenceMs = 0;
    let lastFrameAt = 0;
    let cutting = false;
    let finalCutDone = false;

    const buildTranscript = () =>
      results
        .filter((part) => part && part.trim())
        .join(" ")
        .trim();

    const finishSession = (
      reason: "stopped" | "cancelled" | "error",
      transcript?: string,
    ) => {
      if (ended) return;
      ended = true;
      stopLevelMeter();
      if (currentSeg && currentSeg.recorder.state !== "inactive") {
        try {
          currentSeg.recorder.stop();
        } catch {
          // ignore
        }
      }
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

    // Finish once the final segment has been cut and the queue has drained.
    const maybeComplete = () => {
      if (ended || cancelled || !finalizing || !finalCutDone) return;
      if (queue.length === 0 && !worker) {
        finishSession("stopped", buildTranscript());
      }
    };

    // Transcribe queued segments one at a time so the backend is never flooded.
    const processQueue = () => {
      if (worker || cancelled || ended) return;
      const item = queue.shift();
      if (!item) {
        maybeComplete();
        return;
      }
      worker = true;
      void (async () => {
        try {
          const text = await transcribeAudioBlob(
            item.blob,
            abortController.signal,
          );
          if (!cancelled) results[item.index] = text;
        } catch (error) {
          // Drop one segment rather than failing the whole dictation.
          if (!cancelled && !abortController.signal.aborted) {
            console.error("STT segment error:", error);
          }
        } finally {
          worker = false;
          processQueue();
        }
      })();
    };

    const enqueueSegment = (index: number, blob: Blob, voiced: boolean) => {
      if (voiced && blob.size > 0) {
        queue.push({ index, blob });
        processQueue();
      } else {
        results[index] = "";
        maybeComplete();
      }
    };

    // Start recording a fresh segment on the shared mic stream.
    const startSegment = () => {
      if (ended || cancelled || !stream) return;
      const seg: Segment = {
        index: segCounter++,
        chunks: [],
        startedAt: performance.now(),
        voiced: false,
        recorder: new MediaRecorder(stream, mimeType ? { mimeType } : undefined),
      };
      currentSeg = seg;
      silenceMs = 0;
      seg.recorder.addEventListener("dataavailable", (event) => {
        if (event.data.size > 0) seg.chunks.push(event.data);
      });
      seg.recorder.addEventListener("stop", () => {
        const blob = new Blob(seg.chunks, {
          type: seg.recorder.mimeType || "audio/webm",
        });
        enqueueSegment(seg.index, blob, seg.voiced);
      });
      seg.recorder.start(SEGMENT_TIMESLICE_MS);
    };

    // Close the current segment at a pause and immediately open the next, so
    // recording is continuous while each clip stays independently decodable.
    const cutSegment = () => {
      const seg = currentSeg;
      if (cutting || !seg || finalizing) return;
      cutting = true;
      const rec = seg.recorder;
      if (rec.state !== "inactive") {
        rec.addEventListener(
          "stop",
          () => {
            cutting = false;
          },
          { once: true },
        );
        try {
          rec.stop();
        } catch {
          cutting = false;
        }
      } else {
        cutting = false;
      }
      startSegment();
    };

    // Pause detector: mark voiced frames, and cut the segment after a short
    // silence once it is long enough, or force a cut if it runs too long.
    onAudioFrame = (rawRms, now) => {
      const seg = currentSeg;
      if (!seg || finalizing) {
        lastFrameAt = now;
        return;
      }
      if (rawRms > VOICE_RMS) {
        seg.voiced = true;
        silenceMs = 0;
      } else if (lastFrameAt) {
        silenceMs += now - lastFrameAt;
      }
      lastFrameAt = now;
      const duration = now - seg.startedAt;
      const pauseBreak =
        seg.voiced && duration > MIN_SEGMENT_MS && silenceMs > SILENCE_CUT_MS;
      if (!cutting && (pauseBreak || duration > MAX_SEGMENT_MS)) {
        cutSegment();
      }
    };

    const session: StudioDictationSession = {
      status: { type: "starting" },
      stop: async () => {
        if (!ended && !finalizing) {
          finalizing = true;
          const seg = currentSeg;
          // Cut the final segment (its buffer survives) so only the short tail
          // is left to transcribe, then release the mic immediately.
          if (seg && seg.recorder.state !== "inactive") {
            seg.recorder.addEventListener(
              "stop",
              () => {
                finalCutDone = true;
                maybeComplete();
              },
              { once: true },
            );
            try {
              seg.recorder.stop();
            } catch {
              finalCutDone = true;
              maybeComplete();
            }
          } else {
            finalCutDone = true;
            maybeComplete();
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
        startLevelMeter(stream);
        startSegment();
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
