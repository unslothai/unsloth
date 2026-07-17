// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { hubTokenHeader } from "@/features/hub/lib/hub-token-header";
import {
  applyDictationDictionary,
  isCuratedSttModel,
  recordRecentDictation,
  useVoiceSettingsStore,
} from "@/features/settings/stores/voice-settings-store";
import type { DictationAdapter } from "@assistant-ui/react";
import { toast } from "sonner";
import { startDictationLevelMeter } from "./dictation-level";
import {
  type StudioDictationSession,
  activeDictationChatId,
  isMissingDeviceError,
} from "./studio-web-speech-dictation-adapter";

// Finer timeslice inside a segment so the buffer is ready the moment a segment
// is cut or the user stops.
const SEGMENT_TIMESLICE_MS = 250;
// Whisper pads input to 30 seconds. Keep short dictation in one clip. For long
// dictation, cut at the first pause after 20s or before the 30s boundary.
const MIN_SEGMENT_MS = 20_000;
const MAX_SEGMENT_MS = 28_000;
const SILENCE_CUT_MS = 280;
// Raw RMS (0..1) above which a frame counts as speech. Noise suppression keeps
// the room floor well below this.
const VOICE_RMS = 0.015;

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
  for (const track of stream?.getTracks() ?? []) {
    track.stop();
  }
};

/** Backend STT engine, decided by the selected model: the curated ids run
 * GGML checkpoints through whisper.cpp; a custom Hugging Face repository is
 * a safetensors checkpoint and runs through Transformers. */
export type SttEngine = "transformers" | "gguf";

export function sttEngineFor(model: string): SttEngine {
  return isCuratedSttModel(model) ? "gguf" : "transformers";
}

/** POST audio to the STT sidecar and return the transcript. */
export async function transcribeAudioBlob(
  blob: Blob,
  options: {
    model?: string;
    language?: string;
    engine?: SttEngine;
    signal?: AbortSignal;
  } = {},
): Promise<string> {
  const settings = useVoiceSettingsStore.getState();
  const model = options.model ?? settings.sttModel;
  const language = options.language ?? settings.dictationLanguage;
  const engine = options.engine ?? sttEngineFor(model);
  const params = new URLSearchParams({ model, fast: "true", engine });
  if (language) params.set("language", language);
  const response = await authFetch(
    `/api/inference/audio/transcribe/raw?${params.toString()}`,
    {
      method: "POST",
      headers: { "Content-Type": blob.type || "application/octet-stream" },
      body: blob,
      signal: options.signal,
    },
  );
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

export interface SttDownloadStatus {
  downloading: boolean;
  model: string | null;
  error: string | null;
  bytes_total: number | null;
  bytes_done: number | null;
}

export interface SttEngineStatus {
  available: boolean;
  loaded_model: string | null;
  loading: boolean;
  device: string | null;
  keep_alive_seconds: number;
  default_model: string;
  models: string[];
  downloaded_models: string[];
  download: SttDownloadStatus;
}

export interface SttStatus {
  available: boolean;
  loaded_model: string | null;
  loading: boolean;
  device: string | null;
  keep_alive_seconds: number;
  default_model: string;
  models: string[];
  /** Per-engine state; absent on servers predating the engine split. */
  transformers?: SttEngineStatus;
  gguf?: SttEngineStatus;
}

// Keep load and unload requests ordered. In particular, a new recording must
// not race an unload still finishing for the previous recording.
let sttLifecycle: Promise<void> = Promise.resolve();

function queueSttLifecycle(operation: () => Promise<void>): Promise<void> {
  const result = sttLifecycle.catch(() => {}).then(operation);
  sttLifecycle = result.catch(() => {});
  return result;
}

/** Report whether STT is installed and which model, if any, is resident.
 * Passing a model extends the downloaded check to custom repositories. */
export async function fetchSttStatus(
  refreshKey?: number,
  model?: string,
): Promise<SttStatus> {
  const params = new URLSearchParams();
  if (refreshKey !== undefined) params.set("refresh", String(refreshKey));
  if (model) params.set("model", model);
  const query = params.toString();
  const response = await authFetch(
    `/api/inference/audio/stt/status${query ? `?${query}` : ""}`,
  );
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return (await response.json()) as SttStatus;
}

/** Verify a custom Hub repository is a Transformers Whisper checkpoint. */
export async function validateSttModel(
  model: string,
  hfToken?: string,
): Promise<void> {
  const response = await authFetch("/api/inference/audio/stt/validate", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...hubTokenHeader(hfToken),
    },
    body: JSON.stringify({ model }),
  });
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as {
      detail?: string;
    } | null;
    throw new Error(body?.detail ?? `HTTP ${response.status}`);
  }
}

/** Load a selected model that is already downloaded. */
export function loadSttModel(model: string, engine?: SttEngine): Promise<void> {
  const resolvedEngine = engine ?? sttEngineFor(model);
  return queueSttLifecycle(async () => {
    const response = await authFetch("/api/inference/audio/stt/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model, engine: resolvedEngine }),
    });
    if (!response.ok) {
      const body = (await response.json().catch(() => null)) as {
        detail?: string;
      } | null;
      throw new Error(body?.detail ?? `HTTP ${response.status}`);
    }
  });
}

/** Start a background download of a dictation model. */
export async function startSttDownload(
  model: string,
  hfToken?: string,
): Promise<void> {
  const response = await authFetch("/api/inference/audio/stt/download", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...hubTokenHeader(hfToken),
    },
    body: JSON.stringify({ model, engine: sttEngineFor(model) }),
  });
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as {
      detail?: string;
    } | null;
    throw new Error(body?.detail ?? `HTTP ${response.status}`);
  }
}

/** Release the local STT model and its RAM/VRAM allocations. */
export function unloadSttModel(): Promise<void> {
  return queueSttLifecycle(async () => {
    const response = await authFetch("/api/inference/audio/stt/unload", {
      method: "POST",
    });
    if (!response.ok) {
      const body = (await response.json().catch(() => null)) as {
        detail?: string;
      } | null;
      throw new Error(body?.detail ?? `HTTP ${response.status}`);
    }
  });
}

/**
 * Local model dictation. Short recordings use one pass. Long recordings split
 * near Whisper's 30-second window. Confirm keeps text, discard removes it, and
 * either action releases the microphone immediately.
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

    // Pin the model and language chosen when recording began so later
    // segments are not transcribed with settings the user changed mid-session.
    const { sttModel: sessionModel, dictationLanguage: sessionLanguage } =
      useVoiceSettingsStore.getState();
    const sessionEngine = sttEngineFor(sessionModel);

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
    // Shared waveform meter also feeds this adapter's pause detector.
    let stopLevelMeter = () => {
      // Replaced after microphone access succeeds.
    };
    let onAudioFrame: (rawRms: number, now: number) => void = () => {};

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
      metered: boolean;
      recorder: MediaRecorder;
    };
    const results: string[] = [];
    const queue: { index: number; blob: Blob }[] = [];
    let worker = false;
    let currentSeg: Segment | null = null;
    let segCounter = 0;
    let pendingRecorders = 0;
    let silenceMs = 0;
    let lastFrameAt = 0;
    let cutting = false;
    let finalCutDone = false;
    let reportedTranscriptionError = false;

    const reportTranscriptionError = (error: unknown) => {
      if (reportedTranscriptionError || cancelled || ended) return;
      reportedTranscriptionError = true;
      const message =
        error instanceof Error && error.message
          ? error.message
          : "A recorded segment could not be transcribed.";
      console.error("STT transcription error:", error);
      toast.error(message);
    };

    const buildTranscript = () =>
      results
        .filter((part) => part?.trim())
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
        recordRecentDictation(corrected, activeDictationChatId());
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
      if (pendingRecorders === 0 && queue.length === 0 && !worker) {
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
          const text = await transcribeAudioBlob(item.blob, {
            model: sessionModel,
            language: sessionLanguage,
            engine: sessionEngine,
            signal: abortController.signal,
          });
          if (!cancelled) results[item.index] = text;
        } catch (error) {
          if (!cancelled && !abortController.signal.aborted) {
            // Keep any successfully transcribed segments, but never hide that
            // part of the recording was lost.
            reportTranscriptionError(error);
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
        metered: false,
        recorder: new MediaRecorder(
          stream,
          mimeType ? { mimeType } : undefined,
        ),
      };
      currentSeg = seg;
      silenceMs = 0;
      seg.recorder.addEventListener("dataavailable", (event) => {
        if (event.data.size > 0) seg.chunks.push(event.data);
      });
      seg.recorder.addEventListener("stop", () => {
        pendingRecorders = Math.max(0, pendingRecorders - 1);
        if (cancelled || ended) {
          maybeComplete();
          return;
        }
        const blob = new Blob(seg.chunks, {
          type: seg.recorder.mimeType || "audio/webm",
        });
        // If Web Audio is unavailable, no analyser frames arrive. Preserve the
        // recording instead of treating every segment as silence.
        enqueueSegment(seg.index, blob, seg.voiced || !seg.metered);
      });
      pendingRecorders += 1;
      try {
        seg.recorder.start(SEGMENT_TIMESLICE_MS);
      } catch (error) {
        pendingRecorders = Math.max(0, pendingRecorders - 1);
        if (currentSeg === seg) currentSeg = null;
        throw error;
      }
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

    // Pause detector: mark voiced frames. Short dictations remain one segment;
    // long ones cut at a pause after the target duration, or at the hard limit.
    onAudioFrame = (rawRms, now) => {
      const seg = currentSeg;
      if (!seg || finalizing) {
        lastFrameAt = now;
        return;
      }
      seg.metered = true;
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
          // Stop publishing zero-valued analyser frames as soon as recording
          // ends. The UI can switch immediately to its transcription shimmer.
          stopLevelMeter();
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
        // Wait for microphone access before warming the selected on-device
        // model. The backend uses cache-only loading and never downloads here.
        void loadSttModel(sessionModel, sessionEngine).catch(
          reportTranscriptionError,
        );
        stopLevelMeter = startDictationLevelMeter(stream, (rawRms, now) => {
          onAudioFrame(rawRms, now);
        });
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
