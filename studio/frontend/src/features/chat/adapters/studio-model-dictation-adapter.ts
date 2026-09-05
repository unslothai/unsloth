// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { withModelLoadNotice } from "@/lib/model-lifecycle-events";
import { authFetch } from "@/features/auth";
import { hubTokenHeader } from "@/features/hub/lib/hub-token-header";
import { useSettingsDialogStore } from "@/features/settings/stores/settings-dialog-store";
import { requestSttDownload } from "@/features/settings/stores/stt-download-prompt-store";
import {
  MTMD_STT_MODELS,
  type SttDevice,
  applyDictationDictionary,
  isCuratedSttModel,
  recordRecentDictation,
  resolveModelDictationLanguage,
  useVoiceSettingsStore,
} from "@/features/settings/stores/voice-settings-store";
import type { DictationAdapter } from "@assistant-ui/react";
import { toast } from "sonner";
import { encryptProviderApiKey } from "../api/providers-api";
import { getExternalProviderApiKey } from "../external-providers";
import { useExternalProvidersStore } from "../stores/external-providers-store";
import { startDictationLevelMeter } from "./dictation-level";
import { type SegmentRecorder, createAudioRecorder } from "./pcm-recorder";
import { SttModelNotDownloadedError, sttRequestError } from "./stt-errors";
// Re-exported so the one public entry point for dictation is unchanged.
export { SttModelNotDownloadedError } from "./stt-errors";
import {
  beginDictationSession,
  markDictationFailed,
  markDictationTranscript,
} from "./dictation-outcome";
import {
  type StudioDictationSession,
  isMissingDeviceError,
  resolveDictationChatId,
} from "./studio-web-speech-dictation-adapter";

// Fine timeslice so the buffer is ready the moment a segment is cut or stopped.
const SEGMENT_TIMESLICE_MS = 250;
// Whisper pads input to 30s. Short dictation stays one clip; long dictation cuts at the first
// pause after 20s or before the 30s boundary.
const MIN_SEGMENT_MS = 20_000;
const MAX_SEGMENT_MS = 28_000;
const SILENCE_CUT_MS = 280;
// Raw RMS (0..1) above which a frame counts as speech (well above the room floor after noise suppression).
const VOICE_RMS = 0.015;

// Prefer Opus (small, widely supported); fall back to whatever the browser records. The
// backend decodes any of these with PyAV.
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

/** Backend STT engine, decided by the model: Whisper ids run GGML through whisper.cpp, mtmd
 *  ids run through llama.cpp, and a custom HF repo is safetensors on Transformers. */
export type SttEngine = "transformers" | "gguf" | "mtmd";

export function sttEngineFor(model: string): SttEngine {
  // whisper.cpp is Whisper-only, so the newer ASR models go to llama.cpp.
  if (MTMD_STT_MODELS.has(model.trim())) return "mtmd";
  return isCuratedSttModel(model) ? "gguf" : "transformers";
}

function externalSttLanguage(language: string): string | undefined {
  const normalized = language.trim().replaceAll("_", "-").toLowerCase();
  if (!normalized || normalized === "auto") {
    return undefined;
  }
  return normalized.split("-", 1)[0] || undefined;
}

function dictationFilename(contentType: string): string {
  if (contentType.includes("ogg")) {
    return "dictation.ogg";
  }
  if (contentType.includes("mp4")) {
    return "dictation.m4a";
  }
  if (contentType.includes("wav")) {
    return "dictation.wav";
  }
  return "dictation.webm";
}

async function sttErrorDetail(response: Response): Promise<string> {
  const body = (await response.json().catch(() => null)) as {
    detail?: string;
    error?: { message?: string };
  } | null;
  return body?.detail ?? body?.error?.message ?? `HTTP ${response.status}`;
}

/** post recorded audio to the selected STT backend and return its transcript. */
export async function transcribeAudioBlob(
  blob: Blob,
  options: {
    model?: string;
    language?: string;
    engine?: SttEngine;
    providerId?: string;
    signal?: AbortSignal;
  } = {},
): Promise<string> {
  const settings = useVoiceSettingsStore.getState();
  const usesExternalEndpoint = options.providerId !== undefined;
  const providerId = options.providerId?.trim() ?? "";
  const model = (
    options.model ??
    (usesExternalEndpoint ? settings.sttProviderModel : settings.sttModel)
  ).trim();
  const languageSetting = options.language ?? settings.dictationLanguage;

  if (usesExternalEndpoint) {
    const providersState = useExternalProvidersStore.getState();
    if (!providersState.connectionsEnabled) {
      throw new Error(
        "Connections are disabled. Enable connections before using custom transcription.",
      );
    }
    if (!providerId || !model) {
      throw new Error(
        "Custom transcription is not configured. Pick a connection and model in Settings → Voice.",
      );
    }
    const form = new FormData();
    form.set("file", blob, dictationFilename(blob.type));
    form.set("provider_id", providerId);
    form.set("model", model);
    form.set("response_format", "json");
    const provider = providersState.providers.find(
      (candidate) => candidate.id === providerId,
    );
    const legacyApiKey = provider?.hasApiKey
      ? ""
      : getExternalProviderApiKey(providerId).trim();
    if (legacyApiKey) {
      form.set(
        "encrypted_api_key",
        await encryptProviderApiKey(legacyApiKey),
      );
    }
    const language = externalSttLanguage(languageSetting);
    if (language) {
      form.set("language", language);
    }
    const response = await authFetch("/api/inference/audio/transcriptions", {
      method: "POST",
      body: form,
      signal: options.signal,
    });
    if (!response.ok) {
      throw sttRequestError(response.status, await sttErrorDetail(response));
    }
    const data = (await response.json()) as { text?: string };
    if (typeof data.text !== "string") {
      throw new Error("The transcription endpoint returned no text.");
    }
    return data.text.trim();
  }

  const language = resolveModelDictationLanguage(model, languageSetting);
  const engine = options.engine ?? sttEngineFor(model);
  const params = new URLSearchParams({ model, fast: "true", engine });
  if (language) params.set("language", language);
  params.set("device", settings.sttDevice);
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
    const detail = await sttErrorDetail(response);
    if (response.status === 501) {
      throw new Error(
        "Speech-to-text is not available on this server. Run `unsloth studio update` to install it.",
      );
    }
    throw sttRequestError(response.status, detail);
  }
  const data = (await response.json()) as { text?: string };
  return (data.text ?? "").trim();
}

export interface SttDownloadStatus {
  downloading: boolean;
  model: string | null;
  error: string | null;
  /** The last download was stopped by the user rather than failing. */
  cancelled?: boolean;
  /** Which model that cancellation applies to. `model` goes null once the worker thread stops,
   *  so this is the only way to tell a settled cancellation from an unrelated one. */
  cancelled_model?: string | null;
  bytes_total: number | null;
  bytes_done: number | null;
}

export interface SttEngineStatus {
  available: boolean;
  loaded_model: string | null;
  loading: boolean;
  device: string | null;
  keep_alive_seconds: number;
  default_model: string | null;
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
  mtmd?: SttEngineStatus;
}

// Keep load/unload requests ordered so a new recording cannot race an unload still finishing for the previous one.
let sttLifecycle: Promise<void> = Promise.resolve();

function queueSttLifecycle(operation: () => Promise<void>): Promise<void> {
  const result = sttLifecycle.catch(() => {}).then(operation);
  sttLifecycle = result.catch(() => {});
  return result;
}

/** Report whether STT is installed and which model, if any, is resident. Passing a model
 *  extends the downloaded check to custom repos. */
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

/** The engine block that owns `model`. A curated Whisper prefers whisper.cpp, but without
 *  whisper-server the backend serves it through Transformers, so that is the fallback.
 *  mtmd models run nowhere else. */
export function sttEngineStatusFor(
  status: SttStatus,
  model: string,
  engineOverride?: SttEngine,
): SttEngineStatus | undefined {
  const engine = engineOverride ?? sttEngineFor(model);
  if (engine === "mtmd") return status.mtmd;
  if (engine === "gguf" && status.gguf?.available) return status.gguf;
  return status.transformers;
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
export function loadSttModel(
  model: string,
  engine?: SttEngine,
  signal?: AbortSignal,
  device?: SttDevice,
): Promise<void> {
  const resolvedEngine = engine ?? sttEngineFor(model);
  const resolvedDevice = device ?? useVoiceSettingsStore.getState().sttDevice;
  // Announced so the indicator shows the load immediately, as the toast does.
  return queueSttLifecycle(() =>
    withModelLoadNotice("stt", model, async () => {
    const response = await authFetch("/api/inference/audio/stt/load", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        engine: resolvedEngine,
        device: resolvedDevice,
      }),
      signal,
    });
    if (!response.ok) {
      const body = (await response.json().catch(() => null)) as {
        detail?: string;
      } | null;
      const detail = body?.detail ?? `HTTP ${response.status}`;
      throw sttRequestError(response.status, detail);
    }
    }),
  );
}

/** Start a background download of a dictation model. */
export async function startSttDownload(
  model: string,
  hfToken?: string,
  engine?: SttEngine,
): Promise<void> {
  const response = await authFetch("/api/inference/audio/stt/download", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...hubTokenHeader(hfToken),
    },
    body: JSON.stringify({ model, engine: engine ?? sttEngineFor(model) }),
  });
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as {
      detail?: string;
    } | null;
    throw new Error(body?.detail ?? `HTTP ${response.status}`);
  }
}

/** Stop an in-flight model download. Partial files stay cached, so starting the same download
 *  again resumes from where it stopped. */
export async function cancelSttDownload(
  model: string,
  engine?: SttEngine,
): Promise<void> {
  const response = await authFetch("/api/inference/audio/stt/download/cancel", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, engine: engine ?? sttEngineFor(model) }),
  });
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as {
      detail?: string;
    } | null;
    throw new Error(body?.detail ?? `HTTP ${response.status}`);
  }
}

/** Release the local STT model and its RAM/VRAM allocations. */
/** Release the dictation sidecar. `model` scopes the release to the model the caller claims:
 *  another surface can switch the same engine between the ownership check and this request,
 *  so the backend compares under the sidecar's own lock. */
export function unloadSttModel(
  engine?: SttEngine,
  model?: string,
  options?: { wait?: boolean },
): Promise<void> {
  return queueSttLifecycle(async () => {
    const params = new URLSearchParams();
    if (engine) params.set("engine", engine);
    if (model) params.set("model", model);
    // Opt-out only: the default drains an in-flight transcription, right when the
    // caller needs the memory back now.
    if (options?.wait === false) params.set("wait", "false");
    const query = params.size ? `?${params}` : "";
    const response = await authFetch(
      `/api/inference/audio/stt/unload${query}`,
      { method: "POST" },
    );
    if (!response.ok) {
      const body = (await response.json().catch(() => null)) as {
        detail?: string;
      } | null;
      throw new Error(body?.detail ?? `HTTP ${response.status}`);
    }
  });
}

/** Recorded-audio dictation. Short recordings use one pass; long ones split near whisper's 30s
 *  window. Confirm keeps text, discard removes it, and either releases the microphone. */
export class StudioModelDictationAdapter implements DictationAdapter {
  private readonly chatId: string | null | undefined;

  constructor(options: { chatId?: string | null } = {}) {
    this.chatId = options.chatId;
  }

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

    // Pin the model, language and linked chat chosen when recording began, so a mid-session
    // settings change or thread switch cannot affect later segments or relink the transcript.
    const settings = useVoiceSettingsStore.getState();
    const usesExternalEndpoint = settings.dictationEngine === "custom";
    const sessionProviderId = usesExternalEndpoint
      ? settings.sttProviderId.trim()
      : undefined;
    const sessionModel = usesExternalEndpoint
      ? settings.sttProviderModel.trim()
      : settings.sttModel;
    if (usesExternalEndpoint && (!sessionProviderId || !sessionModel)) {
      throw new Error(
        "Custom transcription is not configured. Pick a connection and model in Settings → Voice.",
      );
    }
    beginDictationSession();
    const sessionLanguage = usesExternalEndpoint
      ? settings.dictationLanguage
      : resolveModelDictationLanguage(sessionModel, settings.dictationLanguage);
    const sessionEngine = usesExternalEndpoint
      ? undefined
      : sttEngineFor(sessionModel);
    const sessionChatId = resolveDictationChatId(this.chatId);

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

    // Background transcription pipeline. Each segment is a self-contained clip transcribed on its
    // own; results are stored by index so the final text keeps its order.
    type Segment = {
      index: number;
      chunks: Blob[];
      startedAt: number;
      voiced: boolean;
      recorder: SegmentRecorder;
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

    const reportTranscriptionError = (
      error: unknown,
      stage: "preload" | "segment" = "segment",
    ) => {
      if (reportedTranscriptionError || cancelled || ended) return;
      reportedTranscriptionError = true;
      console.error("STT transcription error:", error);
      // An undownloaded model is the ordinary first-run state, not a failure. Point at the
      // download; never start it here.
      if (
        !usesExternalEndpoint &&
        error instanceof SttModelNotDownloadedError
      ) {
        requestSttDownload(sessionModel);
        finishSession("cancelled");
        return;
      }
      const message =
        error instanceof Error && error.message
          ? error.message
          : "A recorded segment could not be transcribed.";
      toast.error(message, {
        action: {
          label: "Open Voice settings",
          onClick: () => useSettingsDialogStore.getState().openDialog("voice"),
        },
      });
      // The preload runs cache-only, so nothing it reports is transient: a missing runtime or a
      // load refused for training means no segment of this session can be transcribed. End it
      // rather than let the user keep speaking into a recorder whose audio is already lost.
      if (stage === "preload") finishSession("cancelled");
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
        markDictationTranscript();
        for (const callback of speechCallbacks) {
          callback({ transcript: corrected, isFinal: true });
        }
        recordRecentDictation(corrected, sessionChatId);
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
            providerId: sessionProviderId,
            signal: abortController.signal,
          });
          if (!cancelled) results[item.index] = text;
        } catch (error) {
          if (!cancelled && !abortController.signal.aborted) {
            // Keep transcribed segments, but never hide that part was lost. Only a lost segment is
            // partial: the model preload shares this reporter and can fail without costing any audio.
            markDictationFailed();
            reportTranscriptionError(error);
          }
        } finally {
          worker = false;
          processQueue();
        }
      })();
    };

    // Every non-empty recording is transcribed. The RMS meter only shapes segment boundaries; a
    // quiet microphone or suspended AudioContext can keep it below VOICE_RMS for real speech,
    // so it must never discard audio. Whisper returns an empty transcript for genuine silence.
    const enqueueSegment = (index: number, blob: Blob) => {
      if (blob.size > 0) {
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
        recorder: createAudioRecorder(stream, mimeType),
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
        enqueueSegment(seg.index, blob);
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

    // Close the current segment at a pause and open the next, so recording stays continuous while
    // each clip is independently decodable.
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

    // Pause detector: mark voiced frames. Short dictations stay one segment; long ones cut at a
    // pause after the target duration, or at the hard limit.
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
          // Stop publishing zero-valued frames at once so the UI can switch to its transcription shimmer.
          stopLevelMeter();
          const seg = currentSeg;
          // Cut the final segment (its buffer survives) so only the short tail is left to transcribe,
          // then release the mic immediately.
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
          // Saved mic may be unplugged; fall back to the default.
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
        if (!usesExternalEndpoint && sessionEngine) {
          // warm the model only after mic access; the backend never downloads here.
          void loadSttModel(sessionModel, sessionEngine).catch(
            (error: unknown) => reportTranscriptionError(error, "preload"),
          );
        }
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
