// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { withModelLoadNotice } from "@/lib/model-lifecycle-events";
import {
  type AudioCppRuntimeStatus,
  isAudioCppFolderId,
} from "../../audio/audio-cpp-catalog";
import { authFetch } from "@/features/auth";
import { listGgufVariants } from "@/features/hub/inventory/api";
import { hubTokenHeader } from "@/features/hub/lib/hub-token-header";
import { getHfToken, hfApiToken } from "@/features/hub/stores/hf-token-store";
import { useSettingsDialogStore } from "@/features/settings/stores/settings-dialog-store";
import { requestSttDownload } from "@/features/settings/stores/stt-download-prompt-store";
import {
  AUDIO_CPP_STT_MODELS,
  MTMD_STT_MODELS,
  type SttDevice,
  applyDictationDictionary,
  isCuratedSttModel,
  recordRecentDictation,
  resolveModelDictationLanguage,
  sttListedQuantDownloaded,
  sttModelVariant,
  useVoiceSettingsStore,
  withSttVariant,
} from "@/features/settings/stores/voice-settings-store";
import type { DictationAdapter } from "@assistant-ui/react";
import { toast } from "sonner";
import { withAbort } from "../../hub/lib/abort-signals";
import { encryptProviderApiKey } from "../api/providers-api";
import { getExternalProviderApiKey } from "../external-providers";
import { useExternalProvidersStore } from "../stores/external-providers-store";
import { startDictationLevelMeter } from "./dictation-level";
import { type SegmentRecorder, createAudioRecorder } from "./pcm-recorder";
import { SttModelNotDownloadedError, sttRequestError } from "./stt-errors";
// re-export preserves the public dictation entry point
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
// Whisper pads to 30s; long dictation cuts at the first pause after 20s or before 30s.
const MIN_SEGMENT_MS = 20_000;
const MAX_SEGMENT_MS = 28_000;
const SILENCE_CUT_MS = 280;
// Raw RMS (0..1) above which a frame counts as speech (well above the room floor after noise suppression).
const VOICE_RMS = 0.015;

// Prefer Opus; the backend decodes any of these with PyAV.
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

/** Decided by the model: Whisper ids on whisper.cpp, mtmd on llama.cpp, GGUF audio runtime ids
 *  on audiocpp, a custom HF repo on Transformers. */
export type SttEngine = "transformers" | "gguf" | "mtmd" | "audiocpp";

export function sttEngineFor(model: string): SttEngine {
  const id = model.trim();
  if (
    AUDIO_CPP_STT_MODELS.has(id) ||
    isAudioCppFolderId(id) ||
    (!isCuratedSttModel(id) && /-GGUF\/?$/i.test(id))
  )
    return "audiocpp";
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

export async function transcribeAudioBlob(
  blob: Blob,
  options: {
    model?: string;
    language?: string;
    engine?: SttEngine;
    device?: SttDevice;
    /** package-folder quant; defaults to the saved quant when `model` is also omitted */
    ggufVariant?: string | null;
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
  // cold sidecars resolve bare rows to their default quant, so the pick travels as `row:variant`
  const variant =
    options.ggufVariant !== undefined
      ? options.ggufVariant
      : options.model === undefined
        ? sttModelVariant(model, settings.sttGgufVariant)
        : null;
  const params = new URLSearchParams({
    model:
      engine === "audiocpp" && variant ? withSttVariant(model, variant) : model,
    fast: "true",
    engine,
  });
  if (language) params.set("language", language);
  params.set("device", options.device ?? settings.sttDevice);
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
  /** whether the user stopped the last download */
  cancelled?: boolean;
  /** cancellation target retained after the worker clears `model` */
  cancelled_model?: string | null;
  bytes_total: number | null;
  bytes_done: number | null;
}

export interface SttEngineStatus {
  available: boolean;
  loaded_model: string | null;
  /** resident audiocpp quant; absent for other engines */
  loaded_variant?: string | null;
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
  audiocpp?: SttEngineStatus;
  /** What the audio.cpp runtime can run; absent on servers predating it. */
  audio_cpp_runtime?: AudioCppRuntimeStatus;
}

// Keep load/unload requests ordered so a new recording cannot race an unload still finishing for the previous one.
let sttLifecycle: Promise<void> = Promise.resolve();

function queueSttLifecycle(operation: () => Promise<void>): Promise<void> {
  const result = sttLifecycle.catch(() => {}).then(operation);
  sttLifecycle = result.catch(() => {});
  return result;
}

/** Passing a model extends the downloaded check to custom repos. */
export async function fetchSttStatus(
  refreshKey?: number,
  model?: string,
  signal?: AbortSignal,
): Promise<SttStatus> {
  const params = new URLSearchParams();
  if (refreshKey !== undefined) params.set("refresh", String(refreshKey));
  if (model) params.set("model", model);
  const query = params.toString();
  const request = authFetch(
    `/api/inference/audio/stt/status${query ? `?${query}` : ""}`,
    { signal },
  );
  const response = await withAbort(request, signal);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return (await response.json()) as SttStatus;
}

/** Without whisper-server a curated Whisper is served by Transformers, so fall back to it. */
export function sttEngineStatusFor(
  status: SttStatus,
  model: string,
  engineOverride?: SttEngine,
): SttEngineStatus | undefined {
  const engine = engineOverride ?? sttEngineFor(model);
  if (engine === "mtmd") return status.mtmd;
  if (engine === "audiocpp") return status.audiocpp;
  if (engine === "gguf" && status.gguf?.available) return status.gguf;
  return status.transformers;
}

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

/** checks the quant because row status is true for any cached quant; unreadable listings defer to it */
export async function sttQuantDownloaded(
  model: string,
  ggufVariant: string | null | undefined,
  signal?: AbortSignal,
): Promise<boolean> {
  if (!ggufVariant) return true;
  const listing = await listGgufVariants(model, hfApiToken(getHfToken()), {
    signal,
  }).catch(() => null);
  return !listing || sttListedQuantDownloaded(listing, ggufVariant);
}

/** sends audiocpp quant picks; saved keys imply a package, and other engines take none */
function sttVariantBody(
  engine: SttEngine,
  ggufVariant: string | null | undefined,
): { gguf_variant?: string } {
  return engine === "audiocpp" && ggufVariant
    ? // biome-ignore lint/style/useNamingConvention: API schema
      { gguf_variant: ggufVariant }
    : {};
}

export function loadSttModel(
  model: string,
  engine?: SttEngine,
  signal?: AbortSignal,
  device?: SttDevice,
  ggufVariant?: string | null,
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
        ...sttVariantBody(resolvedEngine, ggufVariant),
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

export async function startSttDownload(
  model: string,
  hfToken?: string,
  engine?: SttEngine,
  ggufVariant?: string | null,
): Promise<void> {
  const resolvedEngine = engine ?? sttEngineFor(model);
  const response = await authFetch("/api/inference/audio/stt/download", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...hubTokenHeader(hfToken),
    },
    body: JSON.stringify({
      model,
      engine: resolvedEngine,
      ...sttVariantBody(resolvedEngine, ggufVariant),
    }),
  });
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as {
      detail?: string;
    } | null;
    throw new Error(body?.detail ?? `HTTP ${response.status}`);
  }
}

/** Partial files stay cached, so restarting resumes. */
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

/** `model` scopes the release: another surface can switch the engine meanwhile, so the backend
 *  compares under the sidecar's own lock. */
export function unloadSttModel(
  engine?: SttEngine,
  model?: string,
  options?: { wait?: boolean },
): Promise<void> {
  return queueSttLifecycle(async () => {
    const params = new URLSearchParams();
    if (engine) params.set("engine", engine);
    if (model) params.set("model", model);
    // Default drains in-flight transcription; opt out when memory is needed now.
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

/** Long recordings split near whisper's 30s window; confirm or discard releases the mic. */
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

    // Pin model, language and chat at start so mid-session changes cannot affect later segments.
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
    const sessionVariant = usesExternalEndpoint
      ? null
      : sttModelVariant(sessionModel, settings.sttGgufVariant);
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
    let stopLevelMeter = () => {
      // Replaced after microphone access succeeds.
    };
    let onAudioFrame: (rawRms: number, now: number) => void = () => {};

    let resolveEnded: (() => void) | null = null;
    const endedPromise = new Promise<void>((resolve) => {
      resolveEnded = resolve;
    });

    // Segments are transcribed independently and stored by index to keep order.
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
      // undownloaded models require confirmation; never download from dictation
      if (
        !usesExternalEndpoint &&
        error instanceof SttModelNotDownloadedError
      ) {
        requestSttDownload(sessionModel, { ggufVariant: sessionVariant });
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
      // cache-only preload failures make the session unusable, so stop before more audio is lost
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

    // serialize segment transcription to avoid flooding the backend
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
            ggufVariant: sessionVariant,
            providerId: sessionProviderId,
            signal: abortController.signal,
          });
          if (!cancelled) results[item.index] = text;
        } catch (error) {
          if (!cancelled && !abortController.signal.aborted) {
            // only a lost segment is partial; preload failures cost no recorded audio
            markDictationFailed();
            reportTranscriptionError(error);
          }
        } finally {
          worker = false;
          processQueue();
        }
      })();
    };

    // Never discard audio on RMS: a quiet mic can stay below VOICE_RMS for real speech.
    const enqueueSegment = (index: number, blob: Blob) => {
      if (blob.size > 0) {
        queue.push({ index, blob });
        processQueue();
      } else {
        results[index] = "";
        maybeComplete();
      }
    };

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

    // Cut at a pause so recording stays continuous while each clip decodes independently.
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
          // Cut the final segment so only the short tail remains, then release the mic now.
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
          // warm the model only after mic access; the backend never downloads here
          void loadSttModel(
            sessionModel,
            sessionEngine,
            undefined,
            undefined,
            sessionVariant,
          ).catch((error: unknown) =>
            reportTranscriptionError(error, "preload"),
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
