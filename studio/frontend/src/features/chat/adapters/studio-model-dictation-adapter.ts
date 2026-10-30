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
import {
  cleanDictationText,
  joinTranscriptParts,
} from "./dictation-text";
import type { SpeechSegment } from "./dictation-segmenter";
import {
  microphoneErrorMessage,
  openMicrophone,
  stopMediaStream,
} from "./microphone";
import { PcmCapture } from "./pcm-capture";
import { encodeWav } from "./pcm-recorder";
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
  resolveDictationChatId,
} from "./studio-web-speech-dictation-adapter";

// A sentence is cut at the first pause after 2.8 seconds, so its transcription
// starts while the next sentence is still being spoken. Fourteen seconds keeps
// the final clip well inside Whisper's 30-second window even when the speaker
// never pauses.
const DEFAULT_MIN_SEGMENT_MS = 2_800;
const DEFAULT_MAX_SEGMENT_MS = 14_000;
const DEFAULT_SILENCE_CUT_MS = 360;
const DEFAULT_PRE_ROLL_MS = 450;
const MIN_WHISPER_CLIP_MS = 1_000;

/**
 * Whisper decoders can drop or hallucinate on sub-second clips ("Yes", "Okay").
 * Pad clips shorter than 1.0s symmetrically with silence so short commands decode cleanly.
 */
function padShortSegment(samples: Float32Array, sampleRate: number): Float32Array {
  const minSamples = Math.round((MIN_WHISPER_CLIP_MS / 1000) * sampleRate);
  if (samples.length === 0 || samples.length >= minSamples) return samples;
  const padded = new Float32Array(minSamples);
  const lead = Math.min(Math.round(0.12 * sampleRate), Math.floor((minSamples - samples.length) / 2));
  padded.set(samples, lead);
  return padded;
}

/** Backend STT engine, decided by the model: Whisper ids run GGML through whisper.cpp, mtmd
 *  ids run through llama.cpp, the GGUF audio runtime's ids (saved keys, package folders and
 *  other GGUF repos) run through audiocpp, and a custom HF repo is safetensors on Transformers. */
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

/** One NDJSON line from the transcription stream. */
interface TranscriptStreamEvent {
  type?: string;
  text?: string;
  message?: string;
}

/** Read the backend's NDJSON transcript stream: progress lines carry the text so
 *  far, the last line carries the result, and an error line replaces the result.
 *  A response that is not NDJSON (an older server) is parsed as JSON instead. */
async function readTranscriptStream(
  response: Response,
  onPartial?: (text: string) => void,
): Promise<string> {
  const reader = response.body?.getReader();
  if (!reader) {
    const data = (await response.json()) as { text?: string };
    return (data.text ?? "").trim();
  }
  const decoder = new TextDecoder();
  let buffer = "";
  let text = "";
  let failure: string | null = null;
  let completed = false;
  const handle = (line: string) => {
    if (!line.trim()) return;
    let event: TranscriptStreamEvent;
    try {
      event = JSON.parse(line) as TranscriptStreamEvent;
    } catch {
      // A truncated line is not worth failing a transcription over.
      return;
    }
    if (event.type === "complete") {
      completed = true;
      if (typeof event.text === "string") text = event.text;
      return;
    }
    if (event.type === "error") {
      failure = event.message?.trim() || "Transcription failed. Try again.";
      return;
    }
    if (
      event.type === "progress" &&
      typeof event.text === "string" &&
      event.text.trim()
    ) {
      onPartial?.(event.text.trim());
    }
  };
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let newline = buffer.indexOf("\n");
    while (newline >= 0) {
      handle(buffer.slice(0, newline));
      buffer = buffer.slice(newline + 1);
      newline = buffer.indexOf("\n");
    }
  }
  buffer += decoder.decode();
  if (buffer.trim()) handle(buffer);
  if (failure) {
    // The stream reports a detail where a JSON error would carry a status; a
    // missing model must still open the download prompt.
    throw sttRequestError(/not downloaded/i.test(failure) ? 409 : 500, failure);
  }
  if (!completed) {
    throw new Error("The transcription stream ended before a result.");
  }
  return text.trim();
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
    /** Ask the local backend for NDJSON progress and live text. */
    stream?: boolean;
    /** Called with a segment's text so far; local engines only. */
    onPartial?: (text: string) => void;
    /** Keep the clip in the transcript gallery; dictation turns this off. */
    save?: boolean;
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
  const streaming = options.stream === true;
  if (streaming) params.set("stream", "true");
  if (options.save === false) params.set("save", "false");
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
  if (
    streaming &&
    response.headers.get("content-type")?.includes("application/x-ndjson")
  ) {
    return readTranscriptStream(response, options.onPartial);
  }
  const data = (await response.json()) as { text?: string };
  return (data.text ?? "").trim();
}

export interface SttDownloadStatus {
  downloading: boolean;
  model: string | null;
  /** opaque identity of this download attempt */
  download_id?: string | null;
  /** bounded history of attempts that completed successfully */
  completed_download_ids?: string[];
  /** active audiocpp quant; absent for other engines */
  variant?: string | null;
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

function queueSttLifecycle<T>(operation: () => Promise<T>): Promise<T> {
  const result = sttLifecycle.catch(() => {}).then(operation);
  sttLifecycle = result.then(() => {}, () => {});
  return result;
}

/** Report whether STT is installed and which model, if any, is resident. Passing a model
 *  extends the downloaded check to custom repos. */
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

/** The engine block that owns `model`. A curated Whisper prefers whisper.cpp, but without
 *  whisper-server the backend serves it through Transformers, so that is the fallback.
 *  mtmd and audiocpp models run nowhere else. */
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

/** Start a background download of a dictation model. */
export async function startSttDownload(
  model: string,
  hfToken?: string,
  engine?: SttEngine,
  ggufVariant?: string | null,
): Promise<SttDownloadStatus> {
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
  return (await response.json()) as SttDownloadStatus;
}

/** Stop an in-flight model download. Partial files stay cached, so starting the same download
 *  again resumes from where it stopped. */
export async function cancelSttDownload(
  model: string,
  engine?: SttEngine,
  ggufVariant?: string | null,
  downloadId?: string | null,
): Promise<void> {
  const resolvedEngine = engine ?? sttEngineFor(model);
  const response = await authFetch("/api/inference/audio/stt/download/cancel", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      engine: resolvedEngine,
      ...sttVariantBody(resolvedEngine, ggufVariant),
      ...(downloadId ? { download_id: downloadId } : {}),
    }),
  });
  const body = (await response.json().catch(() => null)) as
    | (Partial<SttDownloadStatus> & { detail?: string })
    | null;
  if (!response.ok) {
    throw new Error(body?.detail ?? `HTTP ${response.status}`);
  }
  if (body?.downloading && body.cancelled === false) {
    throw new Error("The download changed before cancellation completed.");
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
): Promise<{ released?: boolean }> {
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
    // `released` is false when the backend kept a model it was not asked to unload.
    const body = (await response.json().catch(() => null)) as {
      released?: boolean;
    } | null;
    return { released: body?.released };
  });
}

/** Keep a resident dictation model from being unloaded for another keep-alive window.
 *
 *  Voice typing sends this on a timer while it is enabled, so the first press after a
 *  pause does not pay for a reload. The backend never loads anything on its behalf: if
 *  the model has already gone, `held` is false and the next press loads it as usual. */
export async function keepSttModelAlive(
  engine?: SttEngine,
  model?: string,
  keepAliveSeconds?: number,
): Promise<boolean> {
  const params = new URLSearchParams();
  if (engine) params.set("engine", engine);
  if (model) params.set("model", model);
  if (
    typeof keepAliveSeconds === "number" &&
    Number.isFinite(keepAliveSeconds)
  ) {
    params.set("keep_alive_seconds", String(keepAliveSeconds));
  }
  const query = params.size ? `?${params}` : "";
  const response = await authFetch(
    `/api/inference/audio/stt/keepalive${query}`,
    { method: "POST" },
  );
  if (!response.ok) return false;
  const body = (await response.json().catch(() => null)) as {
    held?: boolean;
  } | null;
  return body?.held === true;
}

function uint8ToBase64(bytes: Uint8Array): string {
  let binary = "";
  const chunkSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    const chunk = bytes.subarray(offset, offset + chunkSize);
    binary += String.fromCharCode(...chunk);
  }
  return typeof btoa === "function" ? btoa(binary) : "";
}

/** Recorded-audio dictation. Short recordings use one pass; long ones split near whisper's 30s
 *  window. Confirm keeps text, discard removes it, and either releases the microphone. */
export interface StudioModelDictationOptions {
  chatId?: string | null;
  /** A microphone the caller keeps open between sessions. The adapter records from
   *  it but never stops its tracks; a session without one opens and releases its
   *  own. */
  microphone?: MediaStream | null;
  /** Remove filler words from every finished segment. */
  removeFillerWords?: boolean;
  /** Turn spoken punctuation into marks. */
  spokenPunctuation?: boolean;
  /** Ask the local backend for NDJSON partials. Off for callers that only ever
   *  read the final transcript (system-wide voice typing): the partials are
   *  decoded text they would throw away, and skipping them keeps the response a
   *  single JSON body. Defaults on, so the composer keeps its live text. */
  stream?: boolean;
}

export class StudioModelDictationAdapter implements DictationAdapter {
  private readonly options: StudioModelDictationOptions;

  constructor(options: StudioModelDictationOptions = {}) {
    this.options = options;
  }

  static isSupported(): boolean {
    if (typeof window === "undefined" || !window.isSecureContext) return false;
    if (navigator.mediaDevices?.getUserMedia === undefined) return false;
    const audioContext =
      window.AudioContext ??
      (window as unknown as { webkitAudioContext?: typeof AudioContext })
        .webkitAudioContext;
    return typeof audioContext === "function";
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
    const sessionVariant = usesExternalEndpoint
      ? null
      : sttModelVariant(sessionModel, settings.sttGgufVariant);
    const sessionChatId = resolveDictationChatId(this.options.chatId);
    // Pinned with the rest of the session: a caller that only reads finals must
    // not start receiving partials because the option changed mid-recording.
    const streamPartials = this.options.stream !== false;

    const speechStartCallbacks = new Set<() => void>();
    const speechEndCallbacks = new Set<
      (result: DictationAdapter.Result) => void
    >();
    const speechCallbacks = new Set<
      (result: DictationAdapter.Result) => void
    >();
    const endCallbacks = new Set<() => void>();

    let stream: MediaStream | null = null;
    let ownsStream = false;
    let capture: PcmCapture | null = null;
    let ended = false;
    let cancelled = false;
    let finalizing = false;
    const abortController = new AbortController();

    let resolveEnded: (() => void) | null = null;
    const endedPromise = new Promise<void>((resolve) => {
      resolveEnded = resolve;
    });

    // Background transcription pipeline. Each cut segment is a self-contained WAV
    // transcribed on its own; results are stored by index so the final text keeps
    // its order, while each finished segment is published at once.
    const results: string[] = [];
    const recordedSegments: SpeechSegment[] = [];
    const queue: { index: number; segment: SpeechSegment }[] = [];
    let worker = false;
    let segCounter = 0;
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

    const buildTranscript = () => joinTranscriptParts(results.filter(Boolean));

    const buildSessionAudioDataUrl = (): string | undefined => {
      if (useVoiceSettingsStore.getState().maxRecentDictations <= 0) {
        return undefined;
      }
      if (recordedSegments.length === 0) return undefined;
      const sampleRate = recordedSegments[0].sampleRate;
      const maxSamples = sampleRate * 45; // Cap stored audio at 45s per dictation
      let totalSamples = 0;
      for (const seg of recordedSegments) {
        totalSamples += seg.samples.length;
      }
      const length = Math.min(totalSamples, maxSamples);
      if (length <= 0) return undefined;
      const combined = new Float32Array(length);
      let offset = 0;
      for (const seg of recordedSegments) {
        if (offset >= length) break;
        const slice =
          offset + seg.samples.length > length
            ? seg.samples.subarray(0, length - offset)
            : seg.samples;
        combined.set(slice, offset);
        offset += slice.length;
      }
      const wavBuffer = encodeWav(combined, sampleRate);
      const base64 = uint8ToBase64(new Uint8Array(wavBuffer));
      return base64 ? `data:audio/wav;base64,${base64}` : undefined;
    };

    const stopCapture = () => {
      const active = capture;
      capture = null;
      active?.stop();
    };

    const releaseOwnedStream = () => {
      if (ownsStream) {
        stopMediaStream(stream);
        ownsStream = false;
      }
      stream = null;
    };

    const finishSession = (
      reason: "stopped" | "cancelled" | "error",
      transcript?: string,
    ) => {
      if (ended) return;
      // Set first: stopping the capture flushes one last segment, which must not
      // be queued into a session that is already over.
      ended = true;
      stopCapture();
      releaseOwnedStream();
      session.status = { type: "ended", reason };
      const corrected =
        reason === "cancelled" ? "" : (transcript ?? buildTranscript());
      if (corrected) {
        recordRecentDictation(
          corrected,
          sessionChatId,
          buildSessionAudioDataUrl(),
        );
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

    /** Publish one finished segment: cleaned, dictionary-corrected, in order. */
    const publishFinal = (index: number, text: string) => {
      results[index] = text;
      if (!text) return;
      markDictationTranscript();
      for (const callback of speechCallbacks) {
        callback({ transcript: text, isFinal: true });
      }
    };

    const cleanSegment = (text: string): string => {
      const trimmed = text.trim();
      if (!trimmed) return "";
      // Not trimmed: a segment can be a spoken "new paragraph", which is
      // whitespace and must survive to the join.
      return cleanDictationText(applyDictationDictionary(trimmed), {
        language: sessionLanguage,
        removeFillers: this.options.removeFillerWords,
        spokenPunctuation: this.options.spokenPunctuation,
      });
    };

    const enqueueSegment = (index: number, segment: SpeechSegment) => {
      if (cancelled || ended) return;
      if (useVoiceSettingsStore.getState().maxRecentDictations > 0) {
        recordedSegments.push(segment);
      }
      queue.push({ index, segment });
      processQueue();
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
      const wav = encodeWav(
        padShortSegment(item.segment.samples, item.segment.sampleRate),
        item.segment.sampleRate,
      );
      const blob = new Blob([wav], { type: "audio/wav" });
      void (async () => {
        try {
          const text = await transcribeAudioBlob(blob, {
            model: sessionModel,
            language: sessionLanguage,
            engine: sessionEngine,
            ggufVariant: sessionVariant,
            providerId: sessionProviderId,
            signal: abortController.signal,
            // The local backend can stream partial text while it decodes; an
            // external endpoint answers with one JSON body. A caller that only
            // reads finals turns this off: the partials would be decoded and
            // thrown away.
            stream: !usesExternalEndpoint && streamPartials,
            save: false,
            onPartial: (partial) => {
              if (cancelled || ended) return;
              for (const callback of speechCallbacks) {
                callback({ transcript: partial, isFinal: false });
              }
            },
          });
          if (!cancelled && !ended) publishFinal(item.index, cleanSegment(text));
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

    const session: StudioDictationSession = {
      status: { type: "starting" },
      stop: async () => {
        if (!ended && !finalizing) {
          finalizing = true;
          // Cut the final segment (its buffer survives) and release the
          // microphone at once; only transcription is left.
          stopCapture();
          releaseOwnedStream();
          finalCutDone = true;
          maybeComplete();
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
        const borrowed = this.options.microphone;
        const borrowedLive = borrowed?.getAudioTracks().some(
          (track) => track.readyState === "live",
        );
        if (borrowed && borrowedLive) {
          stream = borrowed;
        } else {
          stream = await openMicrophone(settings.micDeviceId);
          ownsStream = true;
        }
        if (ended || cancelled) {
          releaseOwnedStream();
          return;
        }
        if (!usesExternalEndpoint && sessionEngine) {
          // warm the model only after mic access; the backend never downloads here
          void loadSttModel(
            sessionModel,
            sessionEngine,
            abortController.signal,
            undefined,
            sessionVariant,
          ).catch((error: unknown) =>
            reportTranscriptionError(error, "preload"),
          );
        }
        capture = new PcmCapture(stream, {
          preRollMs: DEFAULT_PRE_ROLL_MS,
          minSegmentMs: DEFAULT_MIN_SEGMENT_MS,
          maxSegmentMs: DEFAULT_MAX_SEGMENT_MS,
          silenceCutMs: DEFAULT_SILENCE_CUT_MS,
          onSegment: (segment) => enqueueSegment(segCounter++, segment),
        });
        capture.start();
        session.status = { type: "running" };
        for (const callback of speechStartCallbacks) callback();
      } catch (error) {
        console.error("STT microphone error:", error);
        toast.error(microphoneErrorMessage(error));
        finishSession("error");
      }
    })();

    return session;
  }
}
