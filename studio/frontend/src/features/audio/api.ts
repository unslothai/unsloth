// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch, getAuthToken } from "@/features/auth";
import { accountTransitionPending } from "@/lib/account-transition";
import { apiUrl } from "@/lib/api-base";
import {
  formatApiErrorBody,
  readFastApiError,
} from "@/lib/format-fastapi-error";
import {
  type AudioRunRequest,
  type AudioSourceRef,
  type AudioVoiceCreateRequest,
  buildAudioRunBody,
  buildVoiceCreateBody,
  transcribeUrl,
} from "./audio-run-request";

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new Error(await readFastApiError(response));
  }
  return (await response.json()) as T;
}

export interface GeneratedAudio {
  data: string;
  format: string;
  sample_rate: number;
}

export interface GenerateAudioResponse {
  model: string;
  audio: GeneratedAudio;
  clip_id?: string | null;
  choices: { finish_reason: string }[];
}

export interface GenerateAudioOptions {
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  audio_instructions?: string;
  audio_language?: string;
  seed?: number;
  /** A GGUF audio model's own options by name, from its status `audio_options` schema. */
  audio_options?: Record<string, boolean | number | string>;
  signal?: AbortSignal;
}

export interface AudioDownloadPlan {
  entries: {
    repo_id: string;
    files: string[];
    bytes: number;
    gguf_filename: string | null;
    checkpoint?: boolean;
  }[];
  total_bytes: number;
  required_bytes?: number;
  checkpoint_bytes?: number;
}

export async function getAudioDownloadPlan(
  modelPath: string,
  hfToken?: string,
): Promise<AudioDownloadPlan> {
  const response = await authFetch("/api/inference/audio/download-plan", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_path: modelPath, hf_token: hfToken }),
  });
  return parseJson<AudioDownloadPlan>(response);
}

export async function generateAudio(
  text: string,
  options: GenerateAudioOptions = {},
): Promise<GenerateAudioResponse> {
  const { signal, ...sampling } = options;
  const response = await authFetch("/api/inference/audio/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages: [{ role: "user", content: text }],
      stream: false,
      ...sampling,
    }),
    signal,
  });
  return parseJson<GenerateAudioResponse>(response);
}

export interface AudioGalleryClip {
  id: string;
  url: string;
  prompt: string;
  model: string;
  audio_type: string;
  sample_rate: number;
  duration_s: number;
  created_at: string;
  pinned?: boolean;
  archived?: boolean;
  /** The server's unpinned sort key: the drag key, else the file mtime. */
  order_at?: number | null;
  /** The Audio workflow that made the clip. Older servers omit it; read it through clipWorkflow. */
  workflow?: string | null;
  /** Clips one run made together share a group. */
  group_id?: string | null;
  reference_name?: string | null;
}

export interface AudioGalleryListResponse {
  audio: AudioGalleryClip[];
  has_more: boolean;
  next_before_mtime: number | null;
  next_before_id: string | null;
  next_before_pin?: number | null;
}

/** Where the next page starts: the last clip's order key, id and pin rank (null if unpinned). */
export interface AudioGalleryCursor {
  mtime: number;
  id: string;
  pin?: number | null;
}

export function audioGalleryCursor(
  page: AudioGalleryListResponse,
): AudioGalleryCursor | null {
  return page.next_before_mtime !== null && page.next_before_id !== null
    ? {
        mtime: page.next_before_mtime,
        id: page.next_before_id,
        pin: page.next_before_pin ?? null,
      }
    : null;
}

export async function listAudioGallery(
  offset: number,
  limit: number,
  before?: AudioGalleryCursor | null,
  archived = false,
): Promise<AudioGalleryListResponse> {
  const pin =
    before?.pin !== null && before?.pin !== undefined
      ? `&before_pin=${encodeURIComponent(before.pin)}`
      : "";
  const cursor = before
    ? `&before_mtime=${encodeURIComponent(before.mtime)}&before_id=${encodeURIComponent(before.id)}${pin}`
    : "";
  const response = await authFetch(
    `/api/inference/audio/gallery?offset=${offset}&limit=${limit}&archived=${archived}${cursor}`,
  );
  return parseJson<AudioGalleryListResponse>(response);
}

export async function setAudioClipFlags(
  id: string,
  flags: { pinned?: boolean; archived?: boolean },
): Promise<AudioGalleryClip> {
  const response = await authFetch(
    `/api/inference/audio/gallery/${encodeURIComponent(id)}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(flags),
    },
  );
  return parseJson<AudioGalleryClip>(response);
}

/** Move one clip to just after `afterId` (null = top). */
export async function moveAudioClip(
  id: string,
  afterId: string | null,
): Promise<AudioGalleryClip> {
  const response = await authFetch(
    `/api/inference/audio/gallery/${encodeURIComponent(id)}/move`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ after_id: afterId }),
    },
  );
  return parseJson<AudioGalleryClip>(response);
}

/** Copy one clip into a chat project's folder. */
export async function addAudioClipToProject(
  id: string,
  projectId: string,
): Promise<{ path: string; already: boolean }> {
  const response = await authFetch(
    `/api/inference/audio/gallery/${encodeURIComponent(id)}/project`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project_id: projectId }),
    },
  );
  return parseJson<{ path: string; already: boolean }>(response);
}

export async function deleteAudioClip(id: string): Promise<void> {
  const response = await authFetch(
    `/api/inference/audio/gallery/${encodeURIComponent(id)}`,
    { method: "DELETE" },
  );
  if (!response.ok) throw new Error(await readFastApiError(response));
}

export async function clearAudioGallery(
  workflow?: "speak" | "clone" | "music",
): Promise<number> {
  const query = workflow ? `?workflow=${workflow}` : "";
  const response = await authFetch(`/api/inference/audio/gallery${query}`, {
    method: "DELETE",
  });
  const body = await parseJson<{ removed: number }>(response);
  return body.removed;
}

export async function fetchClipBlob(url: string): Promise<Blob> {
  const response = await authFetch(url);
  if (!response.ok) throw new Error(await readFastApiError(response));
  return response.blob();
}

export async function fetchClipObjectUrl(
  url: string,
): Promise<{ url: string; bytes: number; blob: Blob }> {
  const blob = await fetchClipBlob(url);
  return { url: URL.createObjectURL(blob), bytes: blob.size, blob };
}

export async function transcribeWithProgress(
  blob: Blob,
  title: string,
  options: {
    model: string;
    engine: string;
    device: string;
    signal: AbortSignal;
  },
  onProgress: (
    progress: import("./transcript-stream").TranscriptProgress,
  ) => void,
): Promise<import("./transcript-stream").TranscriptResult> {
  const { readTranscriptStream } = await import("./transcript-stream");
  const params = new URLSearchParams({
    model: options.model,
    engine: options.engine,
    device: options.device,
    fast: "true",
    stream: "true",
    title: title.slice(0, 255),
  });
  const response = await authFetch(
    `/api/inference/audio/transcribe/raw?${params}`,
    {
      method: "POST",
      headers: { "Content-Type": blob.type || "application/octet-stream" },
      body: blob,
      signal: options.signal,
    },
  );
  if (!response.ok) throw new Error(await readFastApiError(response));
  if (!response.body) throw new Error("The transcription response was empty.");
  return readTranscriptStream(response.body, onProgress);
}

export async function listTranscripts(
  archived = false,
  before?: string | null,
): Promise<{
  transcripts: import("./transcript-stream").TranscriptRecord[];
  next_cursor: string | null;
}> {
  const params = new URLSearchParams({
    archived: String(archived),
    limit: "50",
  });
  if (before) params.set("before", before);
  return parseJson(
    await authFetch(`/api/inference/audio/transcripts?${params}`),
  );
}

export async function archiveTranscript(
  id: string,
  archived: boolean,
): Promise<void> {
  await parseJson(
    await authFetch(
      `/api/inference/audio/transcripts/${encodeURIComponent(id)}`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ archived }),
      },
    ),
  );
}

export async function deleteTranscript(id?: string): Promise<void> {
  await parseJson(
    await authFetch(
      `/api/inference/audio/transcripts${id ? `/${encodeURIComponent(id)}` : ""}`,
      {
        method: "DELETE",
      },
    ),
  );
}

/** Carries the status so callers can tell an expired upload (404) from a refusal. */
export class AudioApiError extends Error {
  readonly status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "AudioApiError";
    this.status = status;
  }
}

async function parseAudioJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new AudioApiError(await readFastApiError(response), response.status);
  }
  return (await response.json()) as T;
}

interface AudioInputRecord {
  id: string;
  name: string;
  duration_s: number;
  sample_rate: number;
  channels: number;
  url: string;
  expires_at: string;
}

/** XHR is the only browser API with upload progress; the caller retries a 401 via authFetch to refresh the token. */
function uploadWithProgress(
  url: string,
  blob: Blob,
  onProgress: (fraction: number | null) => void,
  signal?: AbortSignal,
): Promise<{ status: number; body: unknown }> {
  return new Promise((resolve, reject) => {
    // Same fence as authFetch: mid-switch, this tab's file would be stored under the next account.
    if (accountTransitionPending()) {
      reject(new Error("Another tab is switching accounts; this tab will reload."));
      return;
    }
    const xhr = new XMLHttpRequest();
    xhr.open("POST", apiUrl(url));
    const token = getAuthToken();
    if (token) xhr.setRequestHeader("Authorization", `Bearer ${token}`);
    xhr.setRequestHeader(
      "Content-Type",
      blob.type || "application/octet-stream",
    );
    xhr.responseType = "json";
    xhr.upload.onprogress = (event) =>
      onProgress(event.lengthComputable ? event.loaded / event.total : null);
    xhr.onload = () => resolve({ status: xhr.status, body: xhr.response });
    xhr.onerror = () =>
      reject(
        new Error("The upload failed. Check the connection and try again."),
      );
    xhr.onabort = () => reject(new DOMException("Aborted", "AbortError"));
    if (signal) {
      if (signal.aborted) {
        reject(new DOMException("Aborted", "AbortError"));
        return;
      }
      signal.addEventListener("abort", () => xhr.abort(), { once: true });
    }
    xhr.send(blob);
  });
}

export async function uploadAudioInput(
  blob: Blob,
  name: string,
  options: {
    onProgress?: (fraction: number | null) => void;
    signal?: AbortSignal;
  } = {},
): Promise<AudioInputRecord> {
  const url = `/api/inference/audio/inputs?name=${encodeURIComponent(name.slice(0, 255))}`;
  if (options.onProgress && typeof XMLHttpRequest !== "undefined") {
    const { status, body } = await uploadWithProgress(
      url,
      blob,
      options.onProgress,
      options.signal,
    );
    if (status >= 200 && status < 300) return body as AudioInputRecord;
    if (status !== 401) {
      throw new AudioApiError(
        formatApiErrorBody(body) ?? `Upload failed (${status})`,
        status,
      );
    }
  }
  const response = await authFetch(url, {
    method: "POST",
    headers: { "Content-Type": blob.type || "application/octet-stream" },
    body: blob,
    signal: options.signal,
  });
  return parseAudioJson<AudioInputRecord>(response);
}

export async function fetchAudioBlob(
  url: string,
  signal?: AbortSignal,
): Promise<Blob> {
  const response = await authFetch(url, { signal });
  if (!response.ok) {
    throw new AudioApiError(await readFastApiError(response), response.status);
  }
  return response.blob();
}

interface TranscribeInputResponse {
  text: string;
  language: string | null;
  model: string;
}

export async function transcribeAudioInput(
  ref: AudioSourceRef,
  body: { model: string; engine?: string; device?: string; language?: string },
  signal?: AbortSignal,
): Promise<TranscribeInputResponse> {
  const response = await authFetch(transcribeUrl(ref), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  return parseAudioJson<TranscribeInputResponse>(response);
}

export interface AudioRunResponse {
  clips: {
    id: string;
    role: string;
    url: string;
    sample_rate: number;
    duration_s: number;
    workflow: string;
  }[];
  model: string;
  audio: GeneratedAudio | null;
}

export async function runAudio(
  request: AudioRunRequest,
  signal?: AbortSignal,
): Promise<AudioRunResponse> {
  const response = await authFetch("/api/inference/audio/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(buildAudioRunBody(request)),
    signal,
  });
  return parseAudioJson<AudioRunResponse>(response);
}

export interface AudioVoice {
  id: string;
  name: string;
  transcript: string | null;
  language: string | null;
  duration_s: number;
  sample_rate: number;
  created_at: string;
  url: string;
}

export async function listVoices(): Promise<AudioVoice[]> {
  const body = await parseAudioJson<{ voices: AudioVoice[] }>(
    await authFetch("/api/inference/audio/voices"),
  );
  return body.voices;
}

export async function createVoice(
  request: AudioVoiceCreateRequest,
): Promise<AudioVoice> {
  return parseAudioJson<AudioVoice>(
    await authFetch("/api/inference/audio/voices", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildVoiceCreateBody(request)),
    }),
  );
}

export async function updateVoice(
  id: string,
  patch: {
    name?: string;
    transcript?: string | null;
    language?: string | null;
  },
): Promise<AudioVoice> {
  return parseAudioJson<AudioVoice>(
    await authFetch(`/api/inference/audio/voices/${encodeURIComponent(id)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    }),
  );
}

export async function deleteVoice(id: string): Promise<void> {
  await parseAudioJson(
    await authFetch(`/api/inference/audio/voices/${encodeURIComponent(id)}`, {
      method: "DELETE",
    }),
  );
}
