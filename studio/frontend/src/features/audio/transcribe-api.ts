// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";
import { AudioApiError } from "./api";
import type { AudioSourceRef } from "./audio-run-request";
import type { SttCapabilities } from "./transcribe-capabilities";
import {
  type TranscriptProgress,
  type TranscriptRecord,
  type TranscriptResult,
  readTranscriptStream,
} from "./transcript-stream";

const JSON_HEADERS = { "Content-Type": "application/json" };

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await authFetch(url, init);
  if (!response.ok) throw new Error(await readFastApiError(response));
  return (await response.json()) as T;
}

const transcriptUrl = (id: string) =>
  `/api/inference/audio/transcripts/${encodeURIComponent(id)}`;

export async function transcribeSourceWithProgress(
  ref: AudioSourceRef,
  title: string,
  options: {
    model: string;
    engine: string;
    device: string;
    language: string;
    timestamps: boolean;
    speakers: boolean;
    signal: AbortSignal;
  },
  onProgress: (progress: TranscriptProgress) => void,
): Promise<TranscriptResult> {
  const response = await authFetch("/api/inference/audio/transcribe/source", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({
      source: ref,
      model: options.model,
      engine: options.engine || null,
      device: options.device || null,
      language: options.language || null,
      timestamps: options.timestamps,
      speakers: options.speakers,
      title: title.slice(0, 255),
    }),
    signal: options.signal,
  });
  // The status matters: a 404 means the upload expired, and the card says so.
  if (!response.ok)
    throw new AudioApiError(await readFastApiError(response), response.status);
  if (!response.body) throw new Error("The transcription response was empty.");
  return readTranscriptStream(response.body, onProgress);
}

export function fetchSttCapabilities(
  model: string,
  engine: string | null,
  signal?: AbortSignal,
): Promise<SttCapabilities> {
  const params = new URLSearchParams({ model });
  if (engine) params.set("engine", engine);
  return fetchJson(`/api/inference/audio/stt/capabilities?${params}`, {
    signal,
  });
}

export function getTranscript(
  id: string,
  signal?: AbortSignal,
): Promise<TranscriptRecord> {
  return fetchJson(transcriptUrl(id), { signal });
}

/** null clears a name back to "Speaker N". */
export function renameTranscriptSpeakers(
  id: string,
  names: Record<string, string | null>,
): Promise<TranscriptRecord> {
  return fetchJson(transcriptUrl(id), {
    method: "PATCH",
    headers: JSON_HEADERS,
    body: JSON.stringify({ speaker_names: names }),
  });
}
