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

export interface TranscribeSourceOptions {
  model: string;
  engine: string;
  device: string;
  /** An ISO code; empty lets the model detect it. */
  language: string;
  timestamps: boolean;
  speakers: boolean;
  signal: AbortSignal;
}

/** The body for /audio/transcribe/source: the source by id, never a path. */
export function transcribeSourceBody(
  ref: AudioSourceRef,
  title: string,
  options: Omit<TranscribeSourceOptions, "signal">,
) {
  return {
    source: ref,
    model: options.model,
    engine: options.engine || null,
    device: options.device || null,
    language: options.language || null,
    timestamps: options.timestamps,
    speakers: options.speakers,
    title: title.slice(0, 255),
  };
}

/** Transcribes an uploaded input, history clip or saved voice; the server reads it from its own store. */
export async function transcribeSourceWithProgress(
  ref: AudioSourceRef,
  title: string,
  { signal, ...options }: TranscribeSourceOptions,
  onProgress: (progress: TranscriptProgress) => void,
): Promise<TranscriptResult> {
  const response = await authFetch("/api/inference/audio/transcribe/source", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(transcribeSourceBody(ref, title, options)),
    signal,
  });
  // The status matters here: a 404 means the upload expired, and the card says so.
  if (!response.ok)
    throw new AudioApiError(await readFastApiError(response), response.status);
  if (!response.body) throw new Error("The transcription response was empty.");
  return readTranscriptStream(response.body, onProgress);
}

export function sttCapabilitiesUrl(
  model: string,
  engine: string | null,
): string {
  const params = new URLSearchParams({ model });
  if (engine) params.set("engine", engine);
  return `/api/inference/audio/stt/capabilities?${params}`;
}

/** What a speech-to-text model can add beyond text; the server answers from its cache, offline. */
export async function fetchSttCapabilities(
  model: string,
  engine: string | null,
  signal?: AbortSignal,
): Promise<SttCapabilities> {
  const response = await authFetch(sttCapabilitiesUrl(model, engine), {
    signal,
  });
  if (!response.ok) throw new Error(await readFastApiError(response));
  return (await response.json()) as SttCapabilities;
}

/** One saved transcript with its segments; the history list only carries counts. */
export async function getTranscript(
  id: string,
  signal?: AbortSignal,
): Promise<TranscriptRecord> {
  const response = await authFetch(
    `/api/inference/audio/transcripts/${encodeURIComponent(id)}`,
    { signal },
  );
  if (!response.ok) throw new Error(await readFastApiError(response));
  return (await response.json()) as TranscriptRecord;
}

/** Saves speaker names on a transcript; null clears one back to "Speaker N". */
export async function renameTranscriptSpeakers(
  id: string,
  names: Record<string, string | null>,
): Promise<TranscriptRecord> {
  const response = await authFetch(
    `/api/inference/audio/transcripts/${encodeURIComponent(id)}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ speaker_names: names }),
    },
  );
  if (!response.ok) throw new Error(await readFastApiError(response));
  return (await response.json()) as TranscriptRecord;
}
