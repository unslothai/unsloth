// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Fetch helpers for the Audio page. TTS model lifecycle goes through the shared
// chat load API (@/features/chat); STT goes through the dictation sidecar
// helpers. What lives here is the audio-only surface: generation and the clip
// gallery.

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

async function parseJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    throw new Error(await readFastApiError(response));
  }
  return (await response.json()) as T;
}

export interface GeneratedAudio {
  /** Base64 WAV payload. */
  data: string;
  format: string;
  sample_rate: number;
}

export interface GenerateAudioResponse {
  model: string;
  audio: GeneratedAudio;
  /** Exact gallery record created for this request; null when persistence failed. */
  clip_id?: string | null;
}

/** Sampling knobs the /audio/generate route reads from its chat-shaped body. */
export interface GenerateAudioOptions {
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  signal?: AbortSignal;
}

/** Synthesize speech with the loaded TTS model. Resolves to base64 WAV. */
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
  /** Auth-protected WAV path; fetch through fetchClipObjectUrl, never a bare src. */
  url: string;
  prompt: string;
  model: string;
  audio_type: string;
  sample_rate: number;
  duration_s: number;
  created_at: string;
}

export interface AudioGalleryListResponse {
  audio: AudioGalleryClip[];
  has_more: boolean;
}

export async function listAudioGallery(
  offset: number,
  limit: number,
): Promise<AudioGalleryListResponse> {
  const response = await authFetch(
    `/api/inference/audio/gallery?offset=${offset}&limit=${limit}`,
  );
  return parseJson<AudioGalleryListResponse>(response);
}

export async function deleteAudioClip(id: string): Promise<void> {
  const response = await authFetch(
    `/api/inference/audio/gallery/${encodeURIComponent(id)}`,
    { method: "DELETE" },
  );
  if (!response.ok) throw new Error(await readFastApiError(response));
}

export async function clearAudioGallery(): Promise<number> {
  const response = await authFetch("/api/inference/audio/gallery", {
    method: "DELETE",
  });
  const body = await parseJson<{ removed: number }>(response);
  return body.removed;
}

/** The clip bytes as an object URL (the gallery is auth-protected, so a bare
 *  <audio src> cannot reach it). The caller owns revocation. */
export async function fetchClipObjectUrl(
  url: string,
): Promise<{ url: string; bytes: number }> {
  const response = await authFetch(url);
  if (!response.ok) throw new Error(await readFastApiError(response));
  const blob = await response.blob();
  return { url: URL.createObjectURL(blob), bytes: blob.size };
}
