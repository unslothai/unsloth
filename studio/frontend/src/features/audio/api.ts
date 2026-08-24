// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

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
}

export interface GenerateAudioOptions {
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  signal?: AbortSignal;
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
  archived?: boolean;
}

export interface AudioGalleryListResponse {
  audio: AudioGalleryClip[];
  has_more: boolean;
  next_before_mtime: number | null;
  next_before_id: string | null;
}

export async function listAudioGallery(
  offset: number,
  limit: number,
  before?: { mtime: number; id: string } | null,
  archived = false,
): Promise<AudioGalleryListResponse> {
  const cursor = before
    ? `&before_mtime=${encodeURIComponent(before.mtime)}&before_id=${encodeURIComponent(before.id)}`
    : "";
  const response = await authFetch(
    `/api/inference/audio/gallery?offset=${offset}&limit=${limit}&archived=${archived}${cursor}`,
  );
  return parseJson<AudioGalleryListResponse>(response);
}

export async function setAudioClipFlags(
  id: string,
  flags: { archived?: boolean },
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

export async function fetchClipObjectUrl(
  url: string,
): Promise<{ url: string; bytes: number }> {
  const response = await authFetch(url);
  if (!response.ok) throw new Error(await readFastApiError(response));
  const blob = await response.blob();
  return { url: URL.createObjectURL(blob), bytes: blob.size };
}
