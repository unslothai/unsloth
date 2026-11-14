// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import { AUDIO_CPP_REPO } from "@/features/audio/audio-cpp-catalog";
import { authFetch } from "@/features/auth";

export const VOICE_DEFAULT_VAD_MODEL = `${AUDIO_CPP_REPO}/PulseVAD-GGUF`;

export interface SpeechSegment {
  start_s: number;
  end_s: number;
  confidence: number;
}

export interface SpeechDetection {
  segments: SpeechSegment[];
  speech_ms: number;
  sample_rate: number;
  model: string;
}

export class VadUnavailableError extends Error {}

async function detail(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string") return body.detail;
  } catch {
  }
  return response.statusText || `HTTP ${response.status}`;
}

// A load (or a detect, which loads too) still in flight would restart the detector after an unload.
const pendingLoads = new Set<Promise<unknown>>();

function track<T>(run: Promise<T>): Promise<T> {
  const tracked: Promise<unknown> = run
    .catch(() => {})
    .finally(() => pendingLoads.delete(tracked));
  pendingLoads.add(tracked);
  return run;
}

export function loadVadModel(model: string, signal?: AbortSignal): Promise<string> {
  return track(startVadModel(model, signal));
}

async function startVadModel(model: string, signal?: AbortSignal): Promise<string> {
  const response = await authFetch("/api/inference/audio/vad/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model }),
    signal,
  });
  if (!response.ok) {
    const message = await detail(response);
    if (response.status === 501 || response.status === 422)
      throw new VadUnavailableError(message);
    throw new Error(message);
  }
  const data = (await response.json()) as { loaded_model?: string };
  return data.loaded_model ?? model;
}

export async function unloadVadModel(): Promise<void> {
  await Promise.all([...pendingLoads]);
  await authFetch("/api/inference/audio/vad/unload", { method: "POST" }).catch(
    () => {},
  );
}

export function detectSpeech(
  wav: Blob,
  model: string,
  signal?: AbortSignal,
): Promise<SpeechDetection> {
  return track(runDetect(wav, model, signal));
}

async function runDetect(
  wav: Blob,
  model: string,
  signal?: AbortSignal,
): Promise<SpeechDetection> {
  const params = new URLSearchParams({ model });
  const response = await authFetch(
    `/api/inference/audio/vad/detect?${params}`,
    {
      method: "POST",
      headers: { "Content-Type": "audio/wav" },
      body: wav,
      signal,
    },
  );
  if (!response.ok) {
    const message = await detail(response);
    if (response.status === 501 || response.status === 422)
      throw new VadUnavailableError(message);
    throw new Error(message);
  }
  return (await response.json()) as SpeechDetection;
}
