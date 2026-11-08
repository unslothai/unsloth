// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Voice mode's speech check on audio.cpp: the loudness gate says when a clip ended, this says
// whether it holds a voice at all.

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
  /** Voiced audio in the clip, summed over segments. */
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
    // not JSON
  }
  return response.statusText || `HTTP ${response.status}`;
}

// Every load in flight. A first load downloads before it starts the server, so an unload sent
// meanwhile would land first and leave the detector running with voice off.
const pendingLoads = new Set<Promise<unknown>>();

/** Download if needed and start the detector; resolves to the loaded model id. */
export function loadVadModel(model: string, signal?: AbortSignal): Promise<string> {
  const run = startVadModel(model, signal);
  const tracked: Promise<unknown> = run
    .catch(() => {})
    .finally(() => pendingLoads.delete(tracked));
  pendingLoads.add(tracked);
  return run;
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

/** Stop the detector, after any load in flight has settled so that load cannot outlive it. */
export async function unloadVadModel(): Promise<void> {
  await Promise.all([...pendingLoads]);
  await authFetch("/api/inference/audio/vad/unload", { method: "POST" }).catch(
    () => {},
  );
}

export async function detectSpeech(
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
