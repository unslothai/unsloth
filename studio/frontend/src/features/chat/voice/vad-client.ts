// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { AUDIO_CPP_REPO } from "@/features/audio/audio-cpp-catalog";
import { authFetch } from "@/features/auth";

// Earshot ships with Studio and runs on the CPU. Mirrors DEFAULT_VAD_MODEL in vad_audiocpp_sidecar.py.
export const VOICE_DEFAULT_VAD_MODEL = "builtin/earshot-vad";
// The Hub detector every supported audio runtime runs.
const HUB_VAD_MODEL = `${AUDIO_CPP_REPO}/PulseVAD-GGUF`;
const BUILTIN_PREFIX = "builtin/";

/** A detector that needs no download: it ships with Studio or with the audio runtime. */
export interface VadDetector {
  id: string;
  label: string;
  available: boolean;
}

const isBuiltinVad = (id: string) => id.toLowerCase().startsWith(BUILTIN_PREFIX);

/**
 * The detector to use when this install cannot run `pick`: the default for another built-in, the
 * Hub detector for the default (an audio runtime older than Earshot), none for a Hub pick.
 */
export function vadFallback(pick: string): string | null {
  if (!isBuiltinVad(pick)) return null;
  return pick.toLowerCase() === VOICE_DEFAULT_VAD_MODEL ? HUB_VAD_MODEL : VOICE_DEFAULT_VAD_MODEL;
}

/** The detector's name for display: the server's label, else the id's last part ("Earshot VAD"). */
export function vadDisplayName(id: string, detectors: readonly VadDetector[] = []): string {
  const known = detectors.find((d) => d.id === id.toLowerCase());
  if (known) return known.label;
  const leaf = id.split(/[\\/]/).filter(Boolean).pop() ?? id;
  if (!isBuiltinVad(id)) return leaf.replace(/-GGUF$/i, "");
  return leaf
    .split("-")
    .map((word) => (word === "vad" ? "VAD" : word.charAt(0).toUpperCase() + word.slice(1)))
    .join(" ");
}

export async function fetchVadDetectors(signal?: AbortSignal): Promise<VadDetector[]> {
  const response = await authFetch("/api/inference/audio/vad/status", { signal });
  if (!response.ok) return [];
  const data = (await response.json()) as { detectors?: VadDetector[] };
  return data.detectors ?? [];
}

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
  /** Voiced time this detector needs before a clip counts as speech; absent on older servers. */
  bars?: { utterance_ms: number; barge_ms: number };
}

// The bars a server that sends none is held to (the Hub detectors'). The barge-in probe holds only
// ~250 ms of voice, so the full bar failed its first check every time.
const VAD_MIN_SPEECH_MS = 300;
const BARGE_VAD_MIN_SPEECH_MS = 200;

/** Voiced time a clip needs to count as speech: the detector's own bar, the shorter one for a barge-in probe. */
export function minSpeechMs(bars: SpeechDetection["bars"], barge: boolean): number {
  return barge
    ? (bars?.barge_ms ?? BARGE_VAD_MIN_SPEECH_MS)
    : (bars?.utterance_ms ?? VAD_MIN_SPEECH_MS);
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

// 501 and 422 mean this build or model has no VAD: the caller falls back instead of retrying.
async function assertOk(response: Response): Promise<void> {
  if (response.ok) return;
  const message = await detail(response);
  if (response.status === 501 || response.status === 422) throw new VadUnavailableError(message);
  throw new Error(message);
}

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
  await assertOk(response);
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
  await assertOk(response);
  return (await response.json()) as SpeechDetection;
}
