// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Raised from the API call itself so every caller is covered. Images/Video pages and the loaded
// models indicator listen, since they would otherwise miss releases and loads made elsewhere.

export const MODEL_EJECTED_EVENT = "unsloth:model-ejected";
export const MODEL_LIFECYCLE_EVENT = "unsloth:model-lifecycle";

export type EjectedModelRuntime = "image" | "video";

/** "tts" shares the chat slot; it is named apart so chat can tell loads it did not start. */
export type ModelRuntime = "chat" | "image" | "video" | "stt" | "tts";

export type ModelLifecycle = {
  runtime: ModelRuntime;
  loading: boolean;
  model: string | null;
};

export function notifyModelEjected(runtime: EjectedModelRuntime): void {
  if (typeof window === "undefined") return;
  window.dispatchEvent(
    new CustomEvent(MODEL_EJECTED_EVENT, { detail: { runtime } }),
  );
}

export function subscribeModelEjected(
  runtime: EjectedModelRuntime,
  onEjected: () => void,
): () => void {
  if (typeof window === "undefined") return () => {};
  const handler = (event: Event) => {
    const detail = (event as CustomEvent<{ runtime?: string }>).detail;
    if (detail?.runtime === runtime) onEjected();
  };
  window.addEventListener(MODEL_EJECTED_EVENT, handler);
  return () => window.removeEventListener(MODEL_EJECTED_EVENT, handler);
}

export function notifyModelLifecycle(detail: ModelLifecycle): void {
  if (typeof window === "undefined") return;
  window.dispatchEvent(new CustomEvent(MODEL_LIFECYCLE_EVENT, { detail }));
}

export function subscribeModelLifecycle(
  onChange: (detail: ModelLifecycle) => void,
): () => void {
  if (typeof window === "undefined") return () => {};
  const handler = (event: Event) => {
    const detail = (event as CustomEvent<ModelLifecycle>).detail;
    if (detail?.runtime) onChange(detail);
  };
  window.addEventListener(MODEL_LIFECYCLE_EVENT, handler);
  return () => window.removeEventListener(MODEL_LIFECYCLE_EVENT, handler);
}

/** Settles on failure too, or a failed load would spin until the next poll. */
export async function withModelLoadNotice<T>(
  runtime: ModelRuntime,
  model: string | null,
  run: () => Promise<T>,
): Promise<T> {
  notifyModelLifecycle({ runtime, loading: true, model });
  try {
    return await run();
  } finally {
    notifyModelLifecycle({ runtime, loading: false, model });
  }
}

/** `null` is terminal once a load started: backends record loading state before the POST answers. */
export type LoadPhase = "downloading" | "finalizing" | "ready" | "error" | null;

export const BACKGROUND_LOAD_POLL_MS = 2000;
export const BACKGROUND_READ_TIMEOUT_MS = 10_000;
/** Timed from the last healthy read, not load start: huge downloads legitimately take hours. */
export const BACKGROUND_STALL_TIMEOUT_MS = 60 * 60 * 1000;

export type BackgroundLoadTiming = {
  pollMs?: number;
  readTimeoutMs?: number;
  stallMs?: number;
};

/**
 * Image and video loads return at once and load in the background, so settle from
 * `load-progress` instead of the POST. The poll survives navigation away from the page.
 */
export async function withBackgroundLoadNotice<T>(
  runtime: ModelRuntime,
  model: string | null,
  start: () => Promise<T>,
  readPhase: (signal: AbortSignal) => Promise<LoadPhase>,
  timing: BackgroundLoadTiming = {},
): Promise<T> {
  notifyModelLifecycle({ runtime, loading: true, model });
  let started = false;
  try {
    const result = await start();
    started = true;
    // Re-announce after the POST: that is when the GPU arbiter has evicted the previous holder.
    // Repeats are no-ops for runtime-keyed rows.
    notifyModelLifecycle({ runtime, loading: true, model });
    void settleWhenLoadEnds(runtime, model, readPhase, timing);
    return result;
  } finally {
    // Exactly one of the two paths ends the notice.
    if (!started) notifyModelLifecycle({ runtime, loading: false, model });
  }
}

async function settleWhenLoadEnds(
  runtime: ModelRuntime,
  model: string | null,
  readPhase: (signal: AbortSignal) => Promise<LoadPhase>,
  timing: BackgroundLoadTiming,
): Promise<void> {
  const pollMs = timing.pollMs ?? BACKGROUND_LOAD_POLL_MS;
  const readTimeoutMs = timing.readTimeoutMs ?? BACKGROUND_READ_TIMEOUT_MS;
  const stallMs = timing.stallMs ?? BACKGROUND_STALL_TIMEOUT_MS;
  let lastHealthy = Date.now();
  try {
    for (;;) {
      await new Promise((resolve) => setTimeout(resolve, pollMs));
      // An unreadable read (`undefined`) is not the terminal `null` phase; do not end the row on it.
      const phase = await boundedRead(readPhase, readTimeoutMs);
      if (phase === undefined) {
        if (Date.now() - lastHealthy >= stallMs) return;
        continue;
      }
      if (phase !== "downloading" && phase !== "finalizing") return;
      lastHealthy = Date.now();
    }
  } finally {
    notifyModelLifecycle({ runtime, loading: false, model });
  }
}

/**
 * Bounds each read so a silent backend cannot park the loop.
 * AbortController, since older WebKitGTK lacks AbortSignal.timeout.
 */
async function boundedRead(
  readPhase: (signal: AbortSignal) => Promise<LoadPhase>,
  readTimeoutMs: number,
): Promise<LoadPhase | undefined> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), readTimeoutMs);
  try {
    return await readPhase(controller.signal);
  } catch {
    return undefined;
  } finally {
    clearTimeout(timer);
  }
}
