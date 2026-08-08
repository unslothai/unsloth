// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// One announcement per model load or release, raised from the API call itself
// so every caller is covered without each page remembering to.
//
// Two things listen. The Images and Video pages hold their own `status` and
// re-read it on tab activation rather than on a timer, so a release from
// anywhere else leaves their controls claiming a model is still loaded. And the
// loaded models indicator polls every 5s, which is long enough that a load
// looked like it had not started: the toast said "loading" while the card said
// nothing. Both now hear the moment it happens.
//
// In lib/, not a feature, since the emitters and the listeners are both features.

export const MODEL_EJECTED_EVENT = "unsloth:model-ejected";
export const MODEL_LIFECYCLE_EVENT = "unsloth:model-lifecycle";

/** Which runtime was released. Only these two own a page holding its status. */
export type EjectedModelRuntime = "image" | "video";

/** Every runtime the indicator lists. */
export type ModelRuntime = "chat" | "image" | "video" | "stt";

export type ModelLifecycle = {
  runtime: ModelRuntime;
  /** True while the load is in flight, false once it settled either way. */
  loading: boolean;
  /** What is being loaded, for the row shown before any status confirms it. */
  model: string | null;
};

export function notifyModelEjected(runtime: EjectedModelRuntime): void {
  if (typeof window === "undefined") return;
  window.dispatchEvent(
    new CustomEvent(MODEL_EJECTED_EVENT, { detail: { runtime } }),
  );
}

/** Calls back when `runtime` is released elsewhere. Returns an unsubscriber. */
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

/** Calls back on every load start and finish, whichever runtime. */
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

/**
 * Announce a load around `run`, so the indicator shows it for the whole time
 * the toast does. Settles on failure too, or a failed load would leave the row
 * spinning until the next poll disagreed.
 */
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

/** What a `load-progress` endpoint reports; `null` before the first byte moves. */
export type LoadPhase = "downloading" | "finalizing" | "ready" | "error" | null;

/** How often the settle poll asks; overridable so a test need not sleep. */
export const BACKGROUND_LOAD_POLL_MS = 2000;
/** A backend that stops answering must not leave the row loading forever. */
export const BACKGROUND_LOAD_DEADLINE_MS = 60 * 60 * 1000;

/**
 * The same announcement for the two loads that only *start* the work: images
 * and video hand off to a background thread and return at once, while /status
 * keeps reporting `loaded: false` for the whole load. Settling on the POST would
 * flash the row for one round trip and then drop it, leaving the toast saying
 * "loading" alone for minutes -- the very gap this notice exists to close.
 *
 * So settle from the same `load-progress` endpoint the page toast watches. The
 * poll belongs to the load call rather than to a page, so the row still settles
 * when the user navigates away mid-load.
 */
export async function withBackgroundLoadNotice<T>(
  runtime: ModelRuntime,
  model: string | null,
  start: () => Promise<T>,
  readPhase: () => Promise<LoadPhase>,
  pollMs: number = BACKGROUND_LOAD_POLL_MS,
): Promise<T> {
  notifyModelLifecycle({ runtime, loading: true, model });
  let started = false;
  try {
    const result = await start();
    started = true;
    void settleWhenLoadEnds(runtime, model, readPhase, pollMs);
    return result;
  } finally {
    // A load that never started settles here; one that did settles from the
    // poll, so exactly one of the two paths ends the notice.
    if (!started) notifyModelLifecycle({ runtime, loading: false, model });
  }
}

async function settleWhenLoadEnds(
  runtime: ModelRuntime,
  model: string | null,
  readPhase: () => Promise<LoadPhase>,
  pollMs: number,
): Promise<void> {
  const deadline = Date.now() + BACKGROUND_LOAD_DEADLINE_MS;
  try {
    while (Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, pollMs));
      // An unreadable read is not proof the load ended: a restarting backend or
      // one dropped request would end the row early and hide a live load. Only
      // a terminal phase settles it, and the deadline covers a backend that
      // never answers again.
      const phase = await readPhase().catch(() => undefined);
      if (phase === "ready" || phase === "error") return;
    }
  } finally {
    notifyModelLifecycle({ runtime, loading: false, model });
  }
}
