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

/**
 * What a `load-progress` endpoint reports. `null` is terminal once a load has
 * started: all three engines return it only for "nothing loading and nothing
 * loaded", and `begin_load` records its loading state before the POST answers,
 * so a read before the first byte moves says `downloading`. It is what a
 * cancelled or evicted load leaves behind -- the Images page's own poll treats
 * it the same way, and must, "else this loop spins forever".
 */
export type LoadPhase = "downloading" | "finalizing" | "ready" | "error" | null;

/** How often the settle poll asks; overridable so a test need not sleep. */
export const BACKGROUND_LOAD_POLL_MS = 2000;
/** Well past a healthy load-progress read, so only a real hang trips it. */
export const BACKGROUND_READ_TIMEOUT_MS = 10_000;
/**
 * How long a run of unreadable polls may last before the row is retired. Timed
 * from the last healthy read rather than from the start of the load: a 100 GB
 * video checkpoint on a slow link legitimately takes hours, and an absolute
 * deadline would have hidden a download that was still visibly progressing.
 */
export const BACKGROUND_STALL_TIMEOUT_MS = 60 * 60 * 1000;

/** Cadences, all optional: the defaults are what production uses. */
export type BackgroundLoadTiming = {
  pollMs?: number;
  readTimeoutMs?: number;
  stallMs?: number;
};

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
  readPhase: (signal: AbortSignal) => Promise<LoadPhase>,
  timing: BackgroundLoadTiming = {},
): Promise<T> {
  notifyModelLifecycle({ runtime, loading: true, model });
  let started = false;
  try {
    const result = await start();
    started = true;
    // Announce a second time, now that the POST has returned. That is the
    // instant the GPU arbiter has committed: acquire_for evicts whoever held
    // the GPU inside this call, ahead of a download that can run for hours, so
    // the first announcement was raised while the status it displaces was still
    // correct. Listeners that re-read another runtime need this edge, and the
    // rows this drives are keyed by runtime, so a repeat is a no-op for them.
    notifyModelLifecycle({ runtime, loading: true, model });
    void settleWhenLoadEnds(runtime, model, readPhase, timing);
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
      // An unreadable read is not proof the load ended: a restarting backend or
      // one dropped request would end the row early and hide a live load. That
      // is `undefined`, kept distinct from the `null` phase precisely so the two
      // are not conflated here.
      const phase = await boundedRead(readPhase, readTimeoutMs);
      if (phase === undefined) {
        // Only a sustained run of unreadable polls gives up. A load that is
        // still reporting progress is never abandoned, however long it takes.
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
 * One read, abandoned if it does not answer. A backend that accepts the
 * connection and never replies would otherwise park this `await` forever, so
 * the loop would never test its deadline again and the row would stay loading
 * for the life of the tab -- the deadline only bounds the loop if every turn of
 * it terminates.
 *
 * A plain AbortController rather than AbortSignal.timeout, which the older
 * WebKitGTK builds Tauri embeds do not have.
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
