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
