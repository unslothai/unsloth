// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Images and Video pages hold their own `status` and re-read it on tab
// activation, not on a timer, so a model released from anywhere else leaves
// their controls claiming it is still loaded. This announces such a release so
// the owning page can re-sync while it is on screen. In lib/, not a feature,
// since both the emitter and the listeners are features.

export const MODEL_EJECTED_EVENT = "unsloth:model-ejected";

/** Which runtime was released. */
export type EjectedModelRuntime = "image" | "video";

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
