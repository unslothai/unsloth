// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import type { CreateTypes as ConfettiInstance } from "canvas-confetti";

type FireworksOpts = {
  durationMs?: number;
  intervalMs?: number;
  zIndex?: number;
};

// Studio CSP (`script-src 'self'`) blocks canvas-confetti's default
// blob: worker, so we mount a dedicated overlay canvas once and drive it
// via `confetti.create(...)` with `useWorker: false`. The shared instance
// is reused across calls so we don't leak canvases per tour completion.
//
// Stacking note: when canvas-confetti runs against a caller-provided
// canvas, the per-fire `zIndex` option no longer drives stacking; the
// canvas element's own CSS `z-index` does. We therefore apply
// `opts.zIndex` (or the default) to the shared canvas's style on each
// call so callers that want to lower/raise the fireworks layer still
// get that behavior.
const DEFAULT_FIREWORKS_Z_INDEX = 99999;
let _sharedCanvas: HTMLCanvasElement | null = null;
let _sharedFire: ConfettiInstance | null = null;
async function getSharedFire(): Promise<ConfettiInstance | null> {
  if (typeof document === "undefined") return null;
  if (_sharedFire) return _sharedFire;
  const confetti = (await import("canvas-confetti")).default;
  _sharedCanvas = document.createElement("canvas");
  _sharedCanvas.style.cssText =
    `position:fixed;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:${DEFAULT_FIREWORKS_Z_INDEX}`;
  document.body.appendChild(_sharedCanvas);
  _sharedFire = confetti.create(_sharedCanvas, {
    resize: true,
    useWorker: false,
  });
  return _sharedFire;
}

export async function fireConfettiFireworks(opts: FireworksOpts = {}) {
  try {
    if (typeof window === "undefined") return;

    const prefersReduce =
      window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches ?? false;
    if (prefersReduce) return;

    const fire = await getSharedFire();
    if (!fire || !_sharedCanvas) return;

    // Honor `opts.zIndex` even though canvas-confetti's per-fire `zIndex`
    // is ignored for caller-provided canvases (see note above): we drive
    // stacking via the shared canvas's CSS instead of the per-fire option.
    _sharedCanvas.style.zIndex = String(opts.zIndex ?? DEFAULT_FIREWORKS_Z_INDEX);

    const duration = opts.durationMs ?? 1200;
    const intervalMs = opts.intervalMs ?? 240;
    const animationEnd = Date.now() + duration;
    const defaults = {
      startVelocity: 28,
      spread: 360,
      ticks: 58,
      disableForReducedMotion: true,
    } as const;

    const randomInRange = (min: number, max: number) =>
      Math.random() * (max - min) + min;

    const interval = window.setInterval(() => {
      const timeLeft = animationEnd - Date.now();
      if (timeLeft <= 0) {
        window.clearInterval(interval);
        return;
      }

      const particleCount = Math.max(
        10,
        Math.floor(36 * (timeLeft / duration)),
      );

      fire({
        ...defaults,
        particleCount,
        origin: { x: randomInRange(0.12, 0.3), y: Math.random() - 0.2 },
      });
      fire({
        ...defaults,
        particleCount,
        origin: { x: randomInRange(0.7, 0.88), y: Math.random() - 0.2 },
      });
    }, intervalMs);
  } catch {
    // best-effort
  }
}
