// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import type { CreateTypes as ConfettiInstance } from "canvas-confetti";

import { prefersReducedMotion } from "@/features/settings";

type FireworksOpts = {
  durationMs?: number;
  intervalMs?: number;
  zIndex?: number;
};

// CSP blocks canvas-confetti's default blob: worker, so reuse a single overlay
// canvas via `confetti.create(..., { useWorker: false })`. Caller canvases
// ignore per-fire `zIndex`; stacking is driven by `_sharedCanvas.style.zIndex`.
const DEFAULT_FIREWORKS_Z_INDEX = 99999;
let _sharedCanvas: HTMLCanvasElement | null = null;
let _sharedFire: ConfettiInstance | null = null;
// Cache the init promise so concurrent callers share one import + canvas.
let _sharedFirePromise: Promise<ConfettiInstance | null> | null = null;
function getSharedFire(): Promise<ConfettiInstance | null> {
  if (typeof document === "undefined") return Promise.resolve(null);
  if (_sharedFire) return Promise.resolve(_sharedFire);
  if (_sharedFirePromise) return _sharedFirePromise;
  _sharedFirePromise = (async () => {
    try {
      const confetti = (await import("canvas-confetti")).default;
      const canvas = document.createElement("canvas");
      canvas.style.cssText =
        `position:fixed;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:${DEFAULT_FIREWORKS_Z_INDEX}`;
      document.body.appendChild(canvas);
      _sharedCanvas = canvas;
      _sharedFire = confetti.create(canvas, {
        resize: true,
        useWorker: false,
      });
      return _sharedFire;
    } catch (err) {
      _sharedFirePromise = null;
      throw err;
    }
  })();
  return _sharedFirePromise;
}

export async function fireConfettiFireworks(opts: FireworksOpts = {}) {
  try {
    if (typeof window === "undefined") return;

    // Honor Appearance > Reduce motion (on/off) first, then the OS preference.
    if (prefersReducedMotion()) return;

    const fire = await getSharedFire();
    if (!fire || !_sharedCanvas) return;

    // Per-fire zIndex is ignored on a shared canvas; drive stacking via CSS.
    _sharedCanvas.style.zIndex = String(opts.zIndex ?? DEFAULT_FIREWORKS_Z_INDEX);

    const duration = opts.durationMs ?? 1200;
    const intervalMs = opts.intervalMs ?? 240;
    const animationEnd = Date.now() + duration;
    const defaults = {
      startVelocity: 28,
      spread: 360,
      ticks: 58,
      // Reduce motion is enforced above (incl. the in-app "off" override), so
      // don't let the library re-suppress purely on the OS preference.
      disableForReducedMotion: false,
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
