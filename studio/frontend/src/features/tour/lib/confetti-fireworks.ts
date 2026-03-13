// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

type FireworksOpts = {
  durationMs?: number;
  intervalMs?: number;
  zIndex?: number;
};

export async function fireConfettiFireworks(opts: FireworksOpts = {}) {
  try {
    if (typeof window === "undefined") return;

    const prefersReduce =
      window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches ?? false;
    if (prefersReduce) return;

    const confetti = (await import("canvas-confetti")).default;

    const duration = opts.durationMs ?? 1200;
    const intervalMs = opts.intervalMs ?? 240;
    const animationEnd = Date.now() + duration;
    const defaults = {
      startVelocity: 28,
      spread: 360,
      ticks: 58,
      zIndex: opts.zIndex ?? 99999,
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

      confetti({
        ...defaults,
        particleCount,
        origin: { x: randomInRange(0.12, 0.3), y: Math.random() - 0.2 },
      });
      confetti({
        ...defaults,
        particleCount,
        origin: { x: randomInRange(0.7, 0.88), y: Math.random() - 0.2 },
      });
    }, intervalMs);
  } catch {
    // best-effort
  }
}
