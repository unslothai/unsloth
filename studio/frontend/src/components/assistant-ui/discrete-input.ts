// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const DISCRETE_INPUT_EVENTS = [
  "keydown",
  "pointerdown",
  "touchstart",
] as const;

export const DISCRETE_INPUT_WINDOW_MS = 50;

export function discreteInputPending(
  lastInputAt: number,
  now: number,
  windowMs: number = DISCRETE_INPUT_WINDOW_MS,
): boolean {
  return now - lastInputAt < windowMs;
}

export type DiscreteInputTracker = {
  pending(): boolean;
  dispose(): void;
};

export function createDiscreteInputTracker(
  target: EventTarget,
  now: () => number = () => performance.now(),
): DiscreteInputTracker {
  let lastInputAt = Number.NEGATIVE_INFINITY;
  const record = () => {
    lastInputAt = now();
  };
  for (const type of DISCRETE_INPUT_EVENTS) {
    target.addEventListener(type, record, { capture: true, passive: true });
  }
  return {
    pending: () => discreteInputPending(lastInputAt, now()),
    dispose: () => {
      for (const type of DISCRETE_INPUT_EVENTS) {
        target.removeEventListener(type, record, { capture: true });
      }
    },
  };
}
