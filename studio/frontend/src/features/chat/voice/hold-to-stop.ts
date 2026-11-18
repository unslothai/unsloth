// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


export const HOLD_TO_STOP_MS = 600;
/** A press released inside this is a tap; a longer one that never armed does nothing. */
export const TAP_MAX_MS = 300;

export type HoldPhase = "idle" | "holding" | "armed";
export type HoldOutcome = "tap" | "stop" | "none";

export interface HoldTimers {
  now: () => number;
  setTimer: (fn: () => void, ms: number) => unknown;
  clearTimer: (handle: unknown) => void;
}

export interface HoldToStop {
  press: () => void;
  release: () => HoldOutcome;
  cancel: () => void;
  phase: () => HoldPhase;
  dispose: () => void;
}

const browserTimers: HoldTimers = {
  now: () => performance.now(),
  setTimer: (fn, ms) => setTimeout(fn, ms),
  clearTimer: (h) => clearTimeout(h as ReturnType<typeof setTimeout>),
};

export function createHoldToStop(
  onPhase: (phase: HoldPhase) => void,
  timers: HoldTimers = browserTimers,
  holdMs: number = HOLD_TO_STOP_MS,
  tapMaxMs: number = TAP_MAX_MS,
): HoldToStop {
  let phase: HoldPhase = "idle";
  let pressedAt = 0;
  let timer: unknown = null;

  const go = (next: HoldPhase) => {
    if (phase === next) return;
    phase = next;
    onPhase(next);
  };
  const clear = () => {
    if (timer !== null) {
      timers.clearTimer(timer);
      timer = null;
    }
  };

  return {
    press() {
      if (phase !== "idle") return;
      pressedAt = timers.now();
      go("holding");
      timer = timers.setTimer(() => {
        timer = null;
        if (phase === "holding") go("armed");
      }, holdMs);
    },
    release() {
      if (phase === "idle") return "none";
      const held = timers.now() - pressedAt;
      const was = phase;
      clear();
      go("idle");
      if (was === "armed") return "stop";
      return held <= tapMaxMs ? "tap" : "none";
    },
    cancel() {
      clear();
      go("idle");
    },
    phase: () => phase,
    dispose: clear,
  };
}
