// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Keep ordinary replies at paint cadence. Beyond 24 KiB, the real-model trace
// already contains enough rich content that 60 publication attempts per second
// create more mutation/reconciliation work than visible progress. 75 ms lands
// at 12.5 Hz on a 60 Hz display, inside the requested smooth 12-15 Hz band.
export const LONG_STREAM_PRESENTATION_CHARS = 24 * 1024;
export const LONG_STREAM_PRESENTATION_MS = 75;

type FrameHandle = number;
type BrowserTimerHandle = ReturnType<typeof setTimeout>;

type PresentationEnvironment<TimerHandle, Value> = {
  publish: (value: Value) => void;
  now: () => number;
  requestFrame: (callback: FrameRequestCallback) => FrameHandle;
  cancelFrame: (handle: FrameHandle) => void;
  setTimer: (callback: () => void, delay: number) => TimerHandle;
  clearTimer: (handle: TimerHandle) => void;
};

export type StreamingTextPresentationScheduler<Value> = {
  cancel: () => void;
  flush: (value: Value) => void;
  schedule: (sourceCharacters: number, value: Value) => void;
};

export function createStreamingTextPresentationScheduler<TimerHandle, Value>({
  publish,
  now,
  requestFrame,
  cancelFrame,
  setTimer,
  clearTimer,
}: PresentationEnvironment<TimerHandle, Value>): StreamingTextPresentationScheduler<Value> {
  let frame: FrameHandle | null = null;
  let timer: TimerHandle | null = null;
  let scheduledLong = false;
  let lastPublishedAt = Number.NEGATIVE_INFINITY;
  let pendingValue!: Value;
  let hasPending = false;

  const cancelScheduled = (): void => {
    if (frame !== null) {
      cancelFrame(frame);
      frame = null;
    }
    if (timer !== null) {
      clearTimer(timer);
      timer = null;
    }
  };

  const cancel = (): void => {
    cancelScheduled();
    hasPending = false;
  };

  const publishOnFrame = (): void => {
    frame = requestFrame(() => {
      frame = null;
      if (!hasPending) return;
      const value = pendingValue;
      hasPending = false;
      lastPublishedAt = now();
      publish(value);
    });
  };

  const schedule = (sourceCharacters: number, value: Value): void => {
    pendingValue = value;
    hasPending = true;
    const long = sourceCharacters >= LONG_STREAM_PRESENTATION_CHARS;
    if (frame !== null || timer !== null) {
      if (scheduledLong === long) return;
      // A replacement can turn a long reply into a short one. Do not leave its
      // first visible update waiting behind the long-tail timer.
      cancelScheduled();
    }
    scheduledLong = long;
    if (!long) {
      publishOnFrame();
      return;
    }

    const delay = Math.max(
      0,
      LONG_STREAM_PRESENTATION_MS - (now() - lastPublishedAt),
    );
    if (delay === 0) {
      publishOnFrame();
      return;
    }
    timer = setTimer(() => {
      timer = null;
      publishOnFrame();
    }, delay);
  };

  const flush = (value: Value): void => {
    cancelScheduled();
    hasPending = false;
    lastPublishedAt = now();
    publish(value);
  };

  return { cancel, flush, schedule };
}

type AfterPaintEnvironment<TimerHandle> = Pick<
  PresentationEnvironment<TimerHandle, never>,
  "requestFrame" | "cancelFrame" | "setTimer" | "clearTimer"
>;

const browserAfterPaintEnvironment = (): AfterPaintEnvironment<BrowserTimerHandle> => ({
  requestFrame: (callback) => requestAnimationFrame(callback),
  cancelFrame: (handle) => cancelAnimationFrame(handle),
  setTimer: (callback, delay) => setTimeout(callback, delay),
  clearTimer: (handle) => clearTimeout(handle),
});

// Two animation frames guarantee the plain, source-complete code block and the
// completion UI get a paint opportunity before a potentially expensive final
// TextMate pass. The zero-delay task then starts colourization outside the RAF.
export function scheduleAfterPaint(
  callback: () => void,
): () => void;
export function scheduleAfterPaint<TimerHandle>(
  callback: () => void,
  environment: AfterPaintEnvironment<TimerHandle>,
): () => void;
export function scheduleAfterPaint<TimerHandle = BrowserTimerHandle>(
  callback: () => void,
  environment: AfterPaintEnvironment<TimerHandle> =
    browserAfterPaintEnvironment() as unknown as AfterPaintEnvironment<TimerHandle>,
): () => void {
  let firstFrame: FrameHandle | null = null;
  let secondFrame: FrameHandle | null = null;
  let timer: TimerHandle | null = null;
  let cancelled = false;

  firstFrame = environment.requestFrame(() => {
    firstFrame = null;
    secondFrame = environment.requestFrame(() => {
      secondFrame = null;
      timer = environment.setTimer(() => {
        timer = null;
        if (!cancelled) callback();
      }, 0);
    });
  });

  return () => {
    cancelled = true;
    if (firstFrame !== null) environment.cancelFrame(firstFrame);
    if (secondFrame !== null) environment.cancelFrame(secondFrame);
    if (timer !== null) environment.clearTimer(timer);
  };
}
