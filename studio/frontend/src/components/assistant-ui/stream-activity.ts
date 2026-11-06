// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export const STREAM_COOLDOWN_MS = 2_000;

type RunningState = { runningByThreadId: Record<string, boolean> };

export type StreamActivityStore = {
  getState(): RunningState;
  subscribe(listener: (next: RunningState, previous: RunningState) => void): () => void;
};

export type StreamActivity = { active(): boolean };

const anyRunning = (running: Record<string, boolean>): boolean =>
  Object.values(running).some(Boolean);

export function createStreamActivity(
  store: StreamActivityStore,
  now: () => number = () => performance.now(),
): StreamActivity {
  let lastRunEndedAt = Number.NEGATIVE_INFINITY;
  store.subscribe((next, previous) => {
    if (
      !anyRunning(next.runningByThreadId)
      && anyRunning(previous.runningByThreadId)
    ) {
      lastRunEndedAt = now();
    }
  });
  return {
    active: () =>
      anyRunning(store.getState().runningByThreadId)
      || now() - lastRunEndedAt < STREAM_COOLDOWN_MS,
  };
}
