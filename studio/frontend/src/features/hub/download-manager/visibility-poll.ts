// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface VisibilityPollTargets {
  document: Pick<EventTarget, "addEventListener" | "removeEventListener"> & {
    readonly hidden: boolean;
  };
  setTimeout: (callback: () => void, ms: number) => unknown;
  clearTimeout: (handle: unknown) => void;
}

export interface VisibilityPoll {
  /** Run now unless a run is in flight (that run re-arms when it settles) or the chain ended. */
  refresh: () => void;
  /** Run now; an in-flight run no longer re-arms, so the chain stays single. */
  restart: () => void;
  stop: () => void;
}

function browserTargets(): VisibilityPollTargets {
  return {
    document,
    setTimeout: (callback, ms) => window.setTimeout(callback, ms),
    clearTimeout: (handle) => window.clearTimeout(handle as number),
  };
}

/**
 * `run` resolves false to end the chain until the next restart, and must not reject. `delay(hidden)` spaces the runs;
 * null parks a hidden tab until it is shown. Becoming visible refreshes at once unless
 * `wakeOnVisible` is false, which then only suits a `delay` that never parks.
 */
export function startVisibilityPoll(
  run: () => Promise<boolean>,
  delay: (hidden: boolean) => number | null,
  options: { wakeOnVisible?: boolean; targets?: VisibilityPollTargets } = {},
): VisibilityPoll {
  const targets = options.targets ?? browserTargets();
  const wakeOnVisible = options.wakeOnVisible ?? true;
  let timer: unknown = null;
  let token = 0;
  let inFlight = false;
  let ended = false;
  let stopped = false;
  const clear = () => {
    if (timer === null) return;
    targets.clearTimeout(timer);
    timer = null;
  };
  const restart = () => {
    if (stopped) return;
    clear();
    const mine = ++token;
    inFlight = true;
    ended = false;
    void run().then((again) => {
      if (mine !== token) return;
      inFlight = false;
      ended = !again;
      if (stopped || ended) return;
      const ms = delay(targets.document.hidden);
      if (ms !== null) timer = targets.setTimeout(restart, ms);
    });
  };
  const refresh = () => {
    if (!inFlight && !ended) restart();
  };
  const onVisibility = () => {
    if (!targets.document.hidden) refresh();
  };
  if (wakeOnVisible)
    targets.document.addEventListener("visibilitychange", onVisibility);
  return {
    refresh,
    restart,
    stop: () => {
      stopped = true;
      token += 1;
      inFlight = false;
      clear();
      if (wakeOnVisible)
        targets.document.removeEventListener("visibilitychange", onVisibility);
    },
  };
}
