// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** What the page reports (agent/page-agent.ts `status`); null when it didn't answer. */
export type PageStatus = { requests?: number; quietMs?: number } | null;

export type PageWait = {
  /** Give up after about this long (one more status call can run past it). */
  maxMs: number;
  /** A DOM that keeps changing with no request open is waited on only this long. */
  changingMaxMs: number;
  quietMs: number;
  now: () => number;
  sleep: (ms: number) => Promise<unknown>;
  /** Open requests started since the action, and how long the DOM has been quiet. */
  status: () => Promise<PageStatus>;
  moved: () => boolean;
  aborted?: () => boolean;
};

const POLL_MS = 120;

/** "settled", "loading" (requests still open at `maxMs`), or "moved". */
export async function waitForPage(wait: PageWait): Promise<"settled" | "loading" | "moved"> {
  const started = wait.now();
  // The changing-DOM allowance runs from when no request is open: a response's own DOM work comes after it.
  let free: number | null = null;
  while (!wait.aborted?.()) {
    if (wait.moved()) return "moved";
    const status = await wait.status();
    if (!status) return "settled";
    const loading = (status.requests ?? 0) > 0;
    const now = wait.now();
    const waited = now - started;
    free = loading ? null : (free ?? now);
    if (free !== null && ((status.quietMs ?? 0) >= wait.quietMs || now - free >= wait.changingMaxMs)) return "settled";
    if (waited >= wait.maxMs) return loading ? "loading" : "settled";
    await wait.sleep(POLL_MS);
  }
  return "settled";
}
