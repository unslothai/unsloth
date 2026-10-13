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
  /** Enough quiet when the page counts its requests and none was open during the wait. */
  idleQuietMs?: number;
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
  // After a request closes, quiet and the changing-DOM allowance count from then: its DOM work comes after it.
  let free: number | null = null;
  let opened = false;
  let counted = true;
  while (!wait.aborted?.()) {
    if (wait.moved()) return "moved";
    const status = await wait.status();
    if (!status) return "settled";
    const loading = (status.requests ?? 0) > 0;
    const now = wait.now();
    const waited = now - started;
    if (loading) opened = true;
    if (status.requests === undefined) counted = false;
    free = loading ? null : (free ?? (opened ? now : started));
    const quiet = opened && free !== null ? Math.min(status.quietMs ?? 0, now - free) : (status.quietMs ?? 0);
    const needed = opened || !counted ? wait.quietMs : Math.min(wait.quietMs, wait.idleQuietMs ?? wait.quietMs);
    if (free !== null && (quiet >= needed || now - free >= wait.changingMaxMs)) return "settled";
    if (waited >= wait.maxMs) return loading ? "loading" : "settled";
    await wait.sleep(POLL_MS);
  }
  return "settled";
}
