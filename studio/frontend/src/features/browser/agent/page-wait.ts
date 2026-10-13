// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Waiting for a page to finish what an action or a load set off, before the agent reads it. */

/** What the page reports (agent/page-agent.ts `status`); null when it didn't answer. */
export type PageStatus = { pending?: number; quietMs?: number } | null;

export type PageWait = {
  /** Give up after about this long (one more status call can run past it). */
  maxMs: number;
  /** A DOM that keeps changing with no request open is waited on only this long. */
  changingMaxMs: number;
  quietMs: number;
  now: () => number;
  sleep: (ms: number) => Promise<unknown>;
  /** Requests the page has open that it started since the action went out, however long they have
   *  been open, and how long its DOM has been quiet. */
  status: () => Promise<PageStatus>;
  /** The page navigated or closed: the caller handles that. */
  moved: () => boolean;
  aborted?: () => boolean;
};

const POLL_MS = 120;

/** "settled", "loading" (requests still open at `maxMs`), or "moved". */
export async function waitForPage(wait: PageWait): Promise<"settled" | "loading" | "moved"> {
  const started = wait.now();
  while (!wait.aborted?.()) {
    if (wait.moved()) return "moved";
    const status = await wait.status();
    if (!status) return "settled";
    const loading = (status.pending ?? 0) > 0;
    const waited = wait.now() - started;
    if (!loading && ((status.quietMs ?? 0) >= wait.quietMs || waited >= wait.changingMaxMs)) return "settled";
    if (waited >= wait.maxMs) return loading ? "loading" : "settled";
    await wait.sleep(POLL_MS);
  }
  return "settled";
}
