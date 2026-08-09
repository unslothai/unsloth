// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Resolve when `work` settles, or after `ms`, whichever comes first. */
export function settleWithin(
  work: Promise<unknown>,
  ms: number,
): Promise<void> {
  return new Promise<void>((resolve) => {
    const timer = setTimeout(resolve, Math.max(ms, 0));
    const finish = () => {
      clearTimeout(timer);
      resolve();
    };
    work.then(finish, finish);
  });
}

/** Report whether work settled within the bound, preserving a rejection that arrives in time. */
export function settlesWithin(
  work: Promise<unknown>,
  ms: number,
): Promise<boolean> {
  return new Promise<boolean>((resolve, reject) => {
    const timer = setTimeout(() => resolve(false), Math.max(ms, 0));
    work.then(
      () => {
        clearTimeout(timer);
        resolve(true);
      },
      (error: unknown) => {
        clearTimeout(timer);
        reject(error);
      },
    );
  });
}

/** If queued work misses its bound, require an independent fallback to confirm within one bound. */
export async function waitForSettledOrRunFallback(
  work: Promise<unknown>,
  fallback: () => Promise<unknown>,
  ms: number,
): Promise<void> {
  if (await settlesWithin(work, ms)) {
    return;
  }
  if (!(await settlesWithin(fallback(), ms))) {
    throw new Error("Timed out waiting for fallback work");
  }
}
