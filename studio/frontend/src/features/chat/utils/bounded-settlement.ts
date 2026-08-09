// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

const TIMED_OUT = Symbol("timed-out");

function deadline(ms: number) {
  let timer: ReturnType<typeof setTimeout>;
  return {
    promise: new Promise<typeof TIMED_OUT>((resolve) => {
      timer = setTimeout(() => resolve(TIMED_OUT), Math.max(ms, 0));
    }),
    cancel: () => clearTimeout(timer),
  };
}

export async function settleWithin(
  work: Promise<unknown>,
  ms: number,
): Promise<void> {
  const bound = deadline(ms);
  const ignored = work.then(
    () => undefined,
    () => undefined,
  );
  await Promise.race([ignored, bound.promise]);
  bound.cancel();
}

export async function waitForSettledOrRunFallback<T>(
  work: Promise<T>,
  fallback: () => Promise<T>,
  ms: number,
): Promise<T> {
  const bound = deadline(ms);
  try {
    const result = await Promise.race([work, bound.promise]);
    return result === TIMED_OUT ? await fallback() : result;
  } finally {
    bound.cancel();
  }
}
