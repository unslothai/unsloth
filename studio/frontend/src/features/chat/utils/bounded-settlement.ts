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

/** Wait for a batch within a bound while preserving every failure observed before it expires. */
export async function waitForSettledBatch(
  work: readonly Promise<unknown>[],
  ms: number,
): Promise<void> {
  let failed = false;
  let firstFailure: unknown;
  const settled = Promise.all(
    work.map((item) =>
      item.then(
        () => undefined,
        (error: unknown) => {
          if (!failed) {
            failed = true;
            firstFailure = error;
          }
        },
      ),
    ),
  );
  await settleWithin(settled, ms);
  if (failed) {
    throw firstFailure;
  }
}
