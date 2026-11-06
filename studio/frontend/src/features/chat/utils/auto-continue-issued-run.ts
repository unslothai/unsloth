// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { AutoContinueIssuedRun } from "./continuation";

/** `startRun` is typed void but returns a promise, so the shape is checked. Non-thenable never
 *  releases early; a rejection settles it too. */
export function issuedRunFrom(
  started: unknown,
): AutoContinueIssuedRun | undefined {
  if (
    started === null ||
    (typeof started !== "object" && typeof started !== "function") ||
    typeof (started as { then?: unknown }).then !== "function"
  ) {
    return undefined;
  }
  const run = started as PromiseLike<unknown>;
  return {
    whenSettled: (onSettled) => {
      run.then(onSettled, onSettled);
    },
  };
}
