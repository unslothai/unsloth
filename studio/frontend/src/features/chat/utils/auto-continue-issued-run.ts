// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { AutoContinueIssuedRun } from "./continuation";

/** `startRun` is DECLARED `void` but returns the roundtrip's promise, so it arrives untyped and
 *  the shape is checked rather than assumed. Not thenable means no signal, so the hold is renewed
 *  as any unarmed hold is, never released early. A rejection settles it too: it says the run is
 *  not coming, not whether the lease may be given back. */
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
