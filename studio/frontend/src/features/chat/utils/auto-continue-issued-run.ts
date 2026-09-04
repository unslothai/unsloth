// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Its own file, importing nothing but a type: `auto-continue-run-keeper.ts` reaches the chat
// store and through it most of the app, which a node test cannot load.

import type { AutoContinueIssuedRun } from "./continuation";

/**
 * The run just issued, watched through its own promise.
 *
 * `ThreadRuntime.startRun` is DECLARED `void` and returns the roundtrip's promise: every hop
 * down to `LocalThreadRuntimeCore.startRun` delegates with `return` and that core is `async`.
 * The declaration is narrow, not the value, so the caller passes it untyped and the shape is
 * checked here. Not thenable means assistant-ui no longer hands the run back, and the answer
 * is `undefined`: no signal, so the hold is renewed as any unarmed hold is. Never an early
 * release.
 *
 * Rejection and cancellation both settle it: they say the run is no longer coming, not whether
 * the lease may be given back. Observing the rejection also stops the unhandled one this call
 * site produced; the failure is still reported per thread by the adapter wrapper.
 */
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
