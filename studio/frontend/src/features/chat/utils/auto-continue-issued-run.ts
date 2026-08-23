// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Turns what `startRun` hands back into the keeper's "this run has ended" signal.
 *
 * Its own file, and one importing nothing but a type: `auto-continue-run-keeper.ts` reaches the
 * chat store and through it most of the app, which a node test cannot load, while the guard
 * below is the part that has to be proved rather than asserted about in prose.
 */

import type { AutoContinueIssuedRun } from "./continuation";

/**
 * The run just issued, watched through its own promise.
 *
 * `ThreadRuntime.startRun` is DECLARED to return `void` and in fact returns the roundtrip's
 * promise: every hop from the store proxy down to `LocalThreadRuntimeCore.startRun` delegates
 * with `return`, and that core is `async`. The declaration is what is narrow, not the value,
 * so the caller passes it through untyped and the shape is checked here rather than assumed.
 *
 * A value that is not thenable is therefore not something to work around -- it is assistant-ui
 * no longer handing the run back, and the honest answer is `undefined`: no signal, so the hold
 * is kept and renewed exactly as every hold that has not armed is. That is the behaviour this
 * fix replaced, never an early release, and the pinning test alongside is what makes the change
 * visible rather than silent.
 *
 * Both outcomes settle the hold. The promise rejects when the roundtrip throws and resolves
 * when the run is cancelled, and neither says anything about whether the lease may be given
 * back -- only whether the run is still coming. The failure itself is already reported per
 * thread by the adapter wrapper, which is where it belongs; observing it here also stops the
 * unhandled rejection this call site produced before, and adds no new reporting of its own.
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
