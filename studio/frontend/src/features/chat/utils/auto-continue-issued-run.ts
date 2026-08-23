// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Counts the runs assistant-ui has outstanding on one thread, preflight included.
 *
 * Its own file, and one importing nothing but a type: `auto-continue-run-keeper.ts` reaches the
 * chat store and through it most of the app, which a node test cannot load, and the id handling
 * here is the part that has to be proved rather than asserted about in prose.
 */

import type { AutoContinueIssuedRun } from "./continuation";

/**
 * The part of the assistant-ui runtime this needs, described structurally.
 *
 * By shape rather than by import so nothing takes a type dependency on the runtime package for
 * one method, and so a test can hand it a stand-in that throws where a real one throws.
 */
export type IssuedRunRuntime = {
  threads: {
    getById(threadId: string): {
      unstable_on(
        event: "runStart" | "runEnd",
        callback: () => void,
      ): () => void;
    };
  };
};

/**
 * Whether assistant-ui still has a run outstanding on this thread.
 *
 * Counted off `runStart` and `runEnd`, which the runtime emits around its own `startRun` -- the
 * first before the roundtrip begins, the second from the `finally` that closes it. The pair
 * therefore spans the ENTIRE call: this chat's settings pairing, `waitForModelReady`, the
 * stream, and every way the run can end, cancellation included.
 *
 * Deliberately NOT read off the running flag that `getState` exposes, which looks like the
 * obvious answer and is a trap. That flag is derived from the status of the message at the HEAD
 * of the current branch, so clicking the branch picker back to the truncated partial while the
 * continuation is still loading a model reads as the run having ended, with the run outstanding
 * the whole time. A hold discarded there stops renewing under a run that goes on to stream, and
 * the lease lapses mid-continuation -- the exact failure the keeper's missing arming deadline
 * exists to avoid, reached without a clock. Run events do not move when the branch does.
 *
 * `pending` answers `undefined` until a `runStart` has actually been seen, and again once the
 * watch is given up. That is "cannot tell", not "no run": `getById` THROWS for any id the
 * thread list cannot resolve to a MOUNTED runtime at that instant -- an alias retired by
 * hydration, but equally an ordinary background thread the user has navigated away from -- and
 * a hold dropped on an unreadable answer is a lease lapsing under a live continuation. Only a
 * run this side watched start and then watched end is ever reported as over.
 */
export function issuedRunFor(
  runtime: IssuedRunRuntime | null | undefined,
  threadIds: readonly (string | null | undefined)[],
): AutoContinueIssuedRun | undefined {
  const ids = [...new Set(threadIds.filter((id): id is string => Boolean(id)))];
  if (!runtime || ids.length === 0) {
    return undefined;
  }
  // `null` until the first run event is seen, and back to `null` whenever the watch is given
  // up: both mean this side cannot answer, which is what keeps a hold rather than ending it.
  let outstanding: number | null = null;
  return {
    pending: () => (outstanding === null ? undefined : outstanding > 0),
    watch: (onChange) => {
      for (const id of ids) {
        let offStart: (() => void) | null = null;
        let offEnd: (() => void) | null = null;
        try {
          const thread = runtime.threads.getById(id);
          offStart = thread.unstable_on("runStart", () => {
            outstanding = (outstanding ?? 0) + 1;
            onChange();
          });
          offEnd = thread.unstable_on("runEnd", () => {
            if (outstanding === null) {
              // The end of a run this side never saw start, so it says nothing about ours.
              // Counting it would take the total negative or, clamped, fake an idle thread.
              return;
            }
            outstanding = Math.max(0, outstanding - 1);
            onChange();
          });
        } catch {
          // The thread list cannot resolve this id to a live runtime. Undo whichever half of
          // the pair was registered before it threw, and try the next spelling.
          offStart?.();
          offEnd?.();
          continue;
        }
        return () => {
          offStart?.();
          offEnd?.();
          outstanding = null;
        };
      }
      return null;
    },
  };
}
