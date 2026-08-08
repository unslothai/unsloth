// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Sequence guard for overlapping /api/inference/status reads on the chat page. Mount
// refresh and the deferred inventory refresh can run concurrently; without this, an older
// response can roll load-param baselines backward after a newer one has already advanced
// them (the Hub uses the same pattern in hub-page.tsx).

import {
  type RefreshSupersession,
  registerRefresh,
  supersedingRefresh,
} from "@/features/hub/lib/superseded-refresh";

let inferenceStatusSeq = 0;
let lastSuccessfulStatusSeq = 0;
const inferenceStatusSupersession: RefreshSupersession = { latest: null };

export function beginInferenceStatusRefresh(): {
  seq: number;
  isCurrent: () => boolean;
  register: (settled: Promise<void>) => Promise<void>;
  superseded: () => Promise<void> | undefined;
  /** Call after this refresh commits status/checkpoint hydration. */
  markApplied: () => void;
  /** True once a newer refresh has successfully applied. */
  shouldSkipAfterSupersession: () => boolean;
} {
  const seq = ++inferenceStatusSeq;
  return {
    seq,
    isCurrent: () => seq === inferenceStatusSeq,
    register: (settled) => {
      registerRefresh(inferenceStatusSupersession, seq, settled);
      return settled;
    },
    superseded: () => supersedingRefresh(inferenceStatusSupersession, seq),
    markApplied: () => {
      lastSuccessfulStatusSeq = seq;
    },
    shouldSkipAfterSupersession: () => lastSuccessfulStatusSeq > seq,
  };
}

export type InferenceStatusRefresh = ReturnType<typeof beginInferenceStatusRefresh>;

/**
 * Wait out any newer overlapping reads before committing this snapshot. Loops
 * through every intervening refresh, not just the first superseder, so a failed
 * refresh 2 cannot let refresh 1 commit while refresh 3 is still pending.
 */
export async function awaitInferenceStatusRefreshTurn(
  refresh: InferenceStatusRefresh,
  options?: { aborted?: () => boolean },
): Promise<boolean> {
  let lastAwaitedSeq = refresh.seq;
  while (!refresh.isCurrent()) {
    if (refresh.shouldSkipAfterSupersession()) return false;
    if (options?.aborted?.()) return false;
    const latest = inferenceStatusSupersession.latest;
    if (!latest || latest.seq <= lastAwaitedSeq) break;
    await latest.settled;
    lastAwaitedSeq = latest.seq;
    if (options?.aborted?.()) return false;
  }
  if (refresh.shouldSkipAfterSupersession()) return false;
  if (options?.aborted?.()) return false;
  return true;
}

/** Test hook: reset module state between cases. */
export function resetInferenceStatusRefreshSeqForTests(): void {
  inferenceStatusSeq = 0;
  lastSuccessfulStatusSeq = 0;
  inferenceStatusSupersession.latest = null;
}
