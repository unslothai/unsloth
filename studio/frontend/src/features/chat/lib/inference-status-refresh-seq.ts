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
const inferenceStatusSupersession: RefreshSupersession = { latest: null };

export function beginInferenceStatusRefresh(): {
  seq: number;
  isCurrent: () => boolean;
  register: (settled: Promise<void>) => Promise<void>;
  superseded: () => Promise<void> | undefined;
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
  };
}

/** Test hook: reset module state between cases. */
export function resetInferenceStatusRefreshSeqForTests(): void {
  inferenceStatusSeq = 0;
  inferenceStatusSupersession.latest = null;
}
