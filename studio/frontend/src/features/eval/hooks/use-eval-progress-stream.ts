// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect } from "react";
import { streamEvalProgress, isAbortError } from "../api/eval-api";
import { useEvalRuntimeStore } from "../stores/eval-runtime-store";

/** While `runId` is set and `enabled`, stream progress into the store. */
export function useEvalProgressStream(runId: string | null, enabled: boolean) {
  const applyProgress = useEvalRuntimeStore((s) => s.applyProgress);
  const finishRun = useEvalRuntimeStore((s) => s.finishRun);

  useEffect(() => {
    if (!runId || !enabled) return;
    const controller = new AbortController();
    let cancelled = false;

    void (async () => {
      try {
        await streamEvalProgress({
          runId,
          signal: controller.signal,
          onEvent: ({ event, payload }) => {
            applyProgress(payload);
            if (event === "complete") finishRun(payload.status);
          },
        });
      } catch (error) {
        if (!cancelled && !isAbortError(error)) {
          // Stream dropped (e.g. server restart). Mark not-running; the
          // detail view re-fetches authoritative state from the DB.
          finishRun("interrupted");
        }
      }
    })();

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [runId, enabled, applyProgress, finishRun]);
}
