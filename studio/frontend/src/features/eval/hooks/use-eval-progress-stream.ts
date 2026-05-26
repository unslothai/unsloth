// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect } from "react";
import { streamEvalProgress, isAbortError } from "../api/eval-api";
import { useEvalRuntimeStore } from "../stores/eval-runtime-store";

/** While `runId` is set and `enabled`, stream progress into the store. */
export function useEvalProgressStream(runId: string | null, enabled: boolean) {
  const applyProgress = useEvalRuntimeStore((s) => s.applyProgress);
  const finishRun = useEvalRuntimeStore((s) => s.finishRun);
  const appendLogs = useEvalRuntimeStore((s) => s.appendLogs);

  useEffect(() => {
    if (!runId || !enabled) return;
    const controller = new AbortController();
    let cancelled = false;
    let sawTerminal = false;

    void (async () => {
      try {
        await streamEvalProgress({
          runId,
          signal: controller.signal,
          onEvent: (evt) => {
            if (evt.event === "log") {
              appendLogs(evt.logs);
              return;
            }
            applyProgress(evt.payload);
            if (evt.event === "complete") {
              sawTerminal = true;
              finishRun(evt.payload.status);
            }
          },
        });
        // Stream ended cleanly without a terminal event (e.g. the manager
        // evicted the run before we read it). Don't leave it stuck running.
        if (!cancelled && !sawTerminal) finishRun("interrupted");
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
  }, [runId, enabled, applyProgress, finishRun, appendLogs]);
}
