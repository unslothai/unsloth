// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { listTrainingRuns } from "../api/history-api";
import {
  onTrainingRunDeleted,
  onTrainingRunsChanged,
  onTrainingRunUpdated,
} from "../events";
import type { TrainingRunSummary } from "../types/history";

const SIDEBAR_LIMIT = 20;
const RUNNING_POLL_MS = 5000;
const INITIAL_RETRY_DELAYS_MS = [500, 1500, 3500];
const LOAD_FAILURE_TOAST_ID = "training-history-load-failure";

function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === "AbortError";
}

export function useTrainingHistorySidebarItems(enabled: boolean) {
  const [items, setItems] = useState<TrainingRunSummary[]>([]);
  const [loaded, setLoaded] = useState(false);
  const controllerRef = useRef<AbortController | null>(null);

  const fetchRuns = useCallback(async (): Promise<void> => {
    if (controllerRef.current && !controllerRef.current.signal.aborted) {
      return;
    }
    const controller = new AbortController();
    controllerRef.current = controller;
    try {
      const result = await listTrainingRuns(
        SIDEBAR_LIMIT,
        0,
        controller.signal,
      );
      if (controller.signal.aborted) {
        return;
      }
      setItems(result.runs);
      setLoaded(true);
      toast.dismiss(LOAD_FAILURE_TOAST_ID);
    } finally {
      if (controllerRef.current === controller) {
        controllerRef.current = null;
      }
    }
  }, []);

  // Background refresh (rename/delete sync, polling): swallow errors so
  // transient failures don't spam toasts; the next successful fetch heals.
  const refresh = useCallback(async (): Promise<void> => {
    try {
      await fetchRuns();
    } catch {
      // intentionally ignored
    }
  }, [fetchRuns]);

  // Initial load: bounded retry-with-backoff, then surface a toast on
  // final failure with a Retry action so the user isn't stuck staring
  // at an empty sidebar after F5 if the backend was slow to come up.
  useEffect(() => {
    if (!enabled) {
      return;
    }
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const showFailureToast = (err: unknown): void => {
      toast.error("Couldn't load training runs", {
        id: LOAD_FAILURE_TOAST_ID,
        description: err instanceof Error ? err.message : undefined,
        action: {
          label: "Retry",
          onClick: () => {
            void (async () => {
              try {
                await fetchRuns();
              } catch (retryErr) {
                if (isAbortError(retryErr)) {
                  return;
                }
                showFailureToast(retryErr);
              }
            })();
          },
        },
      });
    };

    const attempt = async (index: number): Promise<void> => {
      if (cancelled) {
        return;
      }
      try {
        await fetchRuns();
      } catch (err) {
        if (cancelled || isAbortError(err)) {
          return;
        }
        if (index < INITIAL_RETRY_DELAYS_MS.length) {
          timer = setTimeout(
            () => void attempt(index + 1),
            INITIAL_RETRY_DELAYS_MS[index],
          );
          return;
        }
        showFailureToast(err);
      }
    };

    void attempt(0);

    return () => {
      cancelled = true;
      if (timer) {
        clearTimeout(timer);
      }
      controllerRef.current?.abort();
    };
  }, [enabled, fetchRuns]);

  // Poll while there's a running run, but only when the tab is visible.
  // Browsers throttle background timers but don't pause them — gating on
  // visibility avoids hammering the API for tabs left open in the
  // background, which is common during long training runs.
  const hasRunning = items.some((r) => r.status === "running");
  useEffect(() => {
    if (!enabled || !hasRunning) {
      return;
    }

    let timer: ReturnType<typeof setInterval> | null = null;
    const start = () => {
      if (timer !== null) {
        return;
      }
      timer = setInterval(() => void refresh(), RUNNING_POLL_MS);
    };
    const stop = () => {
      if (timer === null) {
        return;
      }
      clearInterval(timer);
      timer = null;
    };

    const onVisibilityChange = () => {
      if (document.visibilityState === "visible") {
        void refresh();
        start();
      } else {
        stop();
      }
    };

    if (document.visibilityState === "visible") {
      start();
    }
    document.addEventListener("visibilitychange", onVisibilityChange);

    return () => {
      document.removeEventListener("visibilitychange", onVisibilityChange);
      stop();
      controllerRef.current?.abort();
    };
  }, [enabled, hasRunning, refresh]);

  useEffect(() => {
    const offUpdated = onTrainingRunUpdated((updated) => {
      controllerRef.current?.abort();
      setItems((prev) =>
        prev.map((run) => (run.id === updated.id ? updated : run)),
      );
    });
    const offDeleted = onTrainingRunDeleted((runId) => {
      controllerRef.current?.abort();
      setItems((prev) => prev.filter((run) => run.id !== runId));
    });
    const offChanged = onTrainingRunsChanged(() => {
      controllerRef.current?.abort();
      void refresh();
    });
    return () => {
      offUpdated();
      offDeleted();
      offChanged();
    };
  }, [refresh]);

  return { items, loaded, refresh };
}
