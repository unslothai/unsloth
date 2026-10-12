// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "@/lib/toast";
import { listTrainingRuns } from "../api/history-api";
import {
  onTrainingRunDeleted,
  onTrainingRunsChanged,
  onTrainingRunUpdated,
} from "../events";
import type { TrainingRunSummary } from "../types/history";

const SIDEBAR_LIMIT = 20;
/** The route's own cap on one request. */
const MAX_REQUEST_LIMIT = 200;
const RUNNING_POLL_MS = 5000;
const INITIAL_RETRY_DELAYS_MS = [500, 1500, 3500];
const LOAD_FAILURE_TOAST_ID = "training-history-load-failure";

function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === "AbortError";
}

export function useTrainingHistorySidebarItems(enabled: boolean) {
  const [items, setItems] = useState<TrainingRunSummary[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [total, setTotal] = useState(0);
  const controllerRef = useRef<AbortController | null>(null);
  // How many runs the sidebar holds, so a refresh or a poll re-reads every page scrolled into.
  const loadedCountRef = useRef(0);
  useEffect(() => {
    loadedCountRef.current = items.length;
  }, [items.length]);

  const fetchRuns = useCallback(async (): Promise<void> => {
    if (controllerRef.current && !controllerRef.current.signal.aborted) {
      return;
    }
    const controller = new AbortController();
    controllerRef.current = controller;
    try {
      const wanted = Math.max(SIDEBAR_LIMIT, loadedCountRef.current);
      const runs: TrainingRunSummary[] = [];
      let runTotal = 0;
      while (runs.length < wanted) {
        const result = await listTrainingRuns(
          Math.min(MAX_REQUEST_LIMIT, wanted - runs.length),
          runs.length,
          controller.signal,
        );
        runs.push(...result.runs);
        runTotal = result.total;
        if (result.runs.length === 0 || runs.length >= result.total) break;
      }
      if (controller.signal.aborted) {
        return;
      }
      setItems(runs);
      setTotal(runTotal);
      setLoaded(true);
      toast.dismiss(LOAD_FAILURE_TOAST_ID);
    } finally {
      if (controllerRef.current === controller) {
        controllerRef.current = null;
      }
    }
  }, []);

  const hasMore = loaded && items.length < total;
  // The next page, when the list is scrolled to its end. Shares the in-flight guard, so a poll and
  // a page never race to write the list.
  const loadMore = useCallback(async (): Promise<void> => {
    if (controllerRef.current && !controllerRef.current.signal.aborted) {
      return;
    }
    const controller = new AbortController();
    controllerRef.current = controller;
    try {
      const result = await listTrainingRuns(
        SIDEBAR_LIMIT,
        loadedCountRef.current,
        controller.signal,
      );
      if (controller.signal.aborted) {
        return;
      }
      setItems((prev) => {
        const seen = new Set(prev.map((run) => run.id));
        return [...prev, ...result.runs.filter((run) => !seen.has(run.id))];
      });
      setTotal(result.total);
    } catch {
      // the next scroll retries
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

  // Initial load: bounded retry-with-backoff, then a toast with Retry on final
  // failure so the user isn't stuck on an empty sidebar if the backend is slow.
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

  // Poll while a run is active, but only when the tab is visible. Browsers
  // throttle background timers but don't pause them, so gating on visibility
  // avoids hammering the API for tabs left open during long training runs.
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
      setTotal((count) => Math.max(0, count - 1));
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

  return { items, loaded, refresh, hasMore, loadMore };
}
