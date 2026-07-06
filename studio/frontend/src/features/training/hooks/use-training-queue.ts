// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef } from "react";
import { toast } from "@/lib/toast";
import {
  getQueueState,
  moveQueueItem,
  pauseQueue,
  removeQueueItem,
  resumeQueue,
} from "../api/queue-api";
import { emitTrainingRunsChanged } from "../events";
import { syncTrainingRuntimeFromBackend } from "../lib/sync-runtime";
import { useTrainingQueueStore } from "../stores/training-queue-store";
import { useTrainingRuntimeStore } from "../stores/training-runtime-store";
import type { TrainingQueueState } from "../types/queue";

const ACTIVE_POLL_MS = 5000;
const IDLE_POLL_MS = 30000;

// Mount once (StudioPage); read useTrainingQueueStore anywhere.
export function useTrainingQueue() {
  const controllerRef = useRef<AbortController | null>(null);

  const refresh = useCallback(async (): Promise<void> => {
    if (controllerRef.current && !controllerRef.current.signal.aborted) {
      return;
    }
    const controller = new AbortController();
    controllerRef.current = controller;
    try {
      const state = await getQueueState(controller.signal);
      if (controller.signal.aborted) {
        return;
      }
      applyQueueState(state);
    } catch {
      // transient; next poll heals
    } finally {
      if (controllerRef.current === controller) {
        controllerRef.current = null;
      }
    }
  }, []);

  useEffect(() => {
    void refresh();

    let timer: ReturnType<typeof setInterval> | null = null;
    let currentInterval = 0;

    const desiredInterval = () => {
      const queue = useTrainingQueueStore.getState();
      const runtime = useTrainingRuntimeStore.getState();
      const busy =
        queue.pendingCount > 0 ||
        queue.activeJobId !== null ||
        runtime.isTrainingRunning;
      return busy ? ACTIVE_POLL_MS : IDLE_POLL_MS;
    };

    const stop = () => {
      if (timer !== null) {
        clearInterval(timer);
        timer = null;
      }
    };

    const start = () => {
      const interval = desiredInterval();
      if (timer !== null && interval === currentInterval) {
        return;
      }
      stop();
      currentInterval = interval;
      timer = setInterval(() => {
        void refresh();
        start();
      }, interval);
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
  }, [refresh]);

  return { refresh };
}

// Dedupe on our own previous snapshot, NOT the runtime store's jobId — the
// faster status poll usually adopts the new job first, which would swallow
// the sidebar nudge here.
function applyQueueState(state: TrainingQueueState): void {
  const previousActiveJobId = useTrainingQueueStore.getState().activeJobId;
  useTrainingQueueStore.getState().applyState(state);

  if (state.active_job_id && state.active_job_id !== previousActiveJobId) {
    emitTrainingRunsChanged();
    if (state.active_job_id !== useTrainingRuntimeStore.getState().jobId) {
      void syncTrainingRuntimeFromBackend().catch(() => undefined);
    }
  }
}

export function useTrainingQueueActions() {
  const withRefresh = useCallback(
    async (action: () => Promise<unknown>, failureTitle: string): Promise<boolean> => {
      try {
        const result = await action();
        if (result && typeof result === "object" && "items" in (result as object)) {
          applyQueueState(result as TrainingQueueState);
        } else {
          applyQueueState(await getQueueState());
        }
        return true;
      } catch (err) {
        toast.error(failureTitle, {
          description: err instanceof Error ? err.message : undefined,
        });
        try {
          applyQueueState(await getQueueState());
        } catch {
          // next poll heals
        }
        return false;
      }
    },
    [],
  );

  const removeItem = useCallback(
    (itemId: string) =>
      withRefresh(() => removeQueueItem(itemId), "Couldn't remove queued job"),
    [withRefresh],
  );

  const moveItem = useCallback(
    (itemId: string, direction: "up" | "down") =>
      withRefresh(() => moveQueueItem(itemId, direction), "Couldn't reorder queue"),
    [withRefresh],
  );

  const pause = useCallback(
    () => withRefresh(() => pauseQueue(), "Couldn't pause queue"),
    [withRefresh],
  );

  const resume = useCallback(
    () => withRefresh(() => resumeQueue(), "Couldn't resume queue"),
    [withRefresh],
  );

  return { removeItem, moveItem, pause, resume };
}
