// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect } from "react";

import { getTrainingStatus } from "../api/train-api";
// eslint-disable-next-line no-restricted-imports
import { checkDiskSpace } from "@/features/settings/low-disk-check";
import { createSingleFlightRequest } from "@/lib/single-flight-request";
import {
  isTrainingStatusRequestCurrent,
  trainingStatusRequestKey,
} from "../lib/training-status-request";
import {
  isTrainingStartPending,
  useTrainingRuntimeStore,
} from "../stores/training-runtime-store";

const WATCH_INTERVAL_MS = 6000;

/**
 * Keep training state fresh while a run is active, even off the Train page.
 *
 * The lifecycle poll (`useTrainingRuntimeLifecycle`) only runs while Train is mounted, so a run
 * finishing on another tab would leave the sidebar spinner stuck. Polls `/api/train/status` only
 * while a run is in progress. Mount once in an always-rendered shell.
 */
export function useTrainingCompletionWatch(): void {
  const active = useTrainingRuntimeStore(isTrainingStartPending);

  useEffect(() => {
    if (!active) {
      return;
    }
    let cancelled = false;

    const tick = createSingleFlightRequest(async () => {
      const initial = useTrainingRuntimeStore.getState();
      const requestKey = trainingStatusRequestKey(initial);
      try {
        const status = await getTrainingStatus(requestKey);
        const runtime = useTrainingRuntimeStore.getState();
        if (!cancelled && isTrainingStatusRequestCurrent(requestKey, runtime)) {
          runtime.applyStatus(status);
        }
      } catch {
        // Transient network/auth hiccup; the next tick retries.
      }
    });

    const id = window.setInterval(tick, WATCH_INTERVAL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(id);
      // The post-download half of the training disk check. startTraining takes the
      // reading BEFORE the run; the worker then downloads the base model and any
      // remote dataset on its own time, so the only honest moment to look again is
      // when the run stops being active. This cleanup is that moment, and it does
      // not care whether the run finished or failed: a run that died because the
      // disk filled is precisely the one worth reporting.
      //
      // forced, for the reason the download manager's finalize is forced: the
      // reading has to be taken AFTER the write, and unforced it would be swallowed
      // by the interval or handed a figure from before the run.
      void checkDiskSpace({ force: true });
    };
  }, [active]);
}
