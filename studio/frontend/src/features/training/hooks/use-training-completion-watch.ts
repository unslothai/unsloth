// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect } from "react";

import { getTrainingStatus } from "../api/train-api";
import { createSingleFlightRequest } from "../lib/single-flight-request";
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
 * The lifecycle poll (`useTrainingRuntimeLifecycle`) only runs while Train is
 * mounted, so a run finishing while the user is on another tab would leave the
 * sidebar spinner stuck. This polls `/api/train/status` only while a run is in
 * progress (no traffic when idle). Mount once in an always-rendered shell.
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
    };
  }, [active]);
}
