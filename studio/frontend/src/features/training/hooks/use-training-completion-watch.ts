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
import { isTrainingStatusPolledElsewhere } from "../lib/training-status-poll-owner";
import {
  isTrainingStartPending,
  useTrainingRuntimeStore,
} from "../stores/training-runtime-store";

const WATCH_INTERVAL_MS = 6000;

/** Keeps the sidebar fresh off the Train page; mount once in an always-rendered shell. */
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

    const id = window.setInterval(() => {
      // On the Train page the lifecycle poll already fetches this status every 3 s.
      if (isTrainingStatusPolledElsewhere()) return;
      void tick();
    }, WATCH_INTERVAL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(id);
      // Recheck disk on stop: the worker downloads after start; forced to read after writes.
      void checkDiskSpace({ force: true });
    };
  }, [active]);
}
