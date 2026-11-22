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

/** The lifecycle poll only runs on the Train page; this keeps the sidebar fresh elsewhere.
 * Mount once in an always-rendered shell. */
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
      // Post-download disk check: the worker downloads after start, so look again when the run stops,
      // finished or failed. Forced so the reading is taken after the writes.
      void checkDiskSpace({ force: true });
    };
  }, [active]);
}
