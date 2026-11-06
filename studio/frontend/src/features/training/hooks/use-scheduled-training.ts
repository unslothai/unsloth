// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect } from "react";
import {
  isTrainingScheduleActive,
  useScheduledTrainingStore,
} from "../stores/scheduled-training-store";

// A waiting start only changes when it fires, is cancelled, or misses; none needs a fast poll.
const REFRESH_MS = 30_000;

/** Loads the account's scheduled start and keeps it current while one waits. */
export function useScheduledTrainingLifecycle(): void {
  const refresh = useScheduledTrainingStore((state) => state.refresh);
  const hasActive = useScheduledTrainingStore((state) =>
    state.schedules.some(isTrainingScheduleActive),
  );

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    if (!hasActive) return;
    const timer = window.setInterval(() => void refresh(), REFRESH_MS);
    // Another tab may have scheduled or cancelled while this one was in the background.
    const onFocus = () => void refresh();
    window.addEventListener("focus", onFocus);
    return () => {
      window.clearInterval(timer);
      window.removeEventListener("focus", onFocus);
    };
  }, [hasActive, refresh]);
}
