// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";

import {
  type DiffusionTrainingStatus,
  getDiffusionTrainingStatusText,
} from "../api";

const POLL_INTERVAL_MS = 1500;
const SETTLED_STATUSES = new Set(["idle", "completed", "stopped", "error"]);

/** No live worker and an idle or finished status: nothing changes until the next start. */
export function isDiffusionTrainingSettled(
  status: DiffusionTrainingStatus | null,
): boolean {
  return (
    status !== null && !status.active && SETTLED_STATUSES.has(status.status)
  );
}

export function useDiffusionTrainingStatus(active: boolean): {
  status: DiffusionTrainingStatus | null;
  poll: () => Promise<void>;
} {
  const [status, setStatus] = useState<DiffusionTrainingStatus | null>(null);
  const lastText = useRef<string | null>(null);
  const settled = useRef(false);

  const poll = useCallback(async () => {
    try {
      const text = await getDiffusionTrainingStatusText();
      // A finished run's status is kept until the next start; skip unchanged replies to avoid re-renders.
      if (text === lastText.current) return;
      const next = JSON.parse(text) as DiffusionTrainingStatus;
      lastText.current = text;
      settled.current = isDiffusionTrainingSettled(next);
      setStatus(next);
    } catch {
      /* best-effort; a failed poll should not surface an error while the tab is open */
    }
  }, []);

  useEffect(() => {
    if (!active) return;
    void poll();
    let skippedHidden = false;
    const id = window.setInterval(() => {
      // A live run keeps polling while hidden so onTrainingComplete fires on time.
      if (document.hidden && settled.current) {
        skippedHidden = true;
        return;
      }
      void poll();
    }, POLL_INTERVAL_MS);
    const onVisibilityChange = () => {
      if (!document.hidden && skippedHidden) {
        skippedHidden = false;
        void poll();
      }
    };
    document.addEventListener("visibilitychange", onVisibilityChange);
    return () => {
      window.clearInterval(id);
      document.removeEventListener("visibilitychange", onVisibilityChange);
    };
  }, [active, poll]);

  return { status, poll };
}
