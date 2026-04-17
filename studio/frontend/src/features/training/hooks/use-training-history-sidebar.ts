// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";
import { listTrainingRuns } from "../api/history-api";
import type { TrainingRunSummary } from "../types/history";

const SIDEBAR_LIMIT = 20;
const RUNNING_POLL_MS = 5000;

export function useTrainingHistorySidebarItems(enabled: boolean) {
  const [items, setItems] = useState<TrainingRunSummary[]>([]);
  const [loaded, setLoaded] = useState(false);
  const controllerRef = useRef<AbortController | null>(null);
  const inFlightRef = useRef(false);

  const fetchRuns = useCallback(async () => {
    if (inFlightRef.current) {
      return;
    }
    const controller = new AbortController();
    controllerRef.current = controller;
    inFlightRef.current = true;
    try {
      const result = await listTrainingRuns(SIDEBAR_LIMIT, 0, controller.signal);
      setItems(result.runs);
      setLoaded(true);
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") return;
    } finally {
      if (controllerRef.current === controller) {
        controllerRef.current = null;
      }
      inFlightRef.current = false;
    }
  }, []);

  useEffect(() => {
    if (!enabled) return;
    void fetchRuns();
    return () => {
      controllerRef.current?.abort();
    };
  }, [enabled, fetchRuns]);

  const hasRunning = items.some((r) => r.status === "running");
  useEffect(() => {
    if (!enabled || !hasRunning) return;
    const timer = setInterval(() => {
      void fetchRuns();
    }, RUNNING_POLL_MS);
    return () => clearInterval(timer);
  }, [enabled, hasRunning, fetchRuns]);

  return { items, loaded, refresh: fetchRuns };
}
