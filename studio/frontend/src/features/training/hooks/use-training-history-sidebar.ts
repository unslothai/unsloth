// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";
import { listTrainingRuns } from "../api/history-api";
import type { TrainingRunSummary } from "../types/history";

const SIDEBAR_LIMIT = 20;
const RUNNING_POLL_MS = 5000;
const RUNS_CACHE_KEY = "unsloth_training_runs_sidebar_cache";

function isValidCachedRun(value: unknown): value is TrainingRunSummary {
  if (!value || typeof value !== "object") return false;
  const r = value as Record<string, unknown>;
  return (
    typeof r.id === "string" &&
    r.id.length > 0 &&
    typeof r.status === "string" &&
    typeof r.model_name === "string" &&
    typeof r.dataset_name === "string" &&
    typeof r.started_at === "string"
  );
}

function loadCachedRuns(): TrainingRunSummary[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(RUNS_CACHE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isValidCachedRun);
  } catch {
    return [];
  }
}

function saveCachedRuns(runs: TrainingRunSummary[]): void {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(RUNS_CACHE_KEY, JSON.stringify(runs));
  } catch {
    /* empty */
  }
}

export function useTrainingHistorySidebarItems(enabled: boolean) {
  const [items, setItems] = useState<TrainingRunSummary[]>(() =>
    loadCachedRuns(),
  );
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
      saveCachedRuns(result.runs);
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

  const applyRunUpdate = useCallback((updated: TrainingRunSummary) => {
    setItems((prev) => {
      const next = prev.map((run) =>
        run.id === updated.id ? updated : run,
      );
      saveCachedRuns(next);
      return next;
    });
  }, []);

  const removeRun = useCallback((runId: string) => {
    setItems((prev) => {
      const next = prev.filter((run) => run.id !== runId);
      saveCachedRuns(next);
      return next;
    });
  }, []);

  return { items, loaded, refresh: fetchRuns, applyRunUpdate, removeRun };
}
