// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  clearApiMonitor,
  getApiMonitor,
  getApiMonitorEntry,
} from "@/features/chat/api/chat-api";
import type {
  ApiMonitorEntry,
  ApiMonitorResponse,
} from "@/features/chat/types/api";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { clearMonitor } from "./clear-monitor";
import { type MonitorStats, computeStats } from "./stats";

const POLL_INTERVAL_MS = 1500;
const MAX_CACHED_PROMPT_CHARS = 64 * 1024 * 1024;

export { computeStats };
export type { MonitorStats };

export type MonitorStatusFilter =
  | "all"
  | "running"
  | "completed"
  | "error"
  | "cancelled";

function retainDetail(
  previous: Record<string, ApiMonitorEntry>,
  id: string,
  entry: ApiMonitorEntry,
  cachedPrompt: string | undefined,
): Record<string, ApiMonitorEntry> {
  const refreshed =
    cachedPrompt == null ? entry : { ...entry, prompt: cachedPrompt };
  const retained: Record<string, ApiMonitorEntry> = { [id]: refreshed };
  let promptChars = refreshed.prompt?.length ?? 0;
  for (const [cachedId, cached] of Object.entries(previous)) {
    if (cachedId === id) {
      continue;
    }
    const cachedChars = cached.prompt?.length ?? 0;
    if (promptChars + cachedChars > MAX_CACHED_PROMPT_CHARS) {
      continue;
    }
    retained[cachedId] = cached;
    promptChars += cachedChars;
  }
  return retained;
}

function dropDetail(
  previous: Record<string, ApiMonitorEntry>,
  id: string,
): Record<string, ApiMonitorEntry> {
  if (!(id in previous)) {
    return previous;
  }
  const next = { ...previous };
  delete next[id];
  return next;
}

export function filterEntries(
  entries: ApiMonitorEntry[],
  status: MonitorStatusFilter,
  query: string,
): ApiMonitorEntry[] {
  const needle = query.trim().toLowerCase();
  return entries.filter((entry) => {
    if (status !== "all" && entry.status !== status) {
      return false;
    }
    if (!needle) {
      return true;
    }
    // Coerced: one malformed network entry throwing here would blank the whole log.
    return [
      entry.model,
      entry.endpoint,
      entry.prompt_preview,
      entry.reply_preview,
      entry.error,
    ].some((field) =>
      String(field ?? "")
        .toLowerCase()
        .includes(needle),
    );
  });
}

interface UseApiMonitorResult {
  data: ApiMonitorResponse | null;
  entries: ApiMonitorEntry[];
  stats: MonitorStats;
  error: string | null;
  loading: boolean;
  refreshing: boolean;
  paused: boolean;
  setPaused: (paused: boolean) => void;
  refresh: () => void;
  clear: () => Promise<void>;
  details: Record<string, ApiMonitorEntry>;
  loadingDetails: ReadonlySet<string>;
  requestDetail: (id: string) => boolean;
}

/** Polls (the ring buffer has no change feed), never overlapping; pausing stops it. */
export function useApiMonitor({
  intervalMs = POLL_INTERVAL_MS,
}: { intervalMs?: number } = {}): UseApiMonitorResult {
  const [data, setData] = useState<ApiMonitorResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [paused, setPaused] = useState(false);
  const [details, setDetails] = useState<Record<string, ApiMonitorEntry>>({});
  const [loadingDetails, setLoadingDetails] = useState<Set<string>>(
    () => new Set(),
  );
  // Mirrors `loadingDetails` outside React state so the guard sees same-tick writes.
  const inFlightDetails = useRef<Set<string>>(new Set());
  const retainedEntryIds = useRef<Set<string>>(new Set());
  const listRequestGeneration = useRef(0);
  const manualLoadGeneration = useRef(0);
  const detailRequestGeneration = useRef(0);

  const updateData = useCallback((next: ApiMonitorResponse): void => {
    const ids = new Set(next.entries.map((entry) => entry.id));
    retainedEntryIds.current = ids;
    setData(next);
    setDetails((previous) => {
      const expired = Object.keys(previous).filter((id) => !ids.has(id));
      if (expired.length === 0) return previous;
      const retained = { ...previous };
      for (const id of expired) delete retained[id];
      return retained;
    });
  }, []);

  const load = useCallback(async (): Promise<void> => {
    const requestGeneration = ++listRequestGeneration.current;
    const manualGeneration = ++manualLoadGeneration.current;
    setRefreshing(true);
    try {
      const next = await getApiMonitor();
      if (listRequestGeneration.current !== requestGeneration) {
        return;
      }
      updateData(next);
      setError(null);
    } catch (err: unknown) {
      if (listRequestGeneration.current !== requestGeneration) {
        return;
      }
      setError(err instanceof Error ? err.message : "Monitor unavailable");
    } finally {
      if (manualLoadGeneration.current === manualGeneration) {
        setRefreshing(false);
      }
      if (listRequestGeneration.current === requestGeneration) {
        setLoading(false);
      }
    }
  }, [updateData]);

  useEffect(() => {
    if (paused) {
      return;
    }
    let cancelled = false;
    let timer: number | undefined;

    function poll(): void {
      const requestGeneration = ++listRequestGeneration.current;
      getApiMonitor()
        .then((next) => {
          if (
            cancelled ||
            listRequestGeneration.current !== requestGeneration
          ) {
            return;
          }
          updateData(next);
          setError(null);
        })
        .catch((err: unknown) => {
          if (
            cancelled ||
            listRequestGeneration.current !== requestGeneration
          ) {
            return;
          }
          setError(err instanceof Error ? err.message : "Monitor unavailable");
        })
        .finally(() => {
          if (cancelled) {
            return;
          }
          if (listRequestGeneration.current === requestGeneration) {
            setLoading(false);
          }
          timer = window.setTimeout(poll, intervalMs);
        });
    }

    poll();
    return () => {
      cancelled = true;
      if (timer !== undefined) {
        window.clearTimeout(timer);
      }
    };
  }, [paused, intervalMs, updateData]);

  // Returns whether a fetch started; recording a refused revision would skip it for good.
  const requestDetail = useCallback(
    (id: string): boolean => {
      if (inFlightDetails.current.has(id)) {
        return false;
      }
      inFlightDetails.current.add(id);
      const requestGeneration = detailRequestGeneration.current;
      setLoadingDetails((prev) => new Set(prev).add(id));
      const cachedPrompt = details[id]?.prompt;
      getApiMonitorEntry(id, cachedPrompt == null)
        .then((entry) => {
          if (
            detailRequestGeneration.current !== requestGeneration ||
            !retainedEntryIds.current.has(id)
          ) {
            return;
          }
          setDetails((prev) => retainDetail(prev, id, entry, cachedPrompt));
        })
        .catch(() => {
          if (detailRequestGeneration.current !== requestGeneration) {
            return;
          }
          // Aged out of the ring buffer: drop the stale copy so the row previews show.
          setDetails((prev) => dropDetail(prev, id));
        })
        .finally(() => {
          if (detailRequestGeneration.current !== requestGeneration) {
            return;
          }
          inFlightDetails.current.delete(id);
          setLoadingDetails((prev) => {
            const next = new Set(prev);
            next.delete(id);
            return next;
          });
        });
      return true;
    },
    [details],
  );

  // The button discards this promise, so a failed DELETE must land in the error banner here.
  const clear = useCallback(
    (): Promise<void> =>
      clearMonitor({
        clearRemote: clearApiMonitor,
        resetDetails: () => {
          listRequestGeneration.current += 1;
          detailRequestGeneration.current += 1;
          retainedEntryIds.current = new Set();
          inFlightDetails.current.clear();
          setLoadingDetails(new Set());
          setDetails({});
        },
        reload: load,
        onError: setError,
      }),
    [load],
  );

  const entries = useMemo(() => data?.entries ?? [], [data]);
  const stats = useMemo(() => computeStats(entries), [entries]);

  return {
    data,
    entries,
    stats,
    error,
    loading,
    refreshing,
    paused,
    setPaused,
    refresh: () => void load(),
    clear,
    details,
    loadingDetails,
    requestDetail,
  };
}
