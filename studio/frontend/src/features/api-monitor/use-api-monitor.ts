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

/** Poll cadence while live. Matches the settings console it replaces. */
const POLL_INTERVAL_MS = 1500;

export { computeStats };
export type { MonitorStats };

export type MonitorStatusFilter =
  | "all"
  | "running"
  | "completed"
  | "error"
  | "cancelled";

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
    // The fields a debugging session keys off. Coerced, not trusted: these arrive over
    // the network, and one malformed entry throwing here would blank the whole log.
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
  /** True until the first response lands, so skeletons show once. */
  loading: boolean;
  refreshing: boolean;
  paused: boolean;
  setPaused: (paused: boolean) => void;
  refresh: () => void;
  clear: () => Promise<void>;
  /** Full prompt/reply for expanded entries, keyed by entry id. */
  details: Record<string, ApiMonitorEntry>;
  loadingDetails: ReadonlySet<string>;
  requestDetail: (id: string) => boolean;
}

/**
 * Live view of the server's OpenAI-compatible API traffic. Polls rather than streams because the
 * backing monitor is a ring buffer with no change feed. Polling self-reschedules (never
 * overlapping), and pausing stops it so reading a payload is not fighting a reordering list.
 * `intervalMs` trades freshness for cost: the closed overlay slows right down.
 */
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

  const load = useCallback(async (): Promise<void> => {
    setRefreshing(true);
    try {
      const next = await getApiMonitor();
      setData(next);
      setError(null);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Monitor unavailable");
    } finally {
      setRefreshing(false);
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (paused) {
      return;
    }
    let cancelled = false;
    let timer: number | undefined;

    function poll(): void {
      getApiMonitor()
        .then((next) => {
          if (cancelled) return;
          setData(next);
          setError(null);
        })
        .catch((err: unknown) => {
          if (cancelled) return;
          setError(err instanceof Error ? err.message : "Monitor unavailable");
        })
        .finally(() => {
          if (cancelled) return;
          setLoading(false);
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
  }, [paused, intervalMs]);

  // Returns whether a fetch started: recording "fetched revision N" when the guard
  // refused would skip that revision once updated_at settles.
  const requestDetail = useCallback((id: string): boolean => {
    if (inFlightDetails.current.has(id)) {
      return false;
    }
    inFlightDetails.current.add(id);
    setLoadingDetails((prev) => new Set(prev).add(id));
    getApiMonitorEntry(id)
      .then((entry) => {
        setDetails((prev) => ({ ...prev, [id]: entry }));
      })
      .catch(() => {
        // Aged out of the ring buffer: drop the stale copy so the row previews show.
        setDetails((prev) => {
          if (!(id in prev)) return prev;
          const next = { ...prev };
          delete next[id];
          return next;
        });
      })
      .finally(() => {
        inFlightDetails.current.delete(id);
        setLoadingDetails((prev) => {
          const next = new Set(prev);
          next.delete(id);
          return next;
        });
      });
    return true;
  }, []);

  // The Clear log button discards this promise, so a failed DELETE has to land in the
  // error banner here: rethrowing leaves an unhandled rejection and a log that silently
  // did not clear. Sequence lives in a plain module so the node --test suite can drive it.
  const clear = useCallback(
    (): Promise<void> =>
      clearMonitor({
        clearRemote: clearApiMonitor,
        resetDetails: () => setDetails({}),
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
