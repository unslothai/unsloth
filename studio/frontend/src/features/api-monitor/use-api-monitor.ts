


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

/** Poll cadence while live. Matches the settings console it replaces. */
const POLL_INTERVAL_MS = 1500;

export type MonitorStatusFilter =
  | "all"
  | "running"
  | "completed"
  | "error"
  | "cancelled";

export interface MonitorStats {
  active: number;
  total: number;
  completed: number;
  errors: number;
  cancelled: number;
  /** Mean duration over finished requests, or null when none have finished. */
  avgDurationMs: number | null;
  /** Slowest finished request, for spotting a pathological call. */
  maxDurationMs: number | null;
  totalTokens: number;
  /** Share of finished requests that failed, 0-1. Null when nothing finished. */
  errorRate: number | null;
  /** Mean completion tokens per second over requests that reported both. */
  tokensPerSecond: number | null;
}

function isTerminal(entry: ApiMonitorEntry): boolean {
  return entry.status !== "running";
}

function completionTokens(entry: ApiMonitorEntry): number | null {
  if (entry.completion_tokens != null) {
    return entry.completion_tokens;
  }
  // Some providers report only a total, so subtract the prompt to estimate generated.
  if (entry.total_tokens != null && entry.prompt_tokens != null) {
    return Math.max(0, entry.total_tokens - entry.prompt_tokens);
  }
  return null;
}

function entryTokens(entry: ApiMonitorEntry): number {
  if (entry.total_tokens != null) {
    return entry.total_tokens;
  }
  return (entry.prompt_tokens ?? 0) + (entry.completion_tokens ?? 0);
}

export function computeStats(entries: ApiMonitorEntry[]): MonitorStats {
  let active = 0;
  let completed = 0;
  let errors = 0;
  let cancelled = 0;
  let totalTokens = 0;
  let durationSum = 0;
  let durationCount = 0;
  let maxDurationMs: number | null = null;
  // Total tokens over total time: averaging rates lets one tiny request outweigh a long one.
  let generatedTokens = 0;
  let generatedDurationMs = 0;

  let requests = 0;

  for (const entry of entries) {
    // A load, unload or download is not an HTTP call: it reads as "running" throughout, so
    // counting it invents an in-flight request. The backend leaves these out of active_count.
    if (entry.kind === "lifecycle") {
      continue;
    }
    requests += 1;
    totalTokens += entryTokens(entry);
    if (entry.status === "running") {
      active += 1;
    } else if (entry.status === "error") {
      errors += 1;
    } else if (entry.status === "cancelled") {
      cancelled += 1;
    } else {
      completed += 1;
    }
    const duration = entry.duration_ms;
    if (duration != null && isTerminal(entry)) {
      durationSum += duration;
      durationCount += 1;
      maxDurationMs =
        maxDurationMs == null ? duration : Math.max(maxDurationMs, duration);
      const generated = completionTokens(entry);
      // A sub-millisecond duration divides into a meaningless rate.
      if (generated != null && generated > 0 && duration > 0) {
        generatedTokens += generated;
        generatedDurationMs += duration;
      }
    }
  }

  const finished = completed + errors + cancelled;
  return {
    active,
    total: requests,
    completed,
    errors,
    cancelled,
    avgDurationMs: durationCount > 0 ? durationSum / durationCount : null,
    maxDurationMs,
    totalTokens,
    errorRate: finished > 0 ? errors / finished : null,
    tokensPerSecond:
      generatedDurationMs > 0
        ? generatedTokens / (generatedDurationMs / 1000)
        : null,
  };
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
