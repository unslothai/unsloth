// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  ActivityIcon,
  ChevronDownIcon,
  CircleIcon,
  PowerOffIcon,
  RefreshCwIcon,
} from "lucide-react";
import {
  type ReactElement,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  getApiMonitor,
  getApiMonitorEntry,
  getInferenceStatus,
  unloadModel,
} from "../../chat/api/chat-api";
import { resolveInferenceCheckpointId } from "../../chat/lib/apply-inference-status-to-store";
import { useChatRuntimeStore } from "../../chat/stores/chat-runtime-store";
import type { ApiMonitorEntry, ApiMonitorResponse } from "../../chat/types/api";

const API_INFERENCE_PREFIX_RE = /^\/api\/inference/;
const V1_PREFIX_RE = /^\/v1\//;
const PAGE_SIZE = 5;

function formatTime(value: number): string {
  return new Date(value * 1000).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function formatDuration(value?: number | null): string {
  if (value == null) {
    return "Running";
  }
  if (value < 1000) {
    return `${value} ms`;
  }
  return `${(value / 1000).toFixed(value < 10000 ? 1 : 0)} s`;
}

function formatTokens(entry: ApiMonitorEntry): string {
  if (entry.total_tokens != null) {
    return `${entry.total_tokens.toLocaleString()} tokens`;
  }
  if (entry.prompt_tokens != null || entry.completion_tokens != null) {
    const prompt = entry.prompt_tokens ?? 0;
    const completion = entry.completion_tokens ?? 0;
    return `${(prompt + completion).toLocaleString()} tokens`;
  }
  return "Tokens pending";
}

function compactEndpoint(endpoint: string): string {
  return endpoint
    .replace(API_INFERENCE_PREFIX_RE, "/api")
    .replace(V1_PREFIX_RE, "/");
}

function statusTone(status: ApiMonitorEntry["status"]): string {
  if (status === "running") {
    return "text-emerald-500";
  }
  if (status === "error") {
    return "text-destructive";
  }
  if (status === "cancelled") {
    return "text-amber-500";
  }
  return "text-muted-foreground";
}

function UsageBar({ value }: { value?: number | null }): ReactElement | null {
  if (value == null) {
    return null;
  }
  const pct = Math.max(0, Math.min(100, Math.round(value * 100)));
  return (
    <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-muted">
      <div
        className="h-full rounded-full bg-control-accent"
        style={{ width: `${pct}%` }}
      />
    </div>
  );
}

function isLifecycle(entry: ApiMonitorEntry): boolean {
  return entry.kind === "lifecycle";
}

function lifecycleLabel(entry: ApiMonitorEntry): string {
  if (entry.event === "unload") {
    return entry.reason === "idle" ? "Model unloaded (idle)" : "Model unloaded";
  }
  if (entry.event === "download") {
    if (entry.status === "running") {
      const pct = entry.progress;
      return typeof pct === "number"
        ? `Downloading model (${Math.round(pct)}%)`
        : "Downloading model";
    }
    if (entry.status === "completed") return "Model downloaded";
    // A cancel is deliberate, so saying it failed misreads the user's own action.
    return entry.status === "cancelled"
      ? "Model download cancelled"
      : "Model download failed";
  }
  if (entry.status === "running") {
    return "Loading model";
  }
  if (entry.status === "completed") {
    return "Model loaded";
  }
  return "Model load failed";
}

// Load/unload rows: label, model and time. No prompt or detail, so nothing to expand.
function LifecycleEntry({ entry }: { entry: ApiMonitorEntry }): ReactElement {
  return (
    <article className="min-w-0 rounded-lg border border-border/70 bg-muted/25">
      <div className="flex w-full min-w-0 items-start justify-between gap-3 p-3">
        <div className="min-w-0">
          <div className="flex min-w-0 items-center gap-2">
            <ActivityIcon
              className={cn("size-3.5 shrink-0", statusTone(entry.status))}
            />
            <span className="truncate text-xs font-medium">
              {lifecycleLabel(entry)}
            </span>
          </div>
          <div className="mt-1 truncate text-ui-11 text-muted-foreground">
            {entry.model}
          </div>
        </div>
        <div className="shrink-0 text-right text-ui-11 text-muted-foreground">
          <div>{formatTime(entry.started_at)}</div>
          {entry.event === "load" || entry.event === "download" ? (
            <div>{formatDuration(entry.duration_ms)}</div>
          ) : null}
        </div>
      </div>
    </article>
  );
}

function MonitorEntry({
  entry,
  detail,
  expanded,
  loading,
  onToggle,
}: {
  entry: ApiMonitorEntry;
  detail?: ApiMonitorEntry;
  expanded: boolean;
  loading: boolean;
  onToggle: () => void;
}): ReactElement {
  const hasCurrentDetail =
    detail &&
    detail.status === entry.status &&
    detail.updated_at >= entry.updated_at;
  const prompt = detail?.prompt ?? entry.prompt_preview;
  const replyText = hasCurrentDetail
    ? detail.error ?? detail.reply ?? entry.error ?? entry.reply_preview
    : entry.error ?? entry.reply_preview;
  const reply = replyText || (entry.status === "running" ? "Waiting..." : "No reply");

  return (
    <article className="min-w-0 rounded-lg border border-border/70 bg-background">
      <button
        type="button"
        onClick={onToggle}
        className="flex w-full min-w-0 items-start justify-between gap-3 p-3 text-left"
        aria-expanded={expanded}
      >
        <div className="min-w-0">
          <div className="flex min-w-0 items-center gap-2">
            <CircleIcon
              className={cn("size-2.5 fill-current", statusTone(entry.status))}
            />
            <span className="truncate text-xs font-medium">
              {compactEndpoint(entry.endpoint)}
            </span>
          </div>
          <div className="mt-1 truncate text-ui-11 text-muted-foreground">
            {entry.model}
          </div>
          <div className="mt-2 line-clamp-2 whitespace-pre-wrap break-words text-xs text-muted-foreground">
            {entry.error ||
              entry.reply_preview ||
              entry.prompt_preview ||
              (entry.status === "running" ? "Waiting..." : "No preview")}
          </div>
        </div>
        <div className="flex shrink-0 items-start gap-2 text-right text-ui-11 text-muted-foreground">
          <div>
            <div>{formatTime(entry.started_at)}</div>
            <div>{formatDuration(entry.duration_ms)}</div>
          </div>
          <ChevronDownIcon
            className={cn(
              "mt-0.5 size-3.5 transition-transform",
              expanded && "rotate-180",
            )}
          />
        </div>
      </button>

      {expanded ? (
        <div className="border-t border-border/60 p-3 pt-2">
          <div className="grid gap-2">
            <div>
              <div className="mb-1 flex items-center justify-between gap-2 text-ui-10 font-semibold uppercase text-muted-foreground">
                <span>Prompt</span>
                {entry.prompt_truncated && !detail ? <span>Preview</span> : null}
              </div>
              <pre className="max-h-44 overflow-auto whitespace-pre-wrap break-words rounded-md bg-muted/45 p-2 text-xs leading-5">
                {loading && !detail ? "Loading..." : prompt || "No prompt text"}
              </pre>
            </div>
            <div>
              <div className="mb-1 flex items-center justify-between gap-2 text-ui-10 font-semibold uppercase text-muted-foreground">
                <span>Reply</span>
                {entry.reply_truncated && !detail ? <span>Preview</span> : null}
              </div>
              <pre className="max-h-44 overflow-auto whitespace-pre-wrap break-words rounded-md bg-muted/45 p-2 text-xs leading-5">
                {loading && !detail ? "Loading..." : reply}
              </pre>
            </div>
          </div>

          <div className="mt-3 text-ui-11 text-muted-foreground">
            {formatTokens(entry)}
            {entry.context_length ? (
              <> / {entry.context_length.toLocaleString()} context</>
            ) : null}
            <UsageBar value={entry.context_usage} />
          </div>
        </div>
      ) : null}
    </article>
  );
}

export function ApiMonitorConsole(): ReactElement {
  const [data, setData] = useState<ApiMonitorResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [unloading, setUnloading] = useState(false);
  const [expandedIds, setExpandedIds] = useState<Set<string>>(() => new Set());
  const [details, setDetails] = useState<Record<string, ApiMonitorEntry>>({});
  const [loadingDetails, setLoadingDetails] = useState<Set<string>>(
    () => new Set(),
  );
  const loadingDetailsRef = useRef<Set<string>>(new Set());
  const detailsRef = useRef<Record<string, ApiMonitorEntry>>({});

  const loadMonitor = useCallback(async (): Promise<void> => {
    setRefreshing(true);
    try {
      setData(await getApiMonitor());
      setError(null);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Monitor unavailable");
    } finally {
      setRefreshing(false);
    }
  }, []);

  // /unload matches on the internal id, which the monitor omits, so read it from status.
  const unloadActiveModel = useCallback(async (): Promise<void> => {
    setUnloading(true);
    try {
      const status = await getInferenceStatus();
      const checkpoint = resolveInferenceCheckpointId(status);
      if (!checkpoint) {
        setError(null);
        return;
      }
      await unloadModel({ model_path: checkpoint });
      // Same as the chat eject flow: the store still holds the freed checkpoint.
      useChatRuntimeStore.getState().clearCheckpoint();
      setError(null);
      await loadMonitor();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to unload the model");
    } finally {
      setUnloading(false);
    }
  }, [loadMonitor]);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;

    function schedule(): void {
      timer = window.setTimeout(poll, 1500);
    }

    function poll(): void {
      getApiMonitor()
        .then((next) => {
          if (cancelled) {
            return;
          }
          setData(next);
          setError(null);
        })
        .catch((err: unknown) => {
          if (cancelled) {
            return;
          }
          setError(err instanceof Error ? err.message : "Monitor unavailable");
        })
        .finally(() => {
          if (!cancelled) {
            schedule();
          }
        });
    }

    poll();
    return () => {
      cancelled = true;
      if (timer !== undefined) {
        window.clearTimeout(timer);
      }
    };
  }, []);

  const statusLabel = data?.status ?? "idle";
  const hasActive = (data?.active_requests ?? 0) > 0;
  const entries = useMemo(() => data?.entries ?? [], [data]);
  // Older backends omit the field; only an explicit `false` means logging is off.
  const loggingDisabled = data?.logging_enabled === false;

  // Page 1 tracks the live list; paging back freezes the id order so history holds still.
  const [page, setPage] = useState(0);
  const [frozenIds, setFrozenIds] = useState<string[] | null>(null);
  const byId = useMemo(
    () => new Map(entries.map((entry) => [entry.id, entry])),
    [entries],
  );
  const ordered = useMemo(() => {
    if (frozenIds === null) {
      return entries;
    }
    return frozenIds.flatMap((id) => {
      const entry = byId.get(id);
      return entry ? [entry] : [];
    });
  }, [byId, entries, frozenIds]);
  const pageCount = Math.max(1, Math.ceil(ordered.length / PAGE_SIZE));
  const pageIndex = Math.min(page, pageCount - 1);
  const visible = ordered.slice(
    pageIndex * PAGE_SIZE,
    pageIndex * PAGE_SIZE + PAGE_SIZE,
  );
  const newerCount =
    frozenIds === null
      ? 0
      : entries.filter((entry) => !frozenIds.includes(entry.id)).length;

  const goToPage = useCallback(
    (next: number): void => {
      if (next <= 0) {
        setFrozenIds(null);
        setPage(0);
        return;
      }
      // Freeze on the way off page 1 so the history under the cursor holds still.
      setFrozenIds((prev) => prev ?? entries.map((entry) => entry.id));
      setPage(next);
    },
    [entries],
  );

  const loadDetail = useCallback(
    (id: string): void => {
      if (loadingDetailsRef.current.has(id)) {
        return;
      }
      loadingDetailsRef.current.add(id);
      setLoadingDetails((prev) => new Set(prev).add(id));
      getApiMonitorEntry(id)
        .then((entry) => {
          setDetails((prev) => {
            const next = { ...prev, [id]: entry };
            detailsRef.current = next;
            return next;
          });
        })
        .catch(() => {
          setDetails((prev) => {
            const next = { ...prev };
            delete next[id];
            detailsRef.current = next;
            return next;
          });
        })
        .finally(() => {
          loadingDetailsRef.current.delete(id);
          setLoadingDetails((prev) => {
            const next = new Set(prev);
            next.delete(id);
            return next;
          });
        });
    },
    [],
  );

  const toggleEntry = useCallback(
    (entry: ApiMonitorEntry): void => {
      setExpandedIds((prev) => {
        const next = new Set(prev);
        if (next.has(entry.id)) {
          next.delete(entry.id);
        } else {
          next.add(entry.id);
          loadDetail(entry.id);
        }
        return next;
      });
    },
    [loadDetail],
  );

  useEffect(() => {
    // Only rows on screen: an expanded row on another page would keep polling.
    for (const entry of visible) {
      if (isLifecycle(entry) || !expandedIds.has(entry.id)) {
        continue;
      }
      const cached = detailsRef.current[entry.id];
      if (!cached || cached.status !== entry.status || entry.status === "running") {
        loadDetail(entry.id);
      }
    }
  }, [visible, expandedIds, loadDetail]);

  return (
    <section className="flex min-w-0 flex-col rounded-lg border border-border/70 bg-background">
      <div className="flex min-w-0 items-start justify-between gap-3 border-b border-border/60 px-4 py-3">
        <div className="flex min-w-0 gap-3">
          <div className="relative mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-md border border-border/70 bg-muted/40">
            <ActivityIcon className="size-4 text-foreground" />
            {hasActive ? (
              <span className="absolute right-1 top-1 size-2 rounded-full bg-emerald-500" />
            ) : null}
          </div>
          <div className="min-w-0">
            <h2 className="text-sm font-semibold text-foreground">
              API monitor
            </h2>
            <p className="truncate text-xs text-muted-foreground">
              {data?.active_model ?? "No model loaded"}
            </p>
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <div className="rounded-full border border-border px-2.5 py-1 text-xs capitalize text-muted-foreground">
            {statusLabel}
          </div>
          {/* Always rendered, disabled when idle: the only manual release must stay visible. */}
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={() => void unloadActiveModel()}
            disabled={unloading || !data?.active_model}
            title={
              data?.active_model
                ? "Unload the model and free its VRAM"
                : "No model is loaded"
            }
          >
            <PowerOffIcon className="size-3.5" />
            {unloading ? "Unloading" : "Unload"}
          </Button>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={() => void loadMonitor()}
            disabled={refreshing}
          >
            <RefreshCwIcon
              className={cn("size-3.5", refreshing && "animate-spin")}
            />
            Refresh
          </Button>
        </div>
      </div>

      <div className="flex items-center justify-between border-b border-border/60 px-4 py-2 text-xs text-muted-foreground">
        <span>
          {(data?.active_requests ?? 0).toLocaleString()} active
          {loggingDisabled ? null : (
            <> / {entries.length.toLocaleString()} recent</>
          )}
        </span>
        {data?.context_length ? (
          <span>{data.context_length.toLocaleString()} context</span>
        ) : null}
      </div>

      <div className="max-h-[420px] min-h-24 overflow-y-auto p-3">
        {error ? (
          <div className="rounded-lg border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
            {error}
          </div>
        ) : loggingDisabled ? (
          <div className="rounded-lg border border-border/70 p-4 text-sm text-muted-foreground">
            The API monitor is disabled by{" "}
            <code className="rounded bg-muted/60 px-1 py-0.5 text-xs">
              UNSLOTH_STUDIO_DISABLE_API_MONITOR
            </code>
            . Requests and model loads still run normally, they are just not recorded
            here. Unset the variable and restart Studio to re-enable.
          </div>
        ) : entries.length === 0 ? (
          <div className="rounded-lg border border-border/70 p-4 text-sm text-muted-foreground">
            No API traffic yet
          </div>
        ) : (
          <div className="grid gap-3">
            {visible.map((entry) =>
              isLifecycle(entry) ? (
                <LifecycleEntry key={entry.id} entry={entry} />
              ) : (
                <MonitorEntry
                  key={entry.id}
                  entry={entry}
                  detail={details[entry.id]}
                  expanded={expandedIds.has(entry.id)}
                  loading={loadingDetails.has(entry.id)}
                  onToggle={() => toggleEntry(entry)}
                />
              ),
            )}
          </div>
        )}
      </div>

      {/* Also while frozen: retention can shrink that list below one page, and hiding the
          pager would strand the console on a stale snapshot. */}
      {ordered.length > PAGE_SIZE || frozenIds !== null ? (
        <div className="flex items-center justify-between gap-2 border-t border-border/60 px-4 py-2 text-xs text-muted-foreground">
          <span>
            Page {pageIndex + 1} of {pageCount}
            {newerCount > 0 ? ` (${newerCount.toLocaleString()} new)` : ""}
          </span>
          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="sm"
              className="h-7 px-2 text-xs"
              onClick={() => goToPage(pageIndex - 1)}
              disabled={pageIndex === 0 && frozenIds === null}
            >
              Newer
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="h-7 px-2 text-xs"
              onClick={() => goToPage(pageIndex + 1)}
              disabled={pageIndex >= pageCount - 1}
            >
              Older
            </Button>
          </div>
        </div>
      ) : null}
    </section>
  );
}
