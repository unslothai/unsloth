import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { PRODUCT_NAME } from "@/config/branding";
import { fetchDeviceType, usePlatformStore } from "@/config/env";
import { getInferenceStatus, unloadModel } from "@/features/chat/api/chat-api";
import { resolveInferenceCheckpointId } from "@/features/chat/lib/apply-inference-status-to-store";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import type { ApiMonitorEntry } from "@/features/chat";
import { isExternalModelId } from "@/features/chat/external-providers";
import { modelIdsMatch } from "@/features/hub/lib/model-identity";
import { useSettingsDialogStore } from "@/features/settings";
import { remoteApiOrigin } from "@/features/settings/api/remote-access-state";
import { getApiBase, isTauri } from "@/lib/api-base";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import {
  Copy01Icon,
  Delete02Icon,
  Globe02Icon,
  PauseIcon,
  PlayIcon,
  PowerSocket01Icon,
  RefreshIcon,
  Settings02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useEffect, useMemo, useRef, useState } from "react";
import { SavedModelSettingsPanel } from "./components/saved-model-settings";
import { isLifecycleEntry, lifecycleLabel } from "./lifecycle";
import { unloadResident } from "./unload-resident";
import {
  type MonitorStatusFilter,
  filterEntries,
  useApiMonitor,
} from "./use-api-monitor";

const API_INFERENCE_PREFIX_RE = /^\/api\/inference/;
const V1_PREFIX_RE = /^\/v1\//;
// Bounded because the usual failure is an entry aged out of the ring buffer.
const DETAIL_FETCH_ATTEMPTS = 3;

const STATUS_FILTERS: { value: MonitorStatusFilter; label: string }[] = [
  { value: "all", label: "All requests" },
  { value: "running", label: "In flight" },
  { value: "completed", label: "Completed" },
  { value: "error", label: "Errors" },
  { value: "cancelled", label: "Cancelled" },
];

function formatTime(value: number): string {
  return new Date(value * 1000).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function formatDuration(value?: number | null): string {
  if (value == null) {
    return "running";
  }
  if (value < 1000) {
    return `${Math.round(value)} ms`;
  }
  return `${(value / 1000).toFixed(value < 10000 ? 1 : 0)} s`;
}

function formatCount(value: number): string {
  return value.toLocaleString();
}

function formatTokPerSec(value?: number | null): string | null {
  if (value == null || value <= 0) {
    return null;
  }
  return `${value >= 100 ? Math.round(value) : value.toFixed(1)} tok/s`;
}

function compactEndpoint(endpoint: string): string {
  return endpoint
    .replace(API_INFERENCE_PREFIX_RE, "/api")
    .replace(V1_PREFIX_RE, "/");
}

function statusDotClass(status: ApiMonitorEntry["status"]): string {
  switch (status) {
    case "running":
      return "bg-blue-500 animate-pulse";
    case "error":
      return "bg-red-500";
    case "cancelled":
      return "bg-amber-500";
    default:
      return "bg-emerald-500";
  }
}

function statusTextClass(status: ApiMonitorEntry["status"]): string {
  switch (status) {
    case "running":
      return "text-blue-600 dark:text-blue-400";
    case "error":
      return "text-red-600 dark:text-red-400";
    case "cancelled":
      return "text-amber-600 dark:text-amber-500";
    default:
      return "text-emerald-600 dark:text-emerald-500";
  }
}

function StatCard({
  label,
  value,
  hint,
  tone,
}: {
  label: string;
  value: string;
  hint?: string;
  tone?: "default" | "error" | "active";
}): ReactElement {
  return (
    <div className="flex min-w-0 flex-col gap-1 rounded-xl border border-border/60 bg-card px-4 py-3">
      <span className="truncate text-ui-11 font-medium uppercase tracking-wider text-muted-foreground">
        {label}
      </span>
      <span
        className={cn(
          "truncate text-ui-22 font-semibold leading-tight tracking-[-0.02em]",
          tone === "error" && "text-red-600 dark:text-red-400",
          tone === "active" && "text-blue-600 dark:text-blue-400",
        )}
      >
        {value}
      </span>
      {hint ? (
        <span className="truncate text-ui-11 text-muted-foreground">
          {hint}
        </span>
      ) : null}
    </div>
  );
}

function CopyButton({
  value,
  label,
}: {
  value: string;
  label: string;
}): ReactElement {
  const [copied, setCopied] = useState(false);
  // Cancel on cleanup, or navigating away mid-flash sets state after unmount.
  const timerRef = useRef<number | undefined>(undefined);
  useEffect(
    () => () => {
      if (timerRef.current !== undefined) window.clearTimeout(timerRef.current);
    },
    [],
  );
  return (
    <Button
      type="button"
      variant="ghost"
      size="sm"
      aria-label={label}
      onClick={async () => {
        if (await copyToClipboard(value)) {
          setCopied(true);
          if (timerRef.current !== undefined) {
            window.clearTimeout(timerRef.current);
          }
          timerRef.current = window.setTimeout(() => setCopied(false), 1800);
        }
      }}
      className="h-7 shrink-0 gap-1.5 px-2 text-ui-11"
    >
      <HugeiconsIcon
        icon={copied ? Tick02Icon : Copy01Icon}
        strokeWidth={1.75}
        className={cn(
          "size-3.5",
          copied && "text-emerald-600 dark:text-emerald-500",
        )}
      />
      {copied ? "Copied" : "Copy"}
    </Button>
  );
}

function ContextUsageBar({
  value,
}: { value?: number | null }): ReactElement | null {
  if (value == null) {
    return null;
  }
  const pct = Math.max(0, Math.min(100, Math.round(value * 100)));
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
        <div
          className={cn(
            "h-full rounded-full transition-[width]",
            // Near-full context is the usual cause of truncated replies.
            pct >= 90
              ? "bg-red-500"
              : pct >= 75
                ? "bg-amber-500"
                : "bg-control-accent",
          )}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="shrink-0 text-ui-11 tabular-nums text-muted-foreground">
        {pct}%
      </span>
    </div>
  );
}

function RequestRow({
  entry,
  selected,
  onSelect,
}: {
  entry: ApiMonitorEntry;
  selected: boolean;
  onSelect: () => void;
}): ReactElement {
  const preview =
    entry.error ||
    entry.reply_preview ||
    entry.prompt_preview ||
    (entry.status === "running" ? "Waiting for output…" : "No preview");
  // A lifecycle row has no payload, so it reads as a status line.
  if (isLifecycleEntry(entry)) {
    return (
      <div className="flex w-full min-w-0 flex-col gap-1 border-b border-border/50 bg-muted/25 px-4 py-3 last:border-b-0">
        <div className="flex min-w-0 items-center gap-2">
          <span
            className={cn(
              "size-2 shrink-0 rounded-full",
              statusDotClass(entry.status),
            )}
            aria-hidden={true}
          />
          <span className="truncate text-ui-13 font-medium text-foreground">
            {lifecycleLabel(entry)}
          </span>
          <span className="ml-auto shrink-0 text-ui-11 tabular-nums text-muted-foreground">
            {formatTime(entry.started_at)}
          </span>
        </div>
        <div className="min-w-0 truncate pl-4 text-ui-11 text-muted-foreground">
          {entry.model}
        </div>
        {entry.error ? (
          <div className="min-w-0 break-words pl-4 text-ui-11 text-red-600 dark:text-red-400">
            {entry.error}
          </div>
        ) : null}
      </div>
    );
  }
  return (
    <button
      type="button"
      onClick={onSelect}
      aria-current={selected}
      className={cn(
        "flex w-full min-w-0 flex-col gap-1 border-b border-border/50 px-4 py-3 text-left transition-colors last:border-b-0",
        selected ? "bg-muted/70" : "hover:bg-muted/40",
      )}
    >
      <div className="flex min-w-0 items-center gap-2">
        <span
          className={cn(
            "size-2 shrink-0 rounded-full",
            statusDotClass(entry.status),
          )}
          aria-hidden={true}
        />
        <span className="truncate text-ui-13 font-medium text-foreground">
          {compactEndpoint(entry.endpoint)}
        </span>
        <span className="ml-auto shrink-0 text-ui-11 tabular-nums text-muted-foreground">
          {formatDuration(entry.duration_ms)}
        </span>
      </div>
      <div className="flex min-w-0 items-center gap-2 pl-4">
        <span className="truncate text-ui-11 text-muted-foreground">
          {entry.model}
        </span>
        <span className="ml-auto flex shrink-0 items-center gap-2 text-ui-11 tabular-nums text-muted-foreground">
          {formatTokPerSec(entry.tok_per_sec) ? (
            <span>{formatTokPerSec(entry.tok_per_sec)}</span>
          ) : null}
          <span>{formatTime(entry.started_at)}</span>
        </span>
      </div>
      <p
        className={cn(
          "line-clamp-2 pl-4 text-ui-11 leading-[1.45]",
          entry.error
            ? "text-red-600 dark:text-red-400"
            : "text-muted-foreground",
        )}
      >
        {preview}
      </p>
    </button>
  );
}

function PayloadBlock({
  title,
  body,
  truncated,
  loading,
  tone,
}: {
  title: string;
  body: string;
  truncated?: boolean;
  loading?: boolean;
  tone?: "error";
}): ReactElement {
  return (
    <section className="flex min-w-0 flex-col gap-1.5">
      <div className="flex items-center justify-between gap-2">
        <h3 className="text-ui-11 font-semibold uppercase tracking-wider text-muted-foreground">
          {title}
        </h3>
        <div className="flex items-center gap-1">
          {truncated ? (
            <span className="text-ui-10 text-muted-foreground">
              preview only
            </span>
          ) : null}
          {body ? (
            <CopyButton value={body} label={`Copy ${title.toLowerCase()}`} />
          ) : null}
        </div>
      </div>
      <pre
        className={cn(
          "max-h-72 overflow-auto whitespace-pre-wrap break-words rounded-lg bg-muted/50 p-3 text-ui-11 leading-[1.55]",
          tone === "error" && "bg-red-500/5 text-red-700 dark:text-red-400",
        )}
      >
        {loading && !body ? "Loading…" : body || "–"}
      </pre>
    </section>
  );
}

function RequestDetail({
  entry,
  detail,
  loading,
}: {
  entry: ApiMonitorEntry;
  detail?: ApiMonitorEntry;
  loading: boolean;
}): ReactElement {
  // The detail fetch can describe an older state of a streaming entry, so prefer it
  // only once it is as fresh as the list row, or the panel rewinds.
  const detailIsCurrent =
    detail != null &&
    detail.status === entry.status &&
    detail.updated_at >= entry.updated_at;
  const prompt = detail?.prompt ?? entry.prompt_preview;
  const reply = detailIsCurrent
    ? (detail.reply ?? entry.reply_preview)
    : entry.reply_preview;


  return (
    <div className="flex min-w-0 flex-col gap-5 p-5">
      <header className="flex min-w-0 flex-col gap-2">
        <div className="flex min-w-0 items-center gap-2">
          <span
            className={cn(
              "size-2 shrink-0 rounded-full",
              statusDotClass(entry.status),
            )}
            aria-hidden={true}
          />
          <span
            className={cn(
              "text-ui-11 font-semibold uppercase tracking-wider",
              statusTextClass(entry.status),
            )}
          >
            {entry.status}
          </span>
          <span className="ml-auto shrink-0 font-mono text-ui-10 text-muted-foreground">
            {entry.id}
          </span>
        </div>
        <h2 className="min-w-0 break-all font-mono text-ui-13 font-medium text-foreground">
          {entry.method} {entry.endpoint}
        </h2>
        <p className="min-w-0 break-all text-ui-11 text-muted-foreground">
          {entry.model}
        </p>
      </header>

      <dl className="grid grid-cols-2 gap-x-4 gap-y-3 rounded-xl border border-border/60 bg-card px-4 py-3 sm:grid-cols-3">
        {[
          { label: "Started", value: formatTime(entry.started_at) },
          { label: "Duration", value: formatDuration(entry.duration_ms) },
          {
            label: "Prompt tokens",
            value:
              entry.prompt_tokens != null
                ? formatCount(entry.prompt_tokens)
                : "–",
          },
          {
            label: "Completion tokens",
            value:
              entry.completion_tokens != null
                ? formatCount(entry.completion_tokens)
                : "–",
          },
          {
            label: "Total tokens",
            value:
              entry.total_tokens != null
                ? formatCount(entry.total_tokens)
                : "–",
          },
          {
            label: "Context",
            value:
              entry.context_length != null
                ? formatCount(entry.context_length)
                : "–",
          },
          {
            label: "First token",
            value: entry.ttft_ms != null ? formatDuration(entry.ttft_ms) : "–",
          },
          // Duration minus this is the queue wait, not slow decoding.
          {
            label: "Generating",
            value: entry.decode_ms != null ? formatDuration(entry.decode_ms) : "–",
          },
          {
            label: "Prompt speed",
            value: formatTokPerSec(entry.prompt_tok_per_sec) ?? "–",
          },
          {
            label: "Generation speed",
            value: formatTokPerSec(entry.tok_per_sec) ?? "–",
          },
          {
            label: "Stop reason",
            value: entry.stop_reason ?? "–",
          },
        ].map((item) => (
          <div key={item.label} className="flex min-w-0 flex-col gap-0.5">
            <dt className="truncate text-ui-10 font-medium uppercase tracking-wider text-muted-foreground">
              {item.label}
            </dt>
            <dd className="truncate text-ui-13 tabular-nums text-foreground">
              {item.value}
            </dd>
          </div>
        ))}
      </dl>

      {entry.context_usage != null ? (
        <div className="flex flex-col gap-1.5">
          <span className="text-ui-11 font-semibold uppercase tracking-wider text-muted-foreground">
            Context used
          </span>
          <ContextUsageBar value={entry.context_usage} />
        </div>
      ) : null}

      {entry.error ? (
        <PayloadBlock title="Error" body={entry.error} tone="error" />
      ) : null}

      <PayloadBlock
        title="Prompt"
        body={prompt}
        truncated={entry.prompt_truncated && detail?.prompt == null}
        loading={loading}
      />
      <PayloadBlock
        title="Reply"
        body={reply}
        truncated={entry.reply_truncated && !detailIsCurrent}
        loading={loading}
      />
    </div>
  );
}

export function ApiMonitorPage(): ReactElement {
  const {
    data,
    entries,
    stats,
    error,
    loading,
    refreshing,
    paused,
    setPaused,
    refresh,
    clear,
    details,
    loadingDetails,
    requestDetail,
  } = useApiMonitor();
  const serverUrl = usePlatformStore((s) => s.serverUrl);
  const cloudflareUrl = usePlatformStore((s) => s.cloudflareUrl);
  const [unloading, setUnloading] = useState(false);
  const [unloadError, setUnloadError] = useState<string | null>(null);

  useEffect(() => {
    const refreshRemoteBase = () => {
      void fetchDeviceType({ force: true });
    };
    refreshRemoteBase();
    window.addEventListener("focus", refreshRemoteBase);
    return () => window.removeEventListener("focus", refreshRemoteBase);
  }, []);

  // Manual release so VRAM frees without the idle timer. /unload matches on the
  // internal id, which the monitor does not carry, so read status. unloadResident owns
  // the read/unload/recheck sequence: this page's own feature (an API auto-switch) can
  // swap the model out from under the read, and /unload naming a replaced model is a
  // successful no-op, so one pass would report success over the model still resident.
  const unloadActiveModel = async (): Promise<void> => {
    setUnloading(true);
    try {
      const { unloadedAliases, stillResident } = await unloadResident({
        readResident: async () => {
          const status = await getInferenceStatus();
          const checkpoint = resolveInferenceCheckpointId(status);
          if (!checkpoint) {
            return null;
          }
          // Both spellings, since status reports the load path while the store may
          // hold the advertised repo id.
          return {
            checkpoint,
            aliases: [checkpoint, status.active_model].filter(
              (alias): alias is string => alias != null,
            ),
          };
        },
        unload: (checkpoint) => unloadModel({ model_path: checkpoint }),
      });
      // As in the chat eject flow, but only when the store holds a model just unloaded: chat
      // can have an external provider selected while a local model stays resident, and
      // clearCheckpoint would delete a selection this button never touched.
      const store = useChatRuntimeStore.getState();
      const selected = store.params.checkpoint;
      if (
        selected &&
        !isExternalModelId(selected) &&
        unloadedAliases.some((alias) => modelIdsMatch(selected, alias))
      ) {
        store.clearCheckpoint();
      }
      setUnloadError(
        stillResident
          ? `The API loaded "${stillResident}" while unloading, so it is still using memory. Unload again to release it.`
          : null,
      );
      refresh();
    } catch (err: unknown) {
      setUnloadError(
        err instanceof Error ? err.message : "Failed to unload the model",
      );
    } finally {
      setUnloading(false);
    }
  };
  const [statusFilter, setStatusFilter] = useState<MonitorStatusFilter>("all");
  const [query, setQuery] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const visible = useMemo(
    () => filterEntries(entries, statusFilter, query),
    [entries, statusFilter, query],
  );
  const selected = useMemo(
    () => visible.find((entry) => entry.id === selectedId) ?? null,
    [visible, selectedId],
  );

  // Refetch the selected entry while it streams. Keyed on identity and revision, never
  // on `details`: the fetch rewrites it on success, so depending on it loops.
  const selectedId_ = selected?.id ?? null;
  const selectedUpdatedAt = selected?.updated_at ?? null;
  const selectedIsMissing = selectedId_ != null && details[selectedId_] == null;
  const lastFetchedRef = useRef<string | null>(null);
  const attemptsRef = useRef<{ revision: string; count: number }>({
    revision: "",
    count: 0,
  });
  const [retryTick, setRetryTick] = useState(0);
  // Flips as a fetch settles either way, which is what lets a failure be noticed.
  const detailInFlight = selectedId_ != null && loadingDetails.has(selectedId_);
  useEffect(() => {
    if (selectedId_ == null || detailInFlight) {
      return;
    }
    // `updated_at` advances per poll and settles when terminal; a missing payload
    // always retries, covering a fetch that failed late.
    const revision = `${selectedId_}@${selectedUpdatedAt ?? ""}`;
    if (!selectedIsMissing && lastFetchedRef.current === revision) {
      return;
    }
    // A terminal row's revision never advances, so a failed fetch had nothing to re-run
    // this effect. `loadingDetails` settling is the trigger; the count bounds it.
    if (attemptsRef.current.revision !== revision) {
      attemptsRef.current = { revision, count: 0 };
    }
    if (attemptsRef.current.count >= DETAIL_FETCH_ATTEMPTS) {
      return;
    }
    attemptsRef.current.count += 1;
    // Only when a fetch started: the guard can refuse, and recording it anyway skips
    // that revision for good.
    if (requestDetail(selectedId_)) {
      lastFetchedRef.current = revision;
      setRetryTick(0);
    } else {
      // Refused: an older fetch is running and no dep changes when it settles.
      const timer = window.setTimeout(() => setRetryTick((n) => n + 1), 250);
      return () => window.clearTimeout(timer);
    }
  }, [
    selectedId_,
    selectedUpdatedAt,
    selectedIsMissing,
    requestDetail,
    retryTick,
    detailInFlight,
  ]);

  // The desktop webview's origin is tauri://, and the packaged app picks its port
  // dynamically. Same source as the Agents tab.
  const origin = typeof window === "undefined" ? "" : window.location.origin;
  const localOrigin = isTauri ? (serverUrl ?? getApiBase()) : origin;
  const baseUrl = `${remoteApiOrigin(cloudflareUrl, localOrigin)}/v1`;
  const serverStatus = data?.status ?? "idle";
  // Older backends omit the field; only an explicit `false` means recording is off.
  const loggingDisabled = data?.logging_enabled === false;
  const statusCopy =
    serverStatus === "generating"
      ? "Serving requests"
      : serverStatus === "ready"
        ? "Ready"
        : "No model loaded";

  return (
    <main className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-6 pb-10 pt-12 font-heading sm:px-10">
      <header className="flex flex-wrap items-start justify-between gap-4">
        <div className="flex min-w-0 flex-col gap-1">
          <h1 className="text-ui-30 font-semibold leading-[1.04] tracking-[-0.028em] text-foreground sm:text-ui-34">
            API
          </h1>
          <p className="text-sm text-muted-foreground">
            Live traffic through {PRODUCT_NAME}&apos;s OpenAI-compatible server.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => setPaused(!paused)}
            className="h-9 gap-1.5 rounded-full"
          >
            <HugeiconsIcon
              icon={paused ? PlayIcon : PauseIcon}
              strokeWidth={1.75}
              className="size-4"
            />
            {paused ? "Resume" : "Pause"}
          </Button>
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => void unloadActiveModel()}
            disabled={unloading || !data?.active_model}
            title={
              data?.active_model
                ? `Unload ${data.active_model} and free its VRAM`
                : "No model is loaded"
            }
            className="h-9 gap-1.5 rounded-full"
          >
            <HugeiconsIcon
              icon={PowerSocket01Icon}
              strokeWidth={1.75}
              className="size-4"
            />
            {unloading ? "Unloading" : "Unload"}
          </Button>
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={refresh}
            disabled={refreshing}
            className="h-9 gap-1.5 rounded-full"
          >
            <HugeiconsIcon
              icon={RefreshIcon}
              strokeWidth={1.75}
              className={cn("size-4", refreshing && "animate-spin")}
            />
            Refresh
          </Button>
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={entries.length === 0}
            onClick={() => {
              setSelectedId(null);
              void clear();
            }}
            className="h-9 gap-1.5 rounded-full"
          >
            <HugeiconsIcon
              icon={Delete02Icon}
              strokeWidth={1.75}
              className="size-4"
            />
            Clear log
          </Button>
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() =>
              useSettingsDialogStore.getState().openDialog("api-keys")
            }
            className="h-9 gap-1.5 rounded-full"
          >
            <HugeiconsIcon
              icon={Settings02Icon}
              strokeWidth={1.75}
              className="size-4"
            />
            API settings
          </Button>
        </div>
      </header>

      {/* Checked first when a client can't reach the API: base URL and what is loaded. */}
      <section className="flex flex-wrap items-center gap-x-6 gap-y-3 rounded-xl border border-border/60 bg-card px-4 py-3">
        <div className="flex min-w-0 items-center gap-2.5">
          <span className="flex size-8 shrink-0 items-center justify-center rounded-lg border border-border/60 bg-muted/40">
            <HugeiconsIcon
              icon={Globe02Icon}
              strokeWidth={1.75}
              className="size-4"
            />
          </span>
          <div className="flex min-w-0 flex-col">
            <span className="text-ui-10 font-medium uppercase tracking-wider text-muted-foreground">
              Base URL
            </span>
            <span className="truncate font-mono text-ui-12 text-foreground">
              {baseUrl}
            </span>
          </div>
          <CopyButton value={baseUrl} label="Copy API base URL" />
        </div>
        <div className="flex min-w-0 flex-col">
          <span className="text-ui-10 font-medium uppercase tracking-wider text-muted-foreground">
            Status
          </span>
          <span className="flex items-center gap-1.5 text-ui-12 text-foreground">
            <span
              className={cn(
                "size-2 rounded-full",
                serverStatus === "generating"
                  ? "bg-blue-500 animate-pulse"
                  : serverStatus === "ready"
                    ? "bg-emerald-500"
                    : "bg-muted-foreground",
              )}
              aria-hidden={true}
            />
            {statusCopy}
          </span>
        </div>
        {data?.queue ? (
          <div className="flex min-w-0 flex-col">
            <span className="text-ui-10 font-medium uppercase tracking-wider text-muted-foreground">
              Slots
            </span>
            <span
              className={cn(
                "text-ui-12 tabular-nums text-foreground",
                data.queue.queued > 0 && "text-amber-700 dark:text-amber-500",
              )}
            >
              {data.queue.active}/{data.queue.capacity} busy
              {data.queue.queued > 0 ? ` · ${data.queue.queued} queued` : ""}
            </span>
          </div>
        ) : null}
        <div className="flex min-w-0 flex-1 flex-col">
          <span className="text-ui-10 font-medium uppercase tracking-wider text-muted-foreground">
            Loaded model
          </span>
          <span className="truncate text-ui-12 text-foreground">
            {data?.active_model ?? "None"}
            {data?.context_length
              ? ` · ${formatCount(data.context_length)} ctx`
              : ""}
          </span>
        </div>
        {paused ? (
          <span className="rounded-full border border-amber-500/40 bg-amber-500/10 px-2.5 py-1 text-ui-11 font-medium text-amber-700 dark:text-amber-500">
            Paused
          </span>
        ) : null}
      </section>

      {error || unloadError ? (
        <div className="rounded-xl border border-red-500/40 bg-red-500/5 px-4 py-3 text-sm text-red-600 dark:text-red-400">
          {error || unloadError}
        </div>
      ) : null}

      <section className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        <StatCard
          label="In flight"
          value={formatCount(stats.active)}
          tone={stats.active > 0 ? "active" : "default"}
        />
        <StatCard
          label="Requests"
          value={formatCount(stats.total)}
          hint={loggingDisabled ? "recording off" : "recent window"}
        />
        <StatCard label="Completed" value={formatCount(stats.completed)} />
        <StatCard
          label="Errors"
          value={formatCount(stats.errors)}
          tone={stats.errors > 0 ? "error" : "default"}
          hint={
            stats.errorRate != null
              ? `${Math.round(stats.errorRate * 100)}% of finished`
              : undefined
          }
        />
        <StatCard
          label="Avg latency"
          value={
            stats.avgDurationMs == null
              ? "–"
              : formatDuration(stats.avgDurationMs)
          }
          hint={
            stats.maxDurationMs != null
              ? `max ${formatDuration(stats.maxDurationMs)}`
              : undefined
          }
        />
        <StatCard
          label="Throughput"
          value={
            stats.tokensPerSecond == null
              ? "–"
              : `${stats.tokensPerSecond.toFixed(1)} tok/s`
          }
          hint={`${formatCount(stats.totalTokens)} tokens · generation only`}
        />
      </section>

      <section className="flex min-h-0 flex-col overflow-hidden rounded-xl border border-border/60 bg-card">
        <div className="flex flex-wrap items-center gap-2 border-b border-border/60 px-4 py-3">
          <Input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search model, endpoint, preview or error"
            aria-label="Search API requests"
            className="h-9 w-full min-w-0 flex-1 rounded-full border-none bg-muted shadow-none dark:bg-background sm:w-64 sm:flex-none"
          />
          <Select
            value={statusFilter}
            onValueChange={(value) =>
              setStatusFilter(value as MonitorStatusFilter)
            }
          >
            <SelectTrigger
              aria-label="Filter by status"
              className="h-9 w-[150px] rounded-full border-none bg-muted shadow-none dark:bg-background"
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {STATUS_FILTERS.map((option) => (
                <SelectItem key={option.value} value={option.value}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <span className="ml-auto shrink-0 text-ui-11 text-muted-foreground">
            {formatCount(visible.length)} of {formatCount(entries.length)}
          </span>
        </div>

        <div className="grid min-h-0 grid-cols-1 lg:grid-cols-[minmax(0,380px)_minmax(0,1fr)]">
          <div className="max-h-[560px] min-h-[220px] overflow-y-auto border-b border-border/60 lg:border-b-0 lg:border-r">
            {loading ? (
              <div className="flex flex-col gap-3 p-4">
                {[0, 1, 2].map((i) => (
                  <Skeleton key={i} className="h-16 w-full rounded-lg" />
                ))}
              </div>
            ) : visible.length === 0 ? (
              <p className="px-4 py-10 text-center text-sm text-muted-foreground">
                {entries.length > 0
                  ? "No requests match this filter."
                  : loggingDisabled
                    ? "Recording is off: UNSLOTH_STUDIO_DISABLE_API_MONITOR is set. Requests and model loads still run normally, they are just not listed here. Unset the variable and restart Studio to re-enable."
                    : "No API traffic yet. Point a client at the base URL above to see requests here."}
              </p>
            ) : (
              visible.map((entry) => (
                <RequestRow
                  key={entry.id}
                  entry={entry}
                  selected={entry.id === selectedId}
                  onSelect={() => setSelectedId(entry.id)}
                />
              ))
            )}
          </div>

          <div className="max-h-[560px] min-h-[220px] overflow-y-auto">
            {selected ? (
              <RequestDetail
                entry={selected}
                detail={details[selected.id]}
                loading={loadingDetails.has(selected.id)}
              />
            ) : (
              <p className="flex h-full items-center justify-center px-6 py-10 text-center text-sm text-muted-foreground">
                Select a request to inspect its prompt, reply, tokens and
                errors.
              </p>
            )}
          </div>
        </div>
      </section>

      <SavedModelSettingsPanel />
    </main>
  );
}
