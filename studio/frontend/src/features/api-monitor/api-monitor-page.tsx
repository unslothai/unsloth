// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useAppShellReadySignal } from "@/components/app-readiness";
import { GuidedTour, useGuidedTourController } from "@/features/tour";
import { apiMonitorTourSteps } from "./tour";

// Full-page monitor for Unsloth's OpenAI-compatible API server. Settings still owns
// configuration (keys, auto-switch, examples); this page owns observability.

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
import { fetchDeviceType, usePlatformStore } from "@/config/env";
import { getInferenceStatus, unloadModel } from "@/features/chat/api/chat-api";
import { resolveInferenceCheckpointId } from "@/features/chat/lib/apply-inference-status-to-store";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { type ApiMonitorEntry, useChatModelRuntime } from "@/features/chat";
import type { ApiMonitorResponse } from "@/features/chat/types/api";
import { isExternalModelId } from "@/features/chat/external-providers";
import { modelIdsMatch } from "@/features/hub/lib/model-identity";
import {
  type ModelConfigHandoffRequest,
  createModelConfigHandoffRequestId,
} from "@/features/model-picker";
import {
  loadLanAccess,
  publishLanUrls,
  useSettingsDialogStore,
} from "@/features/settings";
import { remoteApiOrigin } from "@/features/settings/api/remote-access-state";
import { getApiBase, isTauri } from "@/lib/api-base";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import {
  ApiIcon,
  Cancel01Icon,
  CodeIcon,
  ComputerTerminal01Icon,
  Copy01Icon,
  CursorPointer02Icon,
  Delete02Icon,
  FlashIcon,
  Key01Icon,
  PauseIcon,
  PlayIcon,
  Refresh01Icon,
  RemoveCircleIcon,
  Search01Icon,
  Settings02Icon,
  Timer01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type CSSProperties,
  type KeyboardEvent,
  type ReactElement,
  type ReactNode,
  memo,
  useDeferredValue,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { ApiExampleDialog } from "./components/api-example-dialog";
import { ApiModelLoadControls } from "./components/api-model-load-controls";
import { SavedModelSettingsPanel } from "./components/saved-model-settings";
import { VideoGenerationApiDocs } from "./video-generation-api-docs";
import { isLifecycleEntry, lifecycleLabel } from "./lifecycle";
import { useApiMonitorOverlayStore } from "./overlay-store";
import { useSingleLine } from "./use-single-line";
import { PAYLOAD_SHOW_ALL_MAX_CHARS, payloadPreview } from "./payload-preview";
import { ToolbarTooltip } from "./components/toolbar-tooltip";
import { unloadResident } from "./unload-resident";
import {
  residentHasRunSettings,
  residentHasSlotsSetting,
  keptRunSettingsRequest,
  runSettingsRequest,
  slotsSettingsRequest,
} from "./slots-settings-request";
import {
  monitorSelection,
  useResidentRunTarget,
} from "./use-resident-run-target";
import {
  type MonitorStatusFilter,
  computeStats,
  countByStatus,
  filterEntries,
  useApiMonitor,
} from "./use-api-monitor";

const API_INFERENCE_PREFIX_RE = /^\/api\/inference/;
const V1_PREFIX_RE = /^\/v1\//;
// Bounded because the usual failure is an entry aged out of the ring buffer.
const DETAIL_FETCH_ATTEMPTS = 3;

const ALL_MODELS = "__all__";

const STATUS_FILTERS: { value: MonitorStatusFilter; label: string }[] = [
  { value: "all", label: "All" },
  { value: "running", label: "In flight" },
  { value: "completed", label: "Completed" },
  { value: "error", label: "Errors" },
  { value: "cancelled", label: "Cancelled" },
];

const SERVER_STATUS: Record<
  ApiMonitorResponse["status"],
  { label: string; dot: string }
> = {
  generating: { label: "Serving", dot: "bg-blue-500 animate-pulse" },
  ready: { label: "Ready", dot: "bg-emerald-500" },
  idle: { label: "No model loaded", dot: "bg-muted-foreground/60" },
};

const PANE_HEIGHT =
  "max-h-[calc(560px*var(--ui-space-scale,1))] min-h-[calc(260px*var(--ui-space-scale,1))] max-lg:max-h-[45dvh] 3xl:max-h-[calc(100dvh-22rem*var(--ui-space-scale,1))] overflow-y-auto";

const SURFACE = "elevated-card bg-card";
const DIVIDER = "border-border/60 dark:border-background";
const INSET =
  "border border-border/50 bg-muted/40 dark:border-transparent dark:bg-[rgb(255_255_255_/_calc(0.05*var(--contrast-wash-gain,1)))]";
const INSET_OPAQUE =
  "bg-[color-mix(in_oklab,var(--muted)_40%,var(--card))] dark:bg-[rgb(255_255_255_/_calc(0.05*var(--contrast-wash-gain,1)))]";

function formatTime(value: number): string {
  return new Date(value * 1000).toLocaleTimeString([], {
    hour: "numeric",
    minute: "2-digit",
  });
}

function formatFullTime(value: number): string {
  return new Date(value * 1000).toLocaleString();
}

function formatAgo(value: number, now: number): string {
  const seconds = Math.max(0, Math.round(now - value));
  if (seconds < 10) return "now";
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return new Date(value * 1000).toLocaleDateString([], {
    month: "short",
    day: "numeric",
  });
}

const AGO_TICK_MS = 10_000;

// Server-clock seconds (host clock may differ over a tunnel); ticks itself since a paused feed stops polls.
function useServerNow(
  serverTime: number | undefined,
  ticking: boolean,
): number {
  const [tick, setTick] = useState(0);
  useEffect(() => {
    if (!ticking) return;
    const timer = window.setInterval(() => setTick((n) => n + 1), AGO_TICK_MS);
    return () => window.clearInterval(timer);
  }, [ticking]);
  const skew = useMemo(
    () => (serverTime == null ? 0 : Math.round(serverTime - Date.now() / 1000)),
    [serverTime],
  );
  // biome-ignore lint/correctness/useExhaustiveDependencies: tick is what advances the clock
  return useMemo(() => Date.now() / 1000 + skew, [skew, tick]);
}

function formatDuration(value?: number | null): string {
  if (value == null) {
    return "–";
  }
  if (value < 1000) {
    return `${Math.round(value)} ms`;
  }
  return `${(value / 1000).toFixed(value < 10000 ? 1 : 0)} s`;
}

function entryDuration(entry: ApiMonitorEntry): string {
  if (entry.duration_ms == null && entry.status === "running") {
    return "running";
  }
  return formatDuration(entry.duration_ms);
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

function quickStartSnippet(baseUrl: string, model: string): string {
  const body = JSON.stringify({
    model,
    messages: [{ role: "user", content: "Hello!" }],
  }).replace(/'/g, "'\\''");
  return `curl ${baseUrl}/chat/completions \\
  -H "Authorization: Bearer $UNSLOTH_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '${body}'`;
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

type ServedModel = {
  name: string;
  checkpoint: string;
  active: boolean;
  quant: string | null;
};

function servedModels(
  loaded: { id: string; quant?: string | null; checkpoint?: string }[],
  activeModel: string | null,
  residentCheckpoint: string | null | undefined,
): ServedModel[] {
  const activeId = monitorSelection(activeModel).id;
  // At most one primary row: two local files can share a public name, and marking both ejected the primary.
  const named = (model: { id: string }): boolean =>
    Boolean(activeId) && modelIdsMatch(model.id, activeId);
  const byCheckpoint = residentCheckpoint
    ? loaded.findIndex(
        (model) =>
          named(model) && (model.checkpoint ?? model.id) === residentCheckpoint,
      )
    : -1;
  const activeIndex =
    byCheckpoint >= 0 ? byCheckpoint : loaded.findIndex(named);
  const rows = loaded.map((model, index) => ({
    name:
      model.quant && !model.id.includes(":")
        ? `${model.id}:${model.quant}`
        : model.id,
    checkpoint: model.checkpoint ?? model.id,
    active: index === activeIndex,
    quant: model.quant ?? null,
  }));
  if (activeModel && !rows.some((row) => row.active)) {
    rows.unshift({
      name: activeModel,
      checkpoint: activeModel,
      active: true,
      quant: monitorSelection(activeModel).ggufVariant,
    });
  }
  return rows.sort((a, b) => Number(b.active) - Number(a.active));
}

function openApiSettings(): void {
  useSettingsDialogStore.getState().openDialog("api-keys");
}

function Stat({
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
    <div className="flex min-w-0 flex-col gap-0.5 bg-card px-5 py-4">
      <span className="truncate text-ui-12 text-muted-foreground">{label}</span>
      <span
        className={cn(
          "truncate text-ui-21 font-semibold leading-tight tracking-[-0.02em] tabular-nums text-foreground",
          value === "–" && "text-muted-foreground/70",
          tone === "error" && "text-red-600 dark:text-red-400",
          tone === "active" && "text-blue-600 dark:text-blue-400",
        )}
      >
        {value}
      </span>
      <span className="min-h-[1lh] truncate text-ui-11 text-muted-foreground">
        {hint}
      </span>
    </div>
  );
}

function CopyButton({
  value,
  label,
  revealOnHover = false,
}: {
  value: string;
  label: string;
  revealOnHover?: boolean;
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
    <ToolbarTooltip label={copied ? "Copied" : label}>
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
        className={cn(
          "size-7 shrink-0 rounded-full p-0",
          revealOnHover &&
            !copied &&
            "opacity-0 group-hover/copy:opacity-100 focus-visible:opacity-100 pointer-coarse:opacity-100",
        )}
      >
        <HugeiconsIcon
          icon={copied ? Tick02Icon : Copy01Icon}
          strokeWidth={1.75}
          className={cn(
            "size-3.5",
            copied && "text-emerald-600 dark:text-emerald-500",
          )}
        />
      </Button>
    </ToolbarTooltip>
  );
}

function Fact({
  label,
  children,
  className,
}: {
  label: string;
  children: ReactNode;
  className?: string;
}): ReactElement {
  return (
    <div className={cn("flex min-w-0 items-center gap-2", className)}>
      <dt className="shrink-0 text-ui-12 text-muted-foreground">{label}</dt>
      <dd className="flex min-w-0 items-center gap-1 text-ui-12 text-foreground">
        {children}
      </dd>
    </div>
  );
}

function Banner({
  children,
  onDismiss,
}: {
  children: ReactNode;
  onDismiss?: () => void;
}): ReactElement {
  const [ref, singleLine] = useSingleLine<HTMLDivElement>();
  return (
    <div
      ref={ref}
      role="alert"
      className={cn(
        "flex items-start gap-3 border border-red-500/40 bg-red-500/5 px-4 py-3 text-sm text-red-600 dark:text-red-400",
        singleLine ? "rounded-full" : "rounded-xl",
      )}
    >
      <span className="min-w-0 flex-1 break-words">{children}</span>
      {onDismiss ? (
        <button
          type="button"
          onClick={onDismiss}
          aria-label="Dismiss"
          className="-my-0.5 shrink-0 rounded-full p-0.5 opacity-70 transition-opacity hover:opacity-100"
        >
          <HugeiconsIcon
            icon={Cancel01Icon}
            strokeWidth={1.75}
            className="size-4"
          />
        </button>
      ) : null}
    </div>
  );
}

function ContextUsageBar({ value }: { value?: number | null }): ReactElement {
  if (value == null) {
    return <>–</>;
  }
  const pct = Math.max(0, Math.min(100, Math.round(value * 100)));
  return (
    <div className="flex h-[1lh] items-center gap-2">
      <span className="shrink-0">{pct}%</span>
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
    </div>
  );
}

const RequestRow = memo(function RequestRow({
  entry,
  selected,
  onSelect,
  now,
}: {
  entry: ApiMonitorEntry;
  selected: boolean;
  onSelect: (id: string) => void;
  now: number;
}): ReactElement {
  const age = (
    <span title={formatFullTime(entry.started_at)}>
      {formatAgo(entry.started_at, now)}
    </span>
  );
  const preview =
    entry.error ||
    entry.reply_preview ||
    entry.prompt_preview ||
    (entry.status === "running" ? "Waiting for output…" : "No preview");
  // A lifecycle row has no payload, so it reads as a status line.
  if (isLifecycleEntry(entry)) {
    return (
      <div className="flex w-full min-w-0 flex-col gap-0.5 rounded-lg px-3 py-2">
        <div className="flex min-w-0 items-center gap-2">
          <span
            className={cn(
              "size-1.5 shrink-0 rounded-full",
              statusDotClass(entry.status),
            )}
            aria-hidden={true}
          />
          <span className="truncate text-ui-12 text-muted-foreground">
            {lifecycleLabel(entry)}
          </span>
          <span className="ml-auto shrink-0 text-ui-11 tabular-nums text-muted-foreground">
            {age}
          </span>
        </div>
        <div className="min-w-0 truncate pl-3.5 text-ui-11 text-muted-foreground/80">
          {entry.model}
        </div>
        {entry.error ? (
          // Backend error text can quote the request, so keep it out too.
          <div
            data-reload-snapshot-sensitive
            className="min-w-0 break-words pl-3.5 text-ui-11 text-red-600 dark:text-red-400"
          >
            {entry.error}
          </div>
        ) : null}
      </div>
    );
  }
  const speed = formatTokPerSec(entry.tok_per_sec);
  return (
    <button
      type="button"
      data-entry-id={entry.id}
      onClick={() => onSelect(entry.id)}
      aria-current={selected}
      className={cn(
        "flex w-full min-w-0 flex-col gap-1 rounded-lg px-3 py-2.5 text-left outline-none transition-colors focus-visible:ring-1 focus-visible:ring-ring/70",
        selected ? "bg-muted" : "hover:bg-muted/50",
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
        <span className="truncate font-mono text-ui-12 font-medium text-foreground">
          {compactEndpoint(entry.endpoint)}
        </span>
        <span className="ml-auto shrink-0 text-ui-11 tabular-nums text-muted-foreground">
          {age}
        </span>
      </div>
      <div className="flex min-w-0 items-center gap-2 pl-4">
        <span className="truncate text-ui-11 text-muted-foreground">
          {entry.model}
        </span>
        <span className="ml-auto flex shrink-0 items-center gap-2.5 text-ui-11 tabular-nums text-muted-foreground">
          <span
            className="flex items-center gap-1"
            title={
              entry.prompt_tokens != null || entry.completion_tokens != null
                ? `Total time: queue wait, prompt processing and generation\nTokens: ${optionalCount(entry.prompt_tokens)} in, ${optionalCount(entry.completion_tokens)} out`
                : "Total time: queue wait, prompt processing and generation"
            }
          >
            <HugeiconsIcon
              icon={Timer01Icon}
              strokeWidth={1.75}
              className="size-3"
            />
            {entryDuration(entry)}
          </span>
          {speed ? (
            <span className="flex items-center gap-1" title="Generation speed">
              <HugeiconsIcon
                icon={FlashIcon}
                strokeWidth={1.75}
                className="size-3"
              />
              {speed}
            </span>
          ) : null}
        </span>
      </div>
      {/* A prompt or reply excerpt, same as the expanded payload below it. */}
      <p
        data-reload-snapshot-sensitive
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
});

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
  const [showAll, setShowAll] = useState(false);
  const preview = payloadPreview(body);
  const canShowAll =
    !truncated && preview !== body && body.length <= PAYLOAD_SHOW_ALL_MAX_CHARS;
  const shown = showAll && canShowAll ? body : preview;
  const [boxRef, singleLine] = useSingleLine<HTMLPreElement>();
  const copyable = Boolean(body) && !truncated;
  return (
    <section className="flex min-w-0 flex-col gap-1.5">
      <div className="flex items-center justify-between gap-2">
        <h3 className="text-ui-12 font-medium text-muted-foreground">
          {title}
        </h3>
        <div className="flex items-center gap-1">
          {canShowAll ? (
            <Button
              type="button"
              variant="ghost"
              size="xs"
              aria-pressed={showAll}
              onClick={() => setShowAll((value) => !value)}
              className="text-ui-10 text-muted-foreground"
            >
              {showAll ? "Show less" : "Show all"}
            </Button>
          ) : truncated || preview !== body ? (
            <span className="text-ui-10 text-muted-foreground">
              preview only
            </span>
          ) : null}
        </div>
      </div>
      <div className="relative min-w-0">
        {/* Prompt and reply bodies, so keep them out of the reload snapshot. */}
        <pre
          ref={boxRef}
          data-reload-snapshot-sensitive
          className={cn(
            "max-h-72 overflow-auto whitespace-pre-wrap break-words scroll-rounded p-3 text-ui-11 leading-[1.55]",
            singleLine ? "rounded-full" : "rounded-lg",
            copyable && "pr-11",
            tone === "error"
              ? "bg-red-500/5 text-red-700 dark:text-red-400"
              : INSET,
          )}
        >
          {loading && !body ? "Loading…" : shown || "–"}
        </pre>
        {copyable ? (
          <div
            className={cn(
              "absolute right-1.5",
              singleLine ? "top-1/2 -translate-y-1/2" : "top-1.5",
            )}
          >
            <CopyButton value={body} label={`Copy ${title.toLowerCase()}`} />
          </div>
        ) : null}
      </div>
    </section>
  );
}

function requestStatusLabel(entry: ApiMonitorEntry): string {
  if (entry.status !== "running") return entry.status;
  if (entry.running_phase === "token_generation") return "Token generation";
  if (entry.running_phase !== "prompt_processing") return "Running";
  const progress = entry.prompt_progress;
  if (progress?.percent != null) {
    return `Prompt processing · ${Math.round(progress.percent)}%`;
  }
  if (progress?.processed != null) {
    return `Prompt processing · ${formatCount(progress.processed)} tokens`;
  }
  return "Prompt processing";
}

function MetricGroup({
  title,
  items,
}: {
  title: string;
  items: { label: string; value: ReactNode }[];
}): ReactElement {
  return (
    <section className="flex min-w-0 flex-col gap-2">
      <h3 className="text-ui-12 font-medium text-muted-foreground">{title}</h3>
      <dl className="grid grid-cols-2 gap-px overflow-hidden rounded-xl border border-border/50 bg-border/50 sm:grid-cols-3 dark:border-transparent dark:bg-card">
        {items.map((item) => (
          <div
            key={item.label}
            className={cn(
              "flex min-w-0 flex-col gap-0.5 px-3 py-2",
              INSET_OPAQUE,
            )}
          >
            <dt className="truncate text-ui-11 text-muted-foreground">
              {item.label}
            </dt>
            <dd className="truncate text-ui-13 tabular-nums text-foreground">
              {item.value}
            </dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

function optionalCount(value?: number | null): string {
  return value != null ? formatCount(value) : "–";
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
      <header className="flex min-w-0 flex-col gap-1.5">
        <div className="flex min-w-0 items-center gap-2">
          <span
            className={cn(
              "inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-ui-11 font-medium capitalize",
              INSET,
              statusTextClass(entry.status),
            )}
          >
            <span
              className={cn(
                "size-1.5 shrink-0 rounded-full",
                statusDotClass(entry.status),
              )}
              aria-hidden={true}
            />
            {requestStatusLabel(entry)}
          </span>
          <span className="ml-auto truncate font-mono text-ui-10 text-muted-foreground">
            {entry.id}
          </span>
        </div>
        <h2 className="min-w-0 break-all font-mono text-ui-14 font-medium text-foreground">
          <span className="text-muted-foreground">{entry.method}</span>{" "}
          {entry.endpoint}
        </h2>
        <p className="min-w-0 break-all text-ui-12 text-muted-foreground">
          {entry.model}
        </p>
      </header>

      <MetricGroup
        title="Timing"
        items={[
          { label: "Started", value: formatTime(entry.started_at) },
          { label: "Total time", value: entryDuration(entry) },
          { label: "First token", value: formatDuration(entry.ttft_ms) },
          // Duration minus this is the queue wait, not slow decoding.
          { label: "Generating", value: formatDuration(entry.decode_ms) },
          {
            label: "Prompt speed",
            value: formatTokPerSec(entry.prompt_tok_per_sec) ?? "–",
          },
          {
            label: "Generation speed",
            value: formatTokPerSec(entry.tok_per_sec) ?? "–",
          },
        ]}
      />

      <MetricGroup
        title="Tokens"
        items={[
          {
            label: "Input (prompt)",
            value: optionalCount(entry.prompt_tokens),
          },
          {
            label: "Output (completion)",
            value: optionalCount(entry.completion_tokens),
          },
          { label: "Total", value: optionalCount(entry.total_tokens) },
          { label: "Context", value: optionalCount(entry.context_length) },
          {
            label: "Context used",
            value: <ContextUsageBar value={entry.context_usage} />,
          },
          { label: "Stop reason", value: entry.stop_reason ?? "–" },
        ]}
      />

      {entry.error ? (
        <PayloadBlock title="Error" body={entry.error} tone="error" />
      ) : null}

      <PayloadBlock
        key={`${entry.id}:prompt`}
        title="Prompt"
        body={prompt}
        truncated={entry.prompt_truncated && detail?.prompt == null}
        loading={loading}
      />
      <PayloadBlock
        key={`${entry.id}:reply`}
        title="Reply"
        body={reply}
        truncated={entry.reply_truncated && !detailIsCurrent}
        loading={loading}
      />
    </div>
  );
}

function QuickStart({
  baseUrl,
  model,
}: {
  baseUrl: string;
  model: string | null;
}): ReactElement {
  const snippet = quickStartSnippet(baseUrl, model ?? "MODEL_ID");
  return (
    <div className="flex flex-col items-center gap-4 px-6 py-12 text-center">
      <span
        className={cn(
          "flex size-10 items-center justify-center rounded-xl text-muted-foreground",
          INSET,
        )}
      >
        <HugeiconsIcon
          icon={ComputerTerminal01Icon}
          strokeWidth={1.75}
          className="size-5"
        />
      </span>
      <div className="flex max-w-md flex-col gap-1">
        <h3 className="text-ui-15 font-semibold text-foreground">
          No API traffic yet
        </h3>
        <p className="text-sm text-muted-foreground">
          {model
            ? "Send a request to the base URL and it appears here as it streams."
            : "Load a model, then send a request to the base URL. It appears here as it streams."}
        </p>
      </div>
      <div className="relative w-full max-w-2xl text-left">
        <pre
          className={cn(
            "overflow-x-auto scroll-rounded rounded-lg py-3 pr-11 pl-3.5 font-mono text-ui-11 leading-[1.6] text-foreground",
            INSET,
          )}
        >
          {snippet}
        </pre>
        <div className="absolute top-1.5 right-1.5">
          <CopyButton value={snippet} label="Copy example request" />
        </div>
      </div>
      <Button
        type="button"
        variant="outline"
        size="sm"
        onClick={openApiSettings}
        className="h-8 gap-1.5 rounded-full"
      >
        <HugeiconsIcon icon={Key01Icon} strokeWidth={1.75} className="size-4" />
        Create an API key and see more examples
      </Button>
    </div>
  );
}

export function ApiMonitorPage(): ReactElement {
  const signalReady = useAppShellReadySignal();
  const tour = useGuidedTourController({
    id: "api-monitor",
    steps: apiMonitorTourSteps,
  });
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
  const reloadReadySent = useRef(false);
  useEffect(() => {
    if (loading || reloadReadySent.current) {
      return;
    }
    reloadReadySent.current = true;
    signalReady();
  }, [loading, signalReady]);
  const serverUrl = usePlatformStore((s) => s.serverUrl);
  const cloudflareUrl = usePlatformStore((s) => s.cloudflareUrl);
  const lanUrls = usePlatformStore((s) => s.lanUrls);
  const runtime = useChatModelRuntime();
  const loadedModels = useChatRuntimeStore((s) => s.loadedModels);
  const residentCheckpoint = useChatRuntimeStore((s) => s.residentCheckpoint);
  const [unloading, setUnloading] = useState(false);
  const [ejectingKept, setEjectingKept] = useState(false);
  const [exampleModel, setExampleModel] = useState<ServedModel | null>(null);
  // block raw unloads while a picker load could finish afterward without a lifecycle guard.
  const modelLoading = useChatRuntimeStore((s) => s.modelLoading);
  const residencyKey = useMemo(
    () =>
      entries
        .filter(isLifecycleEntry)
        .map((entry) => `${entry.id}:${entry.status}`)
        .join(","),
    [entries],
  );
  const [unloadError, setUnloadError] = useState<string | null>(null);
  const resident = useResidentRunTarget(data?.active_model);
  const [configRequest, setConfigRequest] =
    useState<ModelConfigHandoffRequest | null>(null);
  const canEditSettings =
    !modelLoading && !unloading && residentHasRunSettings(resident);
  const canEditSlots =
    !modelLoading && !unloading && residentHasSlotsSetting(resident);

  useEffect(() => {
    const refreshRemoteBase = () => {
      void fetchDeviceType({ force: true });
      loadLanAccess()
        .then(publishLanUrls)
        .catch(() => undefined);
    };
    refreshRemoteBase();
    window.addEventListener("focus", refreshRemoteBase);
    return () => window.removeEventListener("focus", refreshRemoteBase);
  }, []);

  // read status for /unload's internal id; recheck because an API auto-switch can make it a no-op.
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
  const [modelFilter, setModelFilter] = useState<string>(ALL_MODELS);
  const [query, setQuery] = useState("");
  const deferredQuery = useDeferredValue(query);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const listRef = useRef<HTMLDivElement>(null);

  const focusRequestId = useApiMonitorOverlayStore((s) => s.focusRequestId);
  const scrollToSelected = useRef(false);
  useEffect(() => {
    if (focusRequestId == null) return;
    setSelectedId(focusRequestId);
    setQuery("");
    setStatusFilter("all");
    setModelFilter(ALL_MODELS);
    scrollToSelected.current = true;
    useApiMonitorOverlayStore.getState().focusRequest(null);
  }, [focusRequestId]);

  const loggedModels = useMemo(() => {
    const names = new Set<string>();
    for (const entry of entries) {
      if (!isLifecycleEntry(entry) && entry.model) names.add(entry.model);
    }
    return [...names].sort();
  }, [entries]);
  const effectiveModelFilter =
    modelFilter !== ALL_MODELS && loggedModels.includes(modelFilter)
      ? modelFilter
      : ALL_MODELS;
  const modelEntries = useMemo(
    () =>
      effectiveModelFilter === ALL_MODELS
        ? entries
        : entries.filter((entry) => entry.model === effectiveModelFilter),
    [entries, effectiveModelFilter],
  );
  const modelStats = useMemo(
    () =>
      effectiveModelFilter === ALL_MODELS ? stats : computeStats(modelEntries),
    [effectiveModelFilter, stats, modelEntries],
  );
  const visible = useMemo(
    () => filterEntries(modelEntries, statusFilter, deferredQuery),
    [modelEntries, statusFilter, deferredQuery],
  );
  const selected = useMemo(
    () => visible.find((entry) => entry.id === selectedId) ?? null,
    [visible, selectedId],
  );
  const statusCounts = useMemo(
    () => countByStatus(modelEntries),
    [modelEntries],
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

  useEffect(() => {
    if (!scrollToSelected.current || selectedId == null) return;
    const row = listRef.current?.querySelector<HTMLElement>(
      `[data-entry-id="${CSS.escape(selectedId)}"]`,
    );
    if (!row) return;
    scrollToSelected.current = false;
    row.scrollIntoView({ block: "nearest" });
  }, [selectedId, visible]);

  const onListKeyDown = (event: KeyboardEvent<HTMLDivElement>): void => {
    if (event.key !== "ArrowDown" && event.key !== "ArrowUp") {
      return;
    }
    const selectable = visible.filter((entry) => !isLifecycleEntry(entry));
    if (selectable.length === 0) {
      return;
    }
    event.preventDefault();
    const index = selectable.findIndex((entry) => entry.id === selectedId);
    const step = event.key === "ArrowDown" ? 1 : -1;
    const next =
      selectable[
        index === -1
          ? 0
          : Math.max(0, Math.min(selectable.length - 1, index + step))
      ];
    setSelectedId(next.id);
    listRef.current
      ?.querySelector<HTMLElement>(`[data-entry-id="${CSS.escape(next.id)}"]`)
      ?.focus();
  };

  // Tauri uses the runtime API base because its tauri:// origin omits the dynamic port.
  const origin = typeof window === "undefined" ? "" : window.location.origin;
  const localOrigin = isTauri ? (serverUrl ?? getApiBase()) : origin;
  const baseUrl = `${remoteApiOrigin(cloudflareUrl, localOrigin, lanUrls)}/v1`;
  const activeModel = data?.active_model ?? null;
  const now = useServerNow(data?.server_time, entries.length > 0);
  const served = useMemo(
    () => servedModels(loadedModels, activeModel, residentCheckpoint),
    [loadedModels, activeModel, residentCheckpoint],
  );
  const serverStatus =
    SERVER_STATUS[
      data?.status === "idle" && served.length > 0
        ? "ready"
        : (data?.status ?? "idle")
    ];
  // older backends omit the field, so only explicit `false` disables recording.
  const loggingDisabled = data?.logging_enabled === false;
  const queued = data?.queue?.queued ?? 0;
  const slotsLabel = data?.queue
    ? `${data.queue.active}/${data.queue.capacity} busy${
        queued > 0 ? ` · ${queued} queued` : ""
      }`
    : "";
  const slotsClassName = cn(
    "tabular-nums",
    queued > 0 && "text-amber-700 dark:text-amber-500",
  );

  const ejecting = unloading || ejectingKept;
  const ejectServed = async (model: ServedModel): Promise<void> => {
    if (model.active) {
      await unloadActiveModel();
      return;
    }
    setEjectingKept(true);
    try {
      await runtime.ejectModel(model.checkpoint);
      refresh();
    } finally {
      setEjectingKept(false);
    }
  };
  const ejectAll = async (): Promise<void> => {
    setEjectingKept(true);
    try {
      await runtime.ejectAllModels();
      refresh();
    } finally {
      setEjectingKept(false);
    }
  };
  const openRunSettings = (model: ServedModel): void => {
    const requestId = createModelConfigHandoffRequestId();
    setConfigRequest(
      model.active
        ? runSettingsRequest(resident, requestId)
        : keptRunSettingsRequest(model, requestId),
    );
  };
  const settingsButton = (model: ServedModel): ReactElement => (
    <ToolbarTooltip label={`Run settings for ${model.name}`}>
      <Button
        type="button"
        variant="ghost"
        size="sm"
        aria-label={`Run settings for ${model.name}`}
        aria-haspopup="dialog"
        onClick={() => openRunSettings(model)}
        // Primary opens only on live settings: the kept-model fallback would send Run its bare label.
        disabled={
          ejecting || modelLoading || (model.active && !canEditSettings)
        }
        className="size-7 shrink-0 rounded-full p-0 text-muted-foreground hover:text-foreground"
      >
        <HugeiconsIcon
          icon={Settings02Icon}
          strokeWidth={1.75}
          className="size-3.5"
        />
      </Button>
    </ToolbarTooltip>
  );
  const exampleButton = (model: ServedModel): ReactElement => (
    <ToolbarTooltip label={`API example for ${model.name}`}>
      <Button
        type="button"
        variant="ghost"
        size="sm"
        aria-label={`API example for ${model.name}`}
        aria-haspopup="dialog"
        onClick={() => setExampleModel(model)}
        className="size-7 shrink-0 rounded-full p-0 text-muted-foreground hover:text-foreground"
      >
        <HugeiconsIcon
          icon={CodeIcon}
          strokeWidth={1.75}
          className="size-3.5"
        />
      </Button>
    </ToolbarTooltip>
  );
  const ejectLabel =
    served.length > 1
      ? `Eject all ${served.length} models and free their memory`
      : served.length === 1
        ? `Eject ${served[0].name} and free its memory`
        : "No model is loaded";
  const contextFact = (
    <>
      {data?.context_length ? (
        <Fact label="Context">
          <span className="tabular-nums">
            {formatCount(data.context_length)} tokens
          </span>
        </Fact>
      ) : null}
    </>
  );
  const slotsFact = (
    <>
      {data?.queue ? (
        <Fact label="Slots">
          {canEditSlots ? (
            <ToolbarTooltip label="Change parallel slots">
              <button
                type="button"
                onClick={() =>
                  setConfigRequest(
                    slotsSettingsRequest(
                      resident,
                      createModelConfigHandoffRequestId(),
                    ),
                  )
                }
                className={cn(
                  slotsClassName,
                  "w-fit cursor-pointer text-left underline decoration-dotted decoration-muted-foreground/60 underline-offset-4 hover:decoration-foreground",
                )}
              >
                {slotsLabel}
              </button>
            </ToolbarTooltip>
          ) : (
            <span className={slotsClassName}>{slotsLabel}</span>
          )}
        </Fact>
      ) : null}
    </>
  );
  const activeFacts = (
    <>
      {contextFact}
      {slotsFact}
    </>
  );

  return (
    <main className="api-rounding mx-auto flex w-full max-w-6xl 3xl:max-w-[calc(1440px*var(--ui-space-scale,1))] 4xl:max-w-[calc(1760px*var(--ui-space-scale,1))] flex-col gap-5 px-6 pb-10 pt-12 max-sm:px-4 max-sm:pt-8 font-heading sm:px-10">
      <GuidedTour {...tour.tourProps} />
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div className="flex min-w-0 flex-col gap-1">
          <div className="flex flex-wrap items-center gap-3">
            <h1 className="text-ui-30 font-semibold leading-[1.04] tracking-[-0.028em] text-foreground sm:text-ui-34">
              API
            </h1>
            <span
              className={cn(
                "inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-ui-12 font-medium text-foreground",
                INSET,
              )}
            >
              <span
                className={cn("size-2 rounded-full", serverStatus.dot)}
                aria-hidden={true}
              />
              {serverStatus.label}
              {served.length > 1 ? (
                <span className="text-muted-foreground">
                  · {served.length} models
                </span>
              ) : null}
            </span>
          </div>
          <p className="text-sm text-muted-foreground">
            Serve your local models to any OpenAI-compatible client and watch
            each request live.
          </p>
        </div>
        <ToolbarTooltip label="API settings">
          <Button
            type="button"
            variant="ghost"
            size="icon"
            onClick={openApiSettings}
            aria-label="API settings"
            className="size-10 rounded-full text-muted-foreground hover:bg-[color-mix(in_oklab,var(--foreground)_calc(10%*var(--contrast-wash-gain,1)),transparent)] hover:text-foreground dark:hover:bg-[color-mix(in_oklab,var(--foreground)_calc(14%*var(--contrast-wash-gain,1)),transparent)]"
          >
            <HugeiconsIcon
              icon={Settings02Icon}
              strokeWidth={1.75}
              className="size-5"
            />
          </Button>
        </ToolbarTooltip>
      </header>

      {/* Checked first when a client can't reach the API: base URL and what is loaded. */}
      <section
        data-tour="api-endpoint"
        className={cn("flex flex-col", SURFACE)}
      >
        <div className="flex flex-wrap items-center justify-between gap-x-6 gap-y-3 px-5 py-4">
          <div className="flex min-w-0 items-center gap-3">
            <span
              className="train-section-chip inline-flex size-9 shrink-0 items-center justify-center rounded-full"
              style={
                { "--chip-tint": "var(--control-accent)" } as CSSProperties
              }
            >
              <HugeiconsIcon
                icon={ApiIcon}
                strokeWidth={1.5}
                className="size-[calc(18px*var(--ui-space-scale,1))]"
              />
            </span>
            <div className="flex min-w-0 flex-col">
              <span className="text-ui-11 text-muted-foreground">Base URL</span>
              <div className="group/copy flex min-w-0 items-center gap-1">
                <span className="truncate font-mono text-ui-13 text-foreground">
                  {baseUrl}
                </span>
                <CopyButton
                  value={baseUrl}
                  label="Copy API base URL"
                  revealOnHover={true}
                />
              </div>
            </div>
          </div>
          <div data-tour="api-toolbar" className="flex min-w-0 max-w-full">
            <ApiModelLoadControls
              runtime={runtime}
              activeModel={data?.active_model}
              residencyKey={residencyKey}
              onSettled={refresh}
              onUnloadActive={unloadActiveModel}
              unloading={ejecting}
              configRequest={configRequest}
              onConfigRequestAdopted={() => setConfigRequest(null)}
              onEjectAll={() => void ejectAll()}
            >
              <ToolbarTooltip label={ejectLabel}>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() =>
                    void (served.length > 1
                      ? ejectAll()
                      : ejectServed(served[0]))
                  }
                  disabled={ejecting || modelLoading || served.length === 0}
                  className="h-9 gap-1.5 rounded-full px-3.5"
                >
                  <HugeiconsIcon
                    icon={RemoveCircleIcon}
                    strokeWidth={1.75}
                    className={cn("size-4", ejecting && "animate-pulse")}
                  />
                  {ejecting
                    ? "Ejecting"
                    : served.length > 1
                      ? "Eject all"
                      : "Eject"}
                </Button>
              </ToolbarTooltip>
            </ApiModelLoadControls>
          </div>
        </div>
        <div className={cn("border-t px-5 pt-3.5 pb-5", DIVIDER)}>
          {served.length > 1 ? (
            <div className="flex flex-col gap-2">
              <p className="text-ui-12 text-muted-foreground">
                {served.length} models answering requests
              </p>
              <div className="grid grid-cols-[minmax(0,max-content)_auto_auto_auto_1fr] items-center gap-x-6 gap-y-1.5">
                {served.map((model) => (
                  <div key={model.checkpoint} className="contents">
                    <span className="group/copy flex min-w-0 items-center gap-1 text-ui-12">
                      <span
                        className="size-1.5 shrink-0 rounded-full bg-emerald-500"
                        aria-hidden={true}
                      />
                      <span className="ml-1 truncate font-mono text-foreground">
                        {model.name}
                      </span>
                      <CopyButton
                        value={model.name}
                        label="Copy model name"
                        revealOnHover={true}
                      />
                    </span>
                    <span className="flex items-center gap-0.5">
                      {settingsButton(model)}
                      {exampleButton(model)}
                    </span>
                    <dl className="min-w-0">
                      {model.active ? contextFact : null}
                    </dl>
                    <dl className="min-w-0">
                      {model.active ? slotsFact : null}
                    </dl>
                    <span className="justify-self-end">
                      <ToolbarTooltip label={`Eject ${model.name}`}>
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          aria-label={`Eject ${model.name}`}
                          onClick={() => void ejectServed(model)}
                          disabled={ejecting || modelLoading}
                          className="size-7 rounded-full p-0 text-muted-foreground hover:text-foreground"
                        >
                          <HugeiconsIcon
                            icon={RemoveCircleIcon}
                            strokeWidth={1.75}
                            className="size-3.5"
                          />
                        </Button>
                      </ToolbarTooltip>
                    </span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <dl className="flex flex-wrap items-center gap-x-6 gap-y-2">
              <Fact label="Model" className="group/copy max-w-full">
                {served.length === 1 ? (
                  <>
                    <span className="truncate font-mono">{served[0].name}</span>
                    <CopyButton
                      value={served[0].name}
                      label="Copy model name"
                      revealOnHover={true}
                    />
                    {settingsButton(served[0])}
                    {exampleButton(served[0])}
                  </>
                ) : (
                  <span className="text-muted-foreground">None loaded</span>
                )}
              </Fact>
              {served[0]?.active ? activeFacts : null}
            </dl>
          )}
        </div>
      </section>

      <VideoGenerationApiDocs baseUrl={baseUrl} />

      {error ? <Banner>{error}</Banner> : null}
      {unloadError ? (
        <Banner onDismiss={() => setUnloadError(null)}>{unloadError}</Banner>
      ) : null}

      <section
        aria-label="Traffic summary"
        className="elevated-card grid grid-cols-2 gap-px overflow-hidden bg-border/60 sm:grid-cols-3 lg:grid-cols-6 dark:bg-background"
      >
        <Stat
          label="In flight"
          value={formatCount(modelStats.active)}
          tone={modelStats.active > 0 ? "active" : "default"}
          hint={queued > 0 ? `${formatCount(queued)} queued` : undefined}
        />
        <Stat
          label="Requests"
          value={formatCount(modelStats.total)}
          hint={loggingDisabled ? "recording off" : "in the recent log"}
        />
        <Stat
          label="Errors"
          value={formatCount(modelStats.errors)}
          tone={modelStats.errors > 0 ? "error" : "default"}
          hint={
            modelStats.errorRate != null
              ? `${Math.round(modelStats.errorRate * 100)}% of finished`
              : undefined
          }
        />
        <Stat
          label="Avg latency"
          value={formatDuration(modelStats.avgDurationMs)}
          hint={
            modelStats.maxDurationMs != null
              ? `max ${formatDuration(modelStats.maxDurationMs)}`
              : undefined
          }
        />
        <Stat
          label="Generation speed"
          value={
            modelStats.tokensPerSecond == null
              ? "–"
              : `${modelStats.tokensPerSecond.toFixed(1)} tok/s`
          }
          hint="while decoding"
        />
        <Stat
          label="Tokens"
          value={formatCount(modelStats.totalTokens)}
          hint={`${formatCount(modelStats.inputTokens)} in · ${formatCount(modelStats.outputTokens)} out`}
        />
      </section>

      <section className={cn("flex min-h-0 flex-col overflow-hidden", SURFACE)}>
        <div
          className={cn(
            "flex flex-wrap items-center gap-2 border-b px-4 py-3",
            DIVIDER,
          )}
        >
          <div className="relative w-full min-w-0 sm:w-56">
            <HugeiconsIcon
              icon={Search01Icon}
              strokeWidth={1.75}
              className="pointer-events-none absolute top-1/2 left-3 size-3.5 -translate-y-1/2 text-muted-foreground"
              aria-hidden={true}
            />
            <Input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search requests"
              aria-label="Search API requests"
              title="Matches the model, endpoint, prompt, reply or error"
              className="pl-8"
            />
          </div>
          <nav
            aria-label="Filter by status"
            className="-my-2 flex max-w-full gap-1 overflow-x-auto px-1 py-2 [scrollbar-width:none]!"
          >
            {STATUS_FILTERS.map((option) => {
              const active = statusFilter === option.value;
              const count = statusCounts[option.value];
              return (
                <button
                  key={option.value}
                  type="button"
                  aria-current={active ? "page" : undefined}
                  onClick={() => setStatusFilter(option.value)}
                  className={cn(
                    "flex h-9 shrink-0 items-center gap-1.5 rounded-full px-3.5 text-ui-14 text-foreground/80 transition-colors hover:text-foreground",
                    active &&
                      "bg-white font-medium text-foreground shadow-[0_2px_8px_-2px_rgba(0,0,0,0.16)] dark:bg-[rgb(255_255_255_/_calc(0.1*var(--contrast-wash-gain,1)))] dark:shadow-none",
                  )}
                >
                  {option.label}
                  {option.value !== "all" && count > 0 ? (
                    <span
                      className={cn(
                        "tabular-nums text-muted-foreground",
                        option.value === "error" &&
                          "text-red-600 dark:text-red-400",
                      )}
                    >
                      {formatCount(count)}
                    </span>
                  ) : null}
                </button>
              );
            })}
          </nav>
          {loggedModels.length > 1 ? (
            <Select value={effectiveModelFilter} onValueChange={setModelFilter}>
              <SelectTrigger
                aria-label="Filter by model"
                className="h-9 max-w-64 rounded-full"
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value={ALL_MODELS}>All models</SelectItem>
                {loggedModels.map((model) => (
                  <SelectItem key={model} value={model}>
                    {model}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          ) : null}
          <div className="ml-auto flex shrink-0 items-center gap-1">
            {visible.length !== entries.length ? (
              <span className="mr-1 text-ui-11 tabular-nums text-muted-foreground">
                {formatCount(visible.length)} of {formatCount(entries.length)}
              </span>
            ) : null}
            <ToolbarTooltip
              label={paused ? "Resume the live feed" : "Pause the live feed"}
            >
              <Button
                type="button"
                variant="ghost"
                size="icon"
                aria-label={
                  paused ? "Resume the live feed" : "Pause the live feed"
                }
                aria-pressed={paused}
                onClick={() => setPaused(!paused)}
                className={cn(
                  "rounded-full",
                  paused &&
                    "bg-amber-500/10 text-amber-700 hover:bg-amber-500/15 dark:text-amber-500",
                )}
              >
                <HugeiconsIcon
                  icon={paused ? PlayIcon : PauseIcon}
                  strokeWidth={1.75}
                  className="size-4"
                />
              </Button>
            </ToolbarTooltip>
            <ToolbarTooltip label="Refresh">
              <Button
                type="button"
                variant="ghost"
                size="icon"
                onClick={refresh}
                disabled={refreshing}
                aria-label="Refresh"
                className="rounded-full"
              >
                <HugeiconsIcon
                  icon={Refresh01Icon}
                  strokeWidth={1.75}
                  className={cn("size-4", refreshing && "animate-spin")}
                />
              </Button>
            </ToolbarTooltip>
            <ToolbarTooltip label="Clear log">
              <Button
                type="button"
                variant="ghost"
                size="icon"
                disabled={entries.length === 0}
                onClick={() => {
                  setSelectedId(null);
                  void clear();
                }}
                aria-label="Clear log"
                className="rounded-full"
              >
                <HugeiconsIcon
                  icon={Delete02Icon}
                  strokeWidth={1.75}
                  className="size-4"
                />
              </Button>
            </ToolbarTooltip>
          </div>
        </div>

        {!loading && entries.length === 0 ? (
          <div data-tour="api-log">
            {loggingDisabled ? (
              <p className="mx-auto max-w-lg px-6 py-14 text-center text-sm text-muted-foreground">
                Recording is off: UNSLOTH_STUDIO_DISABLE_API_MONITOR is set.
                Requests and model loads still run normally, they are just not
                listed here. Unset the variable and restart Unsloth to
                re-enable.
              </p>
            ) : (
              <QuickStart
                baseUrl={baseUrl}
                model={
                  (served.find((m) => m.active) ?? served[0])?.name ?? null
                }
              />
            )}
          </div>
        ) : (
          <div
            data-tour="api-log"
            className="grid min-h-0 grid-cols-1 lg:grid-cols-[minmax(0,380px)_minmax(0,1fr)] 4xl:grid-cols-[minmax(0,480px)_minmax(0,1fr)]"
          >
            <div
              ref={listRef}
              onKeyDown={onListKeyDown}
              className={cn(
                PANE_HEIGHT,
                DIVIDER,
                "flex flex-col gap-0.5 border-b p-1.5 lg:border-b-0 lg:border-r",
              )}
            >
              {loading ? (
                <div className="flex flex-col gap-2 p-1.5">
                  {[0, 1, 2].map((i) => (
                    <Skeleton key={i} className="h-16 w-full rounded-lg" />
                  ))}
                </div>
              ) : visible.length === 0 ? (
                <div className="flex flex-col items-center gap-3 px-4 py-10 text-center text-sm text-muted-foreground">
                  No requests match this filter.
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      setQuery("");
                      setStatusFilter("all");
                      setModelFilter(ALL_MODELS);
                    }}
                    className="h-8 rounded-full"
                  >
                    Clear filters
                  </Button>
                </div>
              ) : (
                visible.map((entry) => (
                  <RequestRow
                    key={entry.id}
                    entry={entry}
                    selected={entry.id === selectedId}
                    onSelect={setSelectedId}
                    now={now}
                  />
                ))
              )}
            </div>

            <div className={PANE_HEIGHT}>
              {selected ? (
                <RequestDetail
                  entry={selected}
                  detail={details[selected.id]}
                  loading={loadingDetails.has(selected.id)}
                />
              ) : (
                <div className="flex h-full flex-col items-center justify-center gap-3 px-6 py-10 text-center">
                  <span
                    className={cn(
                      "flex size-10 items-center justify-center rounded-xl text-muted-foreground",
                      INSET,
                    )}
                  >
                    <HugeiconsIcon
                      icon={CursorPointer02Icon}
                      strokeWidth={1.75}
                      className="size-5"
                    />
                  </span>
                  <p className="max-w-xs text-sm text-muted-foreground">
                    Select a request to inspect its prompt, reply, tokens and
                    errors. Use the arrow keys to step through the list.
                  </p>
                </div>
              )}
            </div>
          </div>
        )}
      </section>

      <SavedModelSettingsPanel />
      <ApiExampleDialog
        open={exampleModel != null}
        onOpenChange={(open) => {
          if (!open) setExampleModel(null);
        }}
        baseUrl={baseUrl}
        model={exampleModel?.name ?? ""}
        samplingModelId={exampleModel?.checkpoint ?? ""}
      />
    </main>
  );
}
