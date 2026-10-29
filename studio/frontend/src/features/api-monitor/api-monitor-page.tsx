// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useAppShellReadySignal } from "@/components/app-readiness";
import { GuidedTour, useGuidedTourController } from "@/features/tour";
import { apiMonitorTourSteps } from "./tour";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useSidebar } from "@/components/ui/sidebar";
import { Skeleton } from "@/components/ui/skeleton";
import { fetchDeviceType, usePlatformStore } from "@/config/env";
import { useIsMobileShell } from "@/hooks/use-mobile";
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
import { useIsAccountOwner } from "@/features/auth";
import {
  loadLanAccess,
  loadMultiModelEnabled,
  publishLanUrls,
  useSettingsDialogStore,
} from "@/features/settings";
// Not through the barrel: these pull in chat, which every barrel importer would then load.
import {
  type ApiExampleKind,
  ApiExampleTabNav,
} from "@/features/settings/components/api-example-tabs";
import { AudioApiExamples } from "@/features/settings/components/audio-api-examples";
import { DecisionApiSection } from "@/features/settings/components/decision-api-section";
import { MediaApiExamples } from "@/features/settings/components/media-api-examples";
import { useApiTunnelPref } from "@/features/settings/components/usage-examples";
import { remoteApiOrigin } from "@/features/settings/api/remote-access-state";
import { getApiBase, isTauri } from "@/lib/api-base";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { Tick02Icon } from "@/lib/tick-icon";
import { cn } from "@/lib/utils";
import {
  ApiIcon,
  ArrowDataTransferVerticalIcon,
  Cancel01Icon,
  Clock01Icon,
  CodeIcon,
  ComputerTerminal01Icon,
  Delete02Icon,
  FileSearchIcon,
  PlusSignIcon,
  Refresh01Icon,
  RemoveCircleIcon,
  Search01Icon,
  Settings02Icon,
} from "@hugeicons/core-free-icons";
import { Copy01Icon } from "@/lib/copy-icon";
import { PauseIcon } from "@/lib/media-control-icons";
import { ReloadIcon } from "@/lib/reload-icon";
import { SpeedMeterIcon } from "@/lib/hugeicons-derived";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
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
import { buildApiExample } from "./api-sampling-example";
import {
  ApiExampleBox,
  useApiExampleChoices,
} from "./components/api-example-box";
import { ApiExampleDialog } from "./components/api-example-dialog";
import { ApiModelLoadControls } from "./components/api-model-load-controls";
import { SavedModelSettingsPanel } from "./components/saved-model-settings";
import { isLifecycleEntry, lifecycleLabel } from "./lifecycle";
import { useApiMonitorOverlayStore } from "./overlay-store";
import { useSingleLine } from "./use-single-line";
import { PAYLOAD_SHOW_ALL_MAX_CHARS, payloadPreview } from "./payload-preview";
import { ToolbarTooltip } from "./components/toolbar-tooltip";
import { unloadResident } from "./unload-resident";
import { useApiModelReload } from "./use-api-model-reload";
import { ContextUsageBar as ContextUsageChip } from "@/features/chat/components/context-usage-bar";
import {
  residentHasRunSettings,
  keptRunSettingsRequest,
  runSettingsRequest,
} from "./slots-settings-request";
import {
  monitorSelection,
  useResidentFileGrant,
  useResidentRunTarget,
} from "./use-resident-run-target";
import {
  type MonitorStatusFilter,
  computeStats,
  countByStatus,
  filterEntries,
  useApiMonitor,
} from "./use-api-monitor";
import { PlayIcon } from "lucide-react";
import { useFreshApiKeyStore } from "@/features/settings/stores/fresh-api-key-store";

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
  /** The id the monitor logs under; the label can carry an MLX folder suffix. */
  id: string;
  checkpoint: string;
  active: boolean;
  quant: string | null;
  isMlx: boolean;
};

function servedModels(
  loaded: { id: string; quant?: string | null; checkpoint?: string }[],
  activeModel: string | null,
  residentCheckpoint: string | null | undefined,
  // Status says MLX only for the primary model; kept slots carry no backend.
  loadedIsMlx: boolean | null,
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
    id: monitorSelection(model.id).id,
    checkpoint: model.checkpoint ?? model.id,
    active: index === activeIndex,
    quant: model.quant ?? null,
    isMlx: index === activeIndex && loadedIsMlx === true,
  }));
  if (activeModel && !rows.some((row) => row.active)) {
    rows.unshift({
      name: activeModel,
      id: activeId,
      checkpoint: activeModel,
      active: true,
      quant: monitorSelection(activeModel).ggufVariant,
      isMlx: loadedIsMlx === true,
    });
  }
  return rows.sort((a, b) => Number(b.active) - Number(a.active));
}

function openApiSettings(): void {
  useSettingsDialogStore.getState().openDialog("api-keys");
}

function MultiModelHint({
  servedCount,
}: {
  servedCount: number;
}): ReactElement | null {
  const keepLoaded = useChatRuntimeStore((s) => s.keepModelsLoaded);
  const isOwner = useIsAccountOwner();
  // Store defaults to off: without the read the hint flashes for users who have it on.
  const [known, setKnown] = useState(false);
  useEffect(() => {
    let cancelled = false;
    loadMultiModelEnabled("Failed to read the multiple models setting").then(
      (enabled) => {
        if (cancelled) return;
        useChatRuntimeStore.getState().setKeepModelsLoaded(enabled);
        setKnown(true);
      },
      () => undefined,
    );
    return () => {
      cancelled = true;
    };
  }, []);
  if (!known || keepLoaded || servedCount > 1) return null;
  return (
    <>
      {" · "}
      {isOwner ? (
        <>
          Enable loading multiple models in{" "}
          <button
            type="button"
            className="font-medium text-foreground underline underline-offset-2 hover:text-primary"
            onClick={() =>
              useSettingsDialogStore.getState().openDialog("resources", {
                scrollTarget: "resources-model-memory",
              })
            }
          >
            Settings
          </button>
        </>
      ) : (
        "One model at a time"
      )}
    </>
  );
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
          "truncate font-heading text-ui-21 font-semibold leading-tight tracking-[-0.02em] tabular-nums text-foreground",
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

const FACT_CELL =
  "order-2 min-w-0 @max-3xl/served:pl-3.5 @max-3xl/served:empty:hidden @3xl/served:order-none";

function Fact({
  label,
  icon,
  children,
  className,
}: {
  label: string;
  icon?: IconSvgElement;
  children: ReactNode;
  className?: string;
}): ReactElement {
  return (
    <div className={cn("flex min-w-0 items-center", className)} title={label}>
      <dt className="sr-only">{label}</dt>
      <dd className="flex min-w-0 items-center gap-1.5 whitespace-nowrap text-ui-12 text-foreground">
        {icon ? (
          <HugeiconsIcon
            icon={icon}
            strokeWidth={1.75}
            className="size-[1em] shrink-0 text-muted-foreground"
          />
        ) : null}
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
  if (isLifecycleEntry(entry)) {
    return (
      <div className="flex w-full min-w-0 flex-col gap-0.5 rounded-md px-3 py-2">
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
        "flex w-full min-w-0 flex-col gap-1 rounded-md px-3 py-2.5 text-left outline-none transition-colors focus-visible:ring-1 focus-visible:ring-ring/70",
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
              icon={Clock01Icon}
              strokeWidth={1.75}
              className="size-3"
            />
            {entryDuration(entry)}
          </span>
          {speed ? (
            <span className="flex items-center gap-1" title="Generation speed">
              <HugeiconsIcon
                icon={SpeedMeterIcon}
                strokeWidth={1.75}
                className="size-3"
              />
              {speed}
            </span>
          ) : null}
        </span>
      </div>
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
  // Prefer the detail only once it is as fresh as the list row, or the panel rewinds.
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
  const choices = useApiExampleChoices(baseUrl, "curl");
  const [useTunnel] = useApiTunnelPref();
  const [kind, setKind] = useState<ApiExampleKind>("chat");
  const freshKey = useFreshApiKeyStore((s) => s.key);
  return (
    <div className="flex flex-col items-center gap-4 px-6 py-12 text-center">
      <span className="flex size-10 items-center justify-center icon-disc text-muted-foreground">
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
            ? "Send a request to the base URL to see it here."
            : "Load a model, then send a request to see it here."}
        </p>
      </div>
      <ApiExampleTabNav kind={kind} onChange={setKind} className="justify-center" />
      {kind === "chat" ? (
        <div className="w-full max-w-2xl text-left">
          <ApiExampleBox
            choices={choices}
            hasRealKey={freshKey !== null}
            code={buildApiExample({
              baseUrl,
              model: model ?? "MODEL_ID",
              sampling: {},
              apiKey: freshKey ?? undefined,
              language: choices.language,
              os: choices.os,
            })}
          />
        </div>
      ) : (
        <div className="w-full max-w-3xl text-left">
          {kind === "media" ? (
            <MediaApiExamples
              apiKey={freshKey}
              useTunnel={useTunnel}
              showHeading={false}
            />
          ) : kind === "audio" ? (
            <AudioApiExamples
              apiKey={freshKey}
              useTunnel={useTunnel}
              showHeading={false}
            />
          ) : (
            <DecisionApiSection showHeading={false} />
          )}
        </div>
      )}
      <Button
        type="button"
        variant="outline"
        size="sm"
        onClick={openApiSettings}
        className="gap-1.5 rounded-full"
      >
        <HugeiconsIcon
          icon={PlusSignIcon}
          strokeWidth={1.75}
          className="size-4"
        />
        Create an API key and see more examples
      </Button>
    </div>
  );
}

export function ApiMonitorPage(): ReactElement {
  const signalReady = useAppShellReadySignal();
  const isMobileShell = useIsMobileShell();
  const { pinned } = useSidebar();
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
  const loadedIsMlx = useChatRuntimeStore((s) => s.loadedIsMlx);
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
  const resident = useResidentRunTarget(data?.active_model);
  const fileGrant = useResidentFileGrant(data?.active_model);
  const reload = useApiModelReload({
    selectModel: runtime.selectModel,
    loadNpuModel: runtime.loadNpuModel,
    readModelLoadEpochs: runtime.readModelLoadEpochs,
    fileGrant,
    onSettled: refresh,
    residencyKey,
  });
  const [unloadError, setUnloadError] = useState<string | null>(null);
  const [configRequest, setConfigRequest] =
    useState<ModelConfigHandoffRequest | null>(null);
  const canEditSettings =
    !modelLoading && !unloading && residentHasRunSettings(resident);

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
          // Both spellings: status reports the load path, the store may hold the repo id.
          return {
            checkpoint,
            aliases: [checkpoint, status.active_model].filter(
              (alias): alias is string => alias != null,
            ),
          };
        },
        unload: (checkpoint) => unloadModel({ model_path: checkpoint }),
      });
      // Only clear when the store holds the model just unloaded: an external provider may be selected.
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

  // Keyed on identity and revision, never `details`, which the fetch rewrites (loop).
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
    const revision = `${selectedId_}@${selectedUpdatedAt ?? ""}`;
    if (!selectedIsMissing && lastFetchedRef.current === revision) {
      return;
    }
    // A terminal row's revision never advances; `loadingDetails` settling re-runs this, bounded by count.
    if (attemptsRef.current.revision !== revision) {
      attemptsRef.current = { revision, count: 0 };
    }
    if (attemptsRef.current.count >= DETAIL_FETCH_ATTEMPTS) {
      return;
    }
    attemptsRef.current.count += 1;
    // Only when a fetch started; recording a refused one skips that revision for good.
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
    () =>
      servedModels(loadedModels, activeModel, residentCheckpoint, loadedIsMlx),
    [loadedModels, activeModel, residentCheckpoint, loadedIsMlx],
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
        className="size-7 shrink-0 rounded-full p-0 text-muted-foreground hover:text-foreground dark:hover:bg-chat-icon-bg-hover"
      >
        <HugeiconsIcon
          icon={Settings02Icon}
          strokeWidth={1.75}
          className="size-3.5"
        />
      </Button>
    </ToolbarTooltip>
  );
  const reloadBusy = reload.reloadingKey != null || ejecting || modelLoading;
  const reloadButton = (model: ServedModel): ReactElement => (
    <ToolbarTooltip label={`Reload ${model.name}`}>
      <Button
        type="button"
        variant="ghost"
        size="sm"
        aria-label={`Reload ${model.name}`}
        onClick={() => void reload.reloadServed(model)}
        disabled={reloadBusy || !reload.canReload(model)}
        className="size-7 shrink-0 rounded-full p-0 text-muted-foreground hover:text-foreground dark:hover:bg-chat-icon-bg-hover"
      >
        <HugeiconsIcon
          icon={ReloadIcon}
          strokeWidth={1.75}
          className={cn(
            "size-3.5",
            (reload.reloadingKey === model.checkpoint ||
              reload.reloadingKey === "*") &&
              "animate-spin",
          )}
        />
      </Button>
    </ToolbarTooltip>
  );
  const ejectButton = (model: ServedModel): ReactElement => (
    <ToolbarTooltip label={`Eject ${model.name}`}>
      <Button
        type="button"
        variant="ghost"
        size="sm"
        aria-label={`Eject ${model.name}`}
        onClick={() => void ejectServed(model)}
        disabled={ejecting || modelLoading}
        className="size-7 shrink-0 rounded-full p-0 text-muted-foreground hover:text-foreground dark:hover:bg-chat-icon-bg-hover"
      >
        <HugeiconsIcon
          icon={RemoveCircleIcon}
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
        className="size-7 shrink-0 rounded-full p-0 text-muted-foreground opacity-0 transition-opacity hover:text-foreground focus-visible:opacity-100 group-hover/row:opacity-100 dark:hover:bg-chat-icon-bg-hover [@media(hover:none)]:opacity-100"
      >
        <HugeiconsIcon icon={CodeIcon} strokeWidth={1.75} className="size-4" />
      </Button>
    </ToolbarTooltip>
  );
  const ejectLabel =
    served.length > 1
      ? `Eject all ${served.length} models and free their memory`
      : served.length === 1
        ? `Eject ${served[0].name} and free its memory`
        : "No model is loaded";
  const rowFacts = (model: ServedModel): ReactElement => {
    const { id } = model;
    // Since the latest completed load: an earlier one may have had another window; a failed one changed nothing.
    let loadedAt = 0;
    for (const entry of entries) {
      if (
        isLifecycleEntry(entry) &&
        entry.event === "load" &&
        entry.status === "completed" &&
        modelIdsMatch(monitorSelection(entry.model).id, id)
      ) {
        loadedAt = Math.max(loadedAt, entry.started_at);
      }
    }
    const requests = entries.filter(
      (entry) =>
        !isLifecycleEntry(entry) &&
        entry.started_at >= loadedAt &&
        // Requests may be logged as "<id>:<quant>".
        modelIdsMatch(monitorSelection(entry.model).id, id),
    );
    let last: ApiMonitorEntry | null = null;
    for (const entry of requests) {
      if (
        entry.total_tokens != null &&
        entry.context_length &&
        (last == null || entry.started_at > last.started_at)
      ) {
        last = entry;
      }
    }
    const rowStats = computeStats(requests);
    const used = last?.total_tokens ?? (model.active ? 0 : null);
    const windowSize =
      last?.context_length ?? (model.active ? data?.context_length : null);
    return (
      <>
        <dl className={FACT_CELL}>
          {used != null && windowSize ? (
            <Fact label="Context">
              <ContextUsageChip
                ringFirst={true}
                pinOnClick={true}
                hoverClassName="rounded-full transition-colors group-hover:bg-muted dark:group-hover:bg-chat-icon-bg-hover"
                padX={2}
                fitContent={true}
                className="-mx-2 -my-1 h-6 font-sans text-ui-12 tabular-nums [--icon-size:--spacing(3.5)] [&>button]:rounded-full"
                used={used}
                total={windowSize}
                promptTokens={last?.prompt_tokens ?? undefined}
                completionTokens={last?.completion_tokens ?? undefined}
              />
            </Fact>
          ) : null}
        </dl>
        <dl className={FACT_CELL}>
          {rowStats.tokensPerSecond != null ? (
            <Fact label="Speed" icon={SpeedMeterIcon}>
              <span className="tabular-nums">
                {rowStats.tokensPerSecond.toFixed(1)} tok/s
              </span>
            </Fact>
          ) : null}
        </dl>
        <dl className={FACT_CELL}>
          {rowStats.avgDurationMs != null ? (
            <Fact label="Total time, average per request" icon={Clock01Icon}>
              <span className="tabular-nums">
                {formatDuration(rowStats.avgDurationMs)}
              </span>
            </Fact>
          ) : null}
        </dl>
        <dl className={FACT_CELL}>
          <Fact label="Requests" icon={ArrowDataTransferVerticalIcon}>
            <span className="tabular-nums">{formatCount(rowStats.total)}</span>
            {rowStats.errors > 0 ? (
              <span className="tabular-nums text-red-600 dark:text-red-400">
                · {formatCount(rowStats.errors)}{" "}
                {rowStats.errors === 1 ? "error" : "errors"}
              </span>
            ) : null}
          </Fact>
        </dl>
      </>
    );
  };
  return (
    <>
      {/* In the macOS titlebar row like Chat's header; click-through so the row still drags. */}
      <div
        className={cn(
          "pointer-events-none sticky top-0 z-40 flex min-h-[calc(48px*var(--ui-space-scale,1))] shrink-0 items-start bg-background pt-[var(--studio-chat-header-padding-top,11px)]",
          !isMobileShell &&
            "-mt-[calc(var(--studio-non-chat-content-top-inset,var(--studio-content-top-inset,0px))-var(--studio-content-top-inset,0px))]",
          isMobileShell
            ? "pl-12"
            : pinned
              ? "pl-2"
              : isTauri
                ? "pl-[var(--studio-collapsed-chat-controls-inset,0.75rem)]"
                : "pl-[calc(0.5rem*var(--ui-space-scale,1)+max(0px,var(--studio-mac-traffic-light-inset,0px)-var(--sidebar-width-icon,3rem)))]",
        )}
      >
        <div data-tour="api-toolbar" className="pointer-events-auto flex min-w-0 max-w-full">
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
            onLastLoadChanged={reload.refreshLastLoad}
          />
        </div>
        <div
          aria-hidden={true}
          className="pointer-events-none absolute inset-x-0 top-full h-6 bg-gradient-to-b from-background to-transparent"
        />
      </div>
      <main className="api-monitor-page @container/page mx-auto flex w-full max-w-6xl 3xl:max-w-[calc(1440px*var(--ui-space-scale,1))] 4xl:max-w-[calc(1760px*var(--ui-space-scale,1))] flex-col gap-5 px-6 pb-10 pt-4 max-sm:px-4 max-sm:pt-2 sm:px-10">
        <GuidedTour {...tour.tourProps} />
        <header className="flex items-start justify-between gap-4 @2xl/page:items-end">
          <div className="flex min-w-0 flex-col gap-1">
            <div className="flex flex-wrap items-center gap-3">
              <h1 className="font-heading text-ui-30 font-semibold leading-[1.04] tracking-[-0.028em] text-foreground sm:text-ui-34">
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
              className="size-8 rounded-full text-muted-foreground hover:bg-[color-mix(in_oklab,var(--foreground)_calc(10%*var(--contrast-wash-gain,1)),transparent)] hover:text-foreground dark:hover:bg-[color-mix(in_oklab,var(--foreground)_calc(14%*var(--contrast-wash-gain,1)),transparent)]"
            >
              <HugeiconsIcon
                icon={Settings02Icon}
                strokeWidth={1.75}
                className="size-[calc(18px*var(--ui-space-scale,1))]"
              />
            </Button>
          </ToolbarTooltip>
        </header>

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
                <span className="text-ui-11 text-muted-foreground">
                  Base URL
                </span>
                {/* h-5: at the copy button's 28px the chip sat below the text. */}
                <div className="group/copy flex h-5 min-w-0 items-center gap-1">
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
            <div className="flex items-center gap-2">
              <ToolbarTooltip label={reload.reloadAllTitle(served)}>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={() => void reload.reloadAll(served)}
                  disabled={
                    reloadBusy ||
                    (served.length === 0 && reload.lastLoadLabel == null) ||
                    (served.length > 0 &&
                      !served.some((model) => reload.canReload(model)))
                  }
                  className="gap-1.5 rounded-full"
                >
                  <HugeiconsIcon
                    icon={ReloadIcon}
                    strokeWidth={1.75}
                    className={cn(
                      "size-4",
                      reload.reloadingKey === "*" && "animate-spin",
                    )}
                  />
                  {reload.reloadingKey === "*"
                    ? "Reloading"
                    : served.length > 1
                      ? "Reload all"
                      : "Reload"}
                </Button>
              </ToolbarTooltip>
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
                  className="gap-1.5 rounded-full"
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
            </div>
            {reload.error ? (
              <span
                role="alert"
                className="basis-full break-words text-ui-12 text-red-600 dark:text-red-400"
              >
                {reload.error}
              </span>
            ) : null}
          </div>
          <div className={cn("border-t px-5 pt-3.5 pb-5", DIVIDER)}>
            {served.length === 0 ? (
              <p className="text-ui-12 text-muted-foreground">
                No models loaded
                <MultiModelHint servedCount={0} />
              </p>
            ) : (
              <div className="@container/served flex flex-col gap-2">
                <p className="text-ui-12 text-muted-foreground">
                  {served.length === 1
                    ? "1 model answering requests"
                    : `${served.length} models answering requests`}
                  <MultiModelHint servedCount={served.length} />
                </p>
                {/* Not fixed widths: they left a short value's slack before the next column. */}
                <div className="flex flex-col gap-3 @3xl/served:grid @3xl/served:grid-cols-[minmax(0,max-content)_auto_auto_auto_auto_1fr] @3xl/served:items-center @3xl/served:gap-x-10 @3xl/served:gap-y-1.5">
                  {served.map((model) => (
                    // A subgrid, not `contents`: the row needs a box (gaps included) to hover.
                    <div
                      key={model.checkpoint}
                      className="group/row flex flex-wrap items-center gap-x-3 gap-y-1 @md/served:gap-x-4 @3xl/served:col-span-full @3xl/served:grid @3xl/served:grid-cols-subgrid @3xl/served:gap-x-10"
                    >
                      <span className="order-1 flex min-w-0 flex-1 items-center gap-1 text-ui-12 @max-3xl/served:basis-full @max-3xl/served:items-start @3xl/served:order-none">
                        <span
                          className="size-1.5 shrink-0 rounded-full bg-emerald-500 @max-3xl/served:mt-1.5"
                          aria-hidden={true}
                        />
                        <span className="ml-1 min-w-0 truncate font-mono text-foreground @max-3xl/served:whitespace-normal @max-3xl/served:[overflow-wrap:anywhere]">
                          {model.name}
                        </span>
                        {exampleButton(model)}
                      </span>
                      {rowFacts(model)}
                      <span className="order-3 ml-auto flex shrink-0 items-center gap-0.5 justify-self-end @3xl/served:order-none @3xl/served:ml-0">
                        {settingsButton(model)}
                        {reloadButton(model)}
                        {ejectButton(model)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </section>

        {error ? <Banner>{error}</Banner> : null}
        {unloadError ? (
          <Banner onDismiss={() => setUnloadError(null)}>{unloadError}</Banner>
        ) : null}

        <section
          aria-label="Traffic summary"
          className="elevated-card grid grid-cols-2 gap-px overflow-hidden bg-border/60 @xl/page:grid-cols-3 @5xl/page:grid-cols-6 dark:bg-background"
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

        <section
          className={cn("flex min-h-0 flex-col overflow-hidden", SURFACE)}
        >
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
                      "flex h-9 shrink-0 items-center gap-1.5 rounded-full px-3.5 font-heading text-ui-14 text-foreground/80 transition-colors hover:text-foreground",
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
              <Select
                value={effectiveModelFilter}
                onValueChange={setModelFilter}
              >
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
                  {paused ? (
                    <PlayIcon strokeWidth={1.75} className="size-4" />
                  ) : (
                    <HugeiconsIcon
                      icon={PauseIcon}
                      strokeWidth={1.75}
                      className="size-4"
                    />
                  )}
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
                  className="rounded-full text-red-500/80 hover:text-red-500 dark:text-red-400/80 dark:hover:text-red-400"
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
                      <Skeleton key={i} className="h-16 w-full rounded-md" />
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
                      className="rounded-full"
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
                    <span className="flex size-10 items-center justify-center icon-disc text-muted-foreground">
                      <HugeiconsIcon
                        icon={FileSearchIcon}
                        strokeWidth={1.75}
                        className="size-5"
                      />
                    </span>
                    <p className="max-w-xs text-sm text-muted-foreground">
                      Select a request to see its details. Arrow keys move
                      through the list.
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
    </>
  );
}
