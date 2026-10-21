// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Floating API monitor: opens itself when API traffic arrives, summarises it
// without taking over the window, and links through to the full page.

import { getApiMonitor } from "@/features/chat/api/chat-api";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import type {
  ApiMonitorEntry,
  ApiMonitorResponse,
} from "@/features/chat/types/api";
import { FIND_PORTAL_ATTRIBUTE } from "@/features/find-in-page/lib/find-attributes";
import { useMonitorFrameStore, useShortcut } from "@/features/settings";
import { useFloatingPanelLayout } from "@/hooks/use-floating-panel-layout";
import {
  useFloatingPanelOrderStore,
  useFloatingPanelZIndex,
} from "@/lib/floating-panel-order";
import { cn } from "@/lib/utils";
import {
  ExpandIcon,
  DragDropVerticalIcon,
  ApiIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate, useRouterState } from "@tanstack/react-router";
import { XIcon } from "lucide-react";
import { AnimatePresence, motion, useIsPresent } from "motion/react";
import { type ReactElement, useEffect, useMemo, useRef, useState } from "react";
import { isLifecycleEntry, lifecycleLabel } from "./lifecycle";
import { mergeMonitorResponse } from "./monitor-snapshot";
import {
  type ApiMonitorWatch,
  createWatch,
  observeResponse,
  rearmWatch,
  standDownWatch,
  startWatching,
} from "./new-traffic";
import { useApiMonitorOverlayStore } from "./overlay-store";
import {
  isFullyCovered,
  PANEL_MARGIN,
  PANEL_TOP_MARGIN,
  placeFloatingPanel,
} from "./panel-placement";
import { computeStats } from "./use-api-monitor";

// Live cadence while the panel is on screen.
const OPEN_POLL_MS = 1500;
// The body scrolls, so the grip may shrink below content size (the hardware monitor's floor).
const MIN_PANEL_WIDTH = 280;
const MIN_PANEL_HEIGHT = 200;
// Closed, the poll only has to notice traffic started, so it backs off.
const IDLE_POLL_MS = 5000;
// Requests shown in the panel; the rest are one click away on the full page.
const VISIBLE_ENTRIES = 4;
// Quiet time before a dismissed panel re-arms.
const REARM_QUIET_MS = 60_000;

const API_INFERENCE_PREFIX_RE = /^\/api\/inference/;
const V1_PREFIX_RE = /^\/v1\//;

function compactEndpoint(endpoint: string): string {
  return endpoint
    .replace(API_INFERENCE_PREFIX_RE, "/api")
    .replace(V1_PREFIX_RE, "/");
}

function formatDuration(value?: number | null): string {
  if (value == null) {
    return "–";
  }
  if (value < 1000) {
    return `${Math.round(value)}ms`;
  }
  return `${(value / 1000).toFixed(value < 10000 ? 1 : 0)}s`;
}

// Only a running row is live: a finished one without a duration has none to show.
function entryDuration(entry: ApiMonitorEntry): string {
  if (entry.duration_ms == null && entry.status === "running") {
    return "live";
  }
  return formatDuration(entry.duration_ms);
}

const SERVER_STATUS: Record<
  ApiMonitorResponse["status"],
  { label: string; dot: string }
> = {
  generating: { label: "Serving", dot: "animate-pulse bg-blue-500" },
  ready: { label: "Ready", dot: "bg-emerald-500" },
  idle: { label: "No model", dot: "bg-muted-foreground" },
};

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

function StatCell({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: "error" | "active";
}): ReactElement {
  return (
    <div className="flex min-w-0 flex-col items-center gap-0.5 px-1">
      <span
        className={cn(
          "truncate text-ui-15 font-semibold tabular-nums leading-none tracking-[-0.01em]",
          tone === "error" && "text-red-600 dark:text-red-400",
          tone === "active" && "text-blue-600 dark:text-blue-400",
          !tone && "text-nav-fg",
        )}
      >
        {value}
      </span>
      {/* Sentence case: metric rows read as words, not headers. */}
      <span className="truncate text-ui-11 tracking-nav text-muted-foreground">
        {label}
      </span>
    </div>
  );
}

export function ApiMonitorOverlay(): ReactElement | null {
  const { isOpen, suppressed, autoOpen, open, close, setAutoOpen } =
    useApiMonitorOverlayStore();
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const onFullPage = pathname === "/api-monitor";

  const [data, setData] = useState<ApiMonitorResponse | null>(null);
  // Kept apart from `data`: a failed poll keeps the last rows instead of blanking them.
  const [unreachable, setUnreachable] = useState(false);

  // The chord lives with the panel: reaching this store from the shell would
  // pull the whole feature index into the root chunk. The full page has
  // nothing to toggle, being the panel's contents already.
  useShortcut(
    "toggleApiMonitor",
    () => {
      if (isOpen) close();
      else open();
    },
    { enabled: !onFullPage },
  );

  // What this session has already shown, and when it started watching.
  const watchRef = useRef<ApiMonitorWatch>(createWatch(0));
  const lastNewEntryAtRef = useRef(0);

  // One loop for both jobs: panel contents while open, traffic watch while closed.
  // Stands down on the full page, which polls itself.
  useEffect(() => {
    // Opted out and closed: nothing to open or show, so polling is pure load.
    if (onFullPage || (!autoOpen && !isOpen)) {
      // Standing down for the opt out leaves the same backlog behind it as the full page
      // does, so write it off the same way: turning automatic opening back on waits for
      // the next call rather than popping with the burst that ran while it was off.
      if (!onFullPage) {
        standDownWatch(watchRef.current);
      }
      return;
    }
    let cancelled = false;
    let timer: number | undefined;
    let inFlight = false;
    // The first snapshot of each stand up is a fresh object, so the watch folds it even when it
    // matches the one from before the stand down. Later polls share what did not change, so a
    // quiet server re-renders nothing.
    let first = true;
    const intervalMs = isOpen ? OPEN_POLL_MS : IDLE_POLL_MS;

    // Anchor the watch here, not at mount: the first snapshot can arrive much later
    // (a hidden tab skips its poll), and everything terminal would read as history.
    startWatching(watchRef.current, performance.now());

    // A hidden tab has nobody to show the panel to, so it stops until the tab shows again.
    function schedule(): void {
      timer = window.setTimeout(() => {
        timer = undefined;
        if (!document.hidden) poll();
      }, intervalMs);
    }

    function poll(): void {
      inFlight = true;
      getApiMonitor()
        .then((next) => {
          if (cancelled) return;
          const fresh = first;
          first = false;
          setData((previous) =>
            fresh ? next : mergeMonitorResponse(previous, next),
          );
          setUnreachable(false);
        })
        .catch(() => {
          if (!cancelled) setUnreachable(true);
        })
        .finally(() => {
          inFlight = false;
          if (!cancelled) schedule();
        });
    }

    function onVisibilityChange(): void {
      if (document.hidden || inFlight || cancelled) return;
      if (timer !== undefined) {
        window.clearTimeout(timer);
        timer = undefined;
      }
      poll();
    }

    if (!document.hidden) poll();
    document.addEventListener("visibilitychange", onVisibilityChange);
    return () => {
      cancelled = true;
      document.removeEventListener("visibilitychange", onVisibilityChange);
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [isOpen, onFullPage, autoOpen]);

  useEffect(() => {
    if (data == null) {
      return;
    }
    const hasNewTraffic = observeResponse(
      watchRef.current,
      data,
      performance.now(),
    );
    if (!hasNewTraffic) {
      return;
    }
    const now = Date.now();
    const quietFor = now - lastNewEntryAtRef.current;
    lastNewEntryAtRef.current = now;
    if (!autoOpen || isOpen) {
      return;
    }
    // A dismissal holds for the burst and re-arms only once the API goes quiet.
    if (suppressed && quietFor < REARM_QUIET_MS) {
      return;
    }
    open();
  }, [data, autoOpen, suppressed, isOpen, open]);

  // The backlog built while the poll stood down is not new traffic. The watch re-anchors
  // when the poll stands back up, not here, or the panel pops with rows just read.
  useEffect(() => {
    if (onFullPage) {
      rearmWatch(watchRef.current);
    }
  }, [onFullPage]);

  const visible = isOpen && !onFullPage;
  const [panelKey, setPanelKey] = useState(0);
  const wasVisibleRef = useRef(visible);
  // Remount on reopen: fresh native size and frame owner.
  useEffect(() => {
    if (wasVisibleRef.current && !visible) {
      setPanelKey((current) => current + 1);
    }
    wasVisibleRef.current = visible;
  }, [visible]);
  return (
    <AnimatePresence>
      {visible && (
        <ApiMonitorPanel
          key={panelKey}
          data={data}
          unreachable={unreachable}
          close={close}
          setAutoOpen={setAutoOpen}
        />
      )}
    </AnimatePresence>
  );
}

function ApiMonitorPanel({
  data,
  unreachable,
  close,
  setAutoOpen,
}: {
  data: ApiMonitorResponse | null;
  unreachable: boolean;
  close: () => void;
  setAutoOpen: (value: boolean) => void;
}): ReactElement {
  const navigate = useNavigate();
  const focusRequest = useApiMonitorOverlayStore((s) => s.focusRequest);
  const entries = useMemo(() => data?.entries ?? [], [data]);
  const stats = useMemo(() => computeStats(entries), [entries]);
  const loadedModels = useChatRuntimeStore((s) => s.loadedModels);
  // The monitor names only the primary; Chat's runtime lists every model answering requests,
  // so a kept model still serving after the primary's eject does not read as "No model".
  const servedNames = useMemo(() => {
    const names = loadedModels.map((model) =>
      model.quant && !model.id.includes(":")
        ? `${model.id}:${model.quant}`
        : model.id,
    );
    const active = data?.active_model;
    if (!active) return names;
    const key = active.toLowerCase();
    return [active, ...names.filter((name) => name.toLowerCase() !== key)];
  }, [loadedModels, data?.active_model]);
  const serverStatus =
    SERVER_STATUS[
      data?.status === "idle" && servedNames.length > 0
        ? "ready"
        : (data?.status ?? "idle")
    ];
  const openApiPage = (requestId?: string): void => {
    if (requestId) focusRequest(requestId);
    close();
    void navigate({ to: "/api-monitor" });
  };
  const isPresent = useIsPresent();
  const [constraintsElement, setConstraintsElement] =
    useState<HTMLDivElement | null>(null);
  const {
    monitorRef,
    scrollRef,
    contentRef,
    layout,
    publisher,
    startDrag,
    updateDrag,
    finishDrag,
  } = useFloatingPanelLayout(
    constraintsElement,
    false,
    // An exiting panel must not stay an obstacle for the one reopening.
    !isPresent,
    (size, bounds) => {
      const initial = placeFloatingPanel(
        size,
        [...useMonitorFrameStore.getState().frames.values()],
        {
          width: window.innerWidth,
          height: window.innerHeight,
        },
      );
      return {
        left: initial.left - bounds.left,
        top: initial.top - bounds.top,
      };
    },
  );
  const frames = useMonitorFrameStore((state) => state.frames);
  const ownFrame = frames.get(publisher);
  const covered =
    ownFrame !== undefined &&
    isFullyCovered(
      ownFrame,
      {
        width: ownFrame.right - ownFrame.left,
        height: ownFrame.bottom - ownFrame.top,
      },
      [...frames]
        .filter(([owner]) => owner !== publisher)
        .map(([, frame]) => frame),
    );
  const zIndex = useFloatingPanelZIndex("api-monitor", covered);
  const raisePanel = useFloatingPanelOrderStore((state) => state.raise);
  useEffect(() => {
    raisePanel("api-monitor");
  }, [raisePanel]);
  return (
    <div
      ref={setConstraintsElement}
      className="pointer-events-none fixed"
      style={{
        zIndex,
        left: PANEL_MARGIN,
        right: PANEL_MARGIN,
        top: PANEL_TOP_MARGIN,
        bottom: PANEL_MARGIN,
      }}
    >
      <motion.div
        {...{ [FIND_PORTAL_ATTRIBUTE]: "" }}
        ref={monitorRef}
        data-testid="api-monitor-panel"
        // Takes its own clicks over a macOS browser page (native-view.ts).
        data-native-clickable=""
        onPointerDownCapture={() => raisePanel("api-monitor")}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        /* Panel language from the sidebar menu: soft surface, 20px corner, heading font. */
        className={cn(
          "menu-soft-surface pointer-events-auto absolute flex max-h-full w-[calc(400px*var(--ui-space-scale,1))] max-w-full cursor-default select-none flex-col overflow-hidden rounded-[20px] border-0 p-2.5 font-heading ring-0",
          // The CSS corner until it has been measured, so the first paint
          // is the corner it has always opened in rather than the top left.
          layout ? "top-0 left-0 resize" : "bottom-0 right-0",
        )}
        style={
          layout
            ? {
                left: layout.left,
                top: layout.top,
                minWidth: Math.min(MIN_PANEL_WIDTH, layout.maxWidth),
                minHeight: Math.min(MIN_PANEL_HEIGHT, layout.maxHeight),
                maxWidth: layout.maxWidth,
                maxHeight: layout.maxHeight,
              }
            : undefined
        }
      >
        <div className="flex shrink-0 items-center justify-between gap-2 px-1.5 pb-2 pt-0.5">
          <div className="flex min-w-0 flex-1 items-center gap-2">
            <HugeiconsIcon
              icon={ApiIcon}
              strokeWidth={1.75}
              className="size-icon shrink-0 text-nav-fg"
            />
            <span className="truncate text-ui-13p5 font-semibold tracking-[0.025em] text-nav-fg dark:tracking-[0.04em]">
              API monitor
            </span>
            <span className="flex shrink-0 items-center gap-1.5 text-ui-11 tracking-nav text-muted-foreground">
              <span
                className={cn(
                  "size-1.5 shrink-0 rounded-full",
                  unreachable ? "bg-red-500" : serverStatus.dot,
                )}
                aria-hidden={true}
              />
              {unreachable ? "Offline" : serverStatus.label}
            </span>
          </div>
          <div className="flex shrink-0 items-center gap-0.5">
            <div
              onPointerDown={startDrag}
              onPointerMove={updateDrag}
              onPointerUp={finishDrag}
              onPointerCancel={finishDrag}
              onLostPointerCapture={finishDrag}
              data-testid="api-monitor-drag-handle"
              className="flex size-7 touch-none cursor-grab items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-nav-surface-hover hover:text-foreground active:cursor-grabbing"
            >
              <HugeiconsIcon
                icon={DragDropVerticalIcon}
                strokeWidth={1.75}
                className="size-4"
              />
            </div>
            <button
              type="button"
              onClick={close}
              title="Close"
              aria-label="Close API monitor"
              className="flex size-7 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-nav-surface-hover hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            >
              <XIcon className="size-3.5" strokeWidth={1.75} />
            </button>
          </div>
        </div>

        <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto">
          <div ref={contentRef}>
            <p
              className={cn(
                "flex min-w-0 items-center gap-1.5 px-1.5 pb-2.5 text-ui-11p5 tracking-nav text-muted-foreground",
                unreachable && "text-red-600 dark:text-red-400",
              )}
              title={
                servedNames.length > 1 ? servedNames.join("\n") : undefined
              }
            >
              <span className="min-w-0 truncate">
                {unreachable
                  ? "Can't reach Unsloth. Retrying…"
                  : (servedNames[0] ?? "No model loaded")}
              </span>
              {!unreachable && servedNames.length > 1 ? (
                <span className="shrink-0 rounded-full bg-muted/60 px-1.5 text-ui-10 tabular-nums dark:bg-background/50">
                  +{servedNames.length - 1}
                </span>
              ) : null}
            </p>

            {/* Soft tile, as the Hub and Train pages group readouts. */}
            <div className="grid grid-cols-4 rounded-[14px] bg-muted/45 py-2.5 dark:bg-background/45">
              <StatCell
                label="In flight"
                value={stats.active.toLocaleString()}
                tone={stats.active > 0 ? "active" : undefined}
              />
              <StatCell label="Requests" value={stats.total.toLocaleString()} />
              <StatCell
                label="Errors"
                value={stats.errors.toLocaleString()}
                tone={stats.errors > 0 ? "error" : undefined}
              />
              <StatCell
                label="Avg time"
                value={
                  stats.avgDurationMs == null
                    ? "--"
                    : formatDuration(stats.avgDurationMs)
                }
              />
            </div>

            {/* Borderless rows on 12px hover pills, as in the sidebar. */}
            <div className="flex flex-col pb-1 pt-1.5">
              {entries.length === 0 ? (
                <p className="px-1.5 py-4 text-center text-ui-11p5 tracking-nav text-muted-foreground">
                  No API requests yet. They show here as they arrive.
                </p>
              ) : (
                entries.slice(0, VISIBLE_ENTRIES).map((entry) => {
                  const lifecycle = isLifecycleEntry(entry);
                  const cells = (
                    <>
                      <span
                        className={cn(
                          "size-1.5 shrink-0 rounded-full",
                          statusDotClass(entry.status),
                        )}
                        aria-hidden={true}
                      />
                      <span className="shrink-0 truncate text-ui-12p5 font-medium tracking-nav text-nav-fg">
                        {lifecycle
                          ? lifecycleLabel(entry)
                          : compactEndpoint(entry.endpoint)}
                      </span>
                      <span
                        className={cn(
                          "min-w-0 flex-1 truncate text-ui-11p5 tracking-nav",
                          entry.error
                            ? "text-red-600 dark:text-red-400"
                            : "text-muted-foreground",
                        )}
                      >
                        {entry.error ? entry.error : entry.model}
                      </span>
                      <span
                        className={cn(
                          "shrink-0 text-ui-11 tabular-nums text-muted-foreground",
                          entry.status === "running" &&
                            "text-blue-600 dark:text-blue-400",
                        )}
                      >
                        {entryDuration(entry)}
                      </span>
                    </>
                  );
                  // A load or unload has no payload to inspect, so only requests open the page.
                  return lifecycle ? (
                    <div
                      key={entry.id}
                      className="flex h-9 min-w-0 items-center gap-2.5 px-3"
                    >
                      {cells}
                    </div>
                  ) : (
                    <button
                      key={entry.id}
                      type="button"
                      data-entry-id={entry.id}
                      onClick={() => openApiPage(entry.id)}
                      title="Inspect this request on the API page"
                      className="flex h-9 w-full min-w-0 items-center gap-2.5 rounded-[12px] px-3 text-left transition-colors hover:bg-nav-surface-hover focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                    >
                      {cells}
                    </button>
                  );
                })
              )}
            </div>
          </div>
        </div>
        {/* Through to payloads, filters and per request tokens. */}
        <button
          type="button"
          onClick={() => openApiPage()}
          className="mt-1 flex shrink-0 h-[calc(33px*var(--ui-space-scale,1))] w-full items-center justify-center gap-[calc(8.5px*var(--ui-space-scale,1))] rounded-full bg-muted/60 text-ui-13p5 font-medium tracking-nav text-nav-fg transition-colors hover:bg-nav-surface-hover focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring dark:bg-background/50"
        >
          <HugeiconsIcon
            icon={ExpandIcon}
            strokeWidth={1.75}
            className="size-icon shrink-0"
          />
          Open API page
        </button>

        {/* Closing silences this burst; this is the permanent off. */}
        <button
          type="button"
          onClick={() => {
            setAutoOpen(false);
            close();
          }}
          title="Turn it back on in Settings, under API"
          className="mt-1.5 shrink-0 w-full rounded-[12px] py-1 text-ui-11 tracking-nav text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
        >
          Stop opening this automatically
        </button>
      </motion.div>
    </div>
  );
}
