// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Floating API monitor: opens itself when API traffic arrives, summarises it
// without taking over the window, and links through to the full page.

import { getApiMonitor } from "@/features/chat/api/chat-api";
import type { ApiMonitorEntry } from "@/features/chat/types/api";
import { cn } from "@/lib/utils";
import {
  ArrowExpand01Icon,
  DragDropVerticalIcon,
  Globe02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useNavigate, useRouterState } from "@tanstack/react-router";
import { XIcon } from "lucide-react";
import { AnimatePresence, motion, useDragControls } from "motion/react";
import {
  type PointerEvent,
  type ReactElement,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { isLifecycleEntry, lifecycleLabel } from "./lifecycle";
import {
  type ApiMonitorWatch,
  createWatch,
  observeResponse,
  rearmWatch,
  standDownWatch,
  startWatching,
} from "./new-traffic";
import { useApiMonitorOverlayStore } from "./overlay-store";
import { computeStats } from "./use-api-monitor";

// Live cadence while the panel is on screen.
const OPEN_POLL_MS = 1500;
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
    return "live";
  }
  if (value < 1000) {
    return `${Math.round(value)}ms`;
  }
  return `${(value / 1000).toFixed(value < 10000 ? 1 : 0)}s`;
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
  const navigate = useNavigate();
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const onFullPage = pathname === "/api-monitor";

  const [data, setData] = useState<Awaited<
    ReturnType<typeof getApiMonitor>
  > | null>(null);

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
    const intervalMs = isOpen ? OPEN_POLL_MS : IDLE_POLL_MS;

    // Anchor the watch here, not at mount: the first snapshot can arrive much later
    // (a hidden tab skips its poll), and everything terminal would read as history.
    startWatching(watchRef.current, performance.now());

    function schedule(): void {
      timer = window.setTimeout(poll, intervalMs);
    }

    function poll(): void {
      // A hidden tab has nobody to show the panel to.
      if (document.hidden) {
        schedule();
        return;
      }
      getApiMonitor()
        .then((next) => {
          if (!cancelled) setData(next);
        })
        .catch(() => {
          // An unreachable server is the full page's story to tell.
          if (!cancelled) setData(null);
        })
        .finally(() => {
          if (!cancelled) schedule();
        });
    }

    poll();
    return () => {
      cancelled = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [isOpen, onFullPage, autoOpen]);

  const entries = useMemo(() => data?.entries ?? [], [data]);
  const stats = useMemo(() => computeStats(entries), [entries]);

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

  const [constraintsElement, setConstraintsElement] =
    useState<HTMLDivElement | null>(null);
  const constraintsRef = useMemo(
    () => ({ current: constraintsElement }),
    [constraintsElement],
  );
  const dragControls = useDragControls();

  function startDrag(event: PointerEvent<HTMLDivElement>): void {
    event.preventDefault();
    dragControls.start(event);
  }

  const visible = isOpen && !onFullPage;
  const serverStatus = data?.status ?? "idle";

  return (
    <AnimatePresence>
      {visible && (
        <div
          ref={setConstraintsElement}
          className="pointer-events-none fixed inset-0 z-50"
        >
          <motion.div
            drag={true}
            dragControls={dragControls}
            dragListener={false}
            dragConstraints={constraintsRef}
            dragElastic={0}
            dragMomentum={false}
            initial={{ opacity: 0, scale: 0.94 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.94 }}
            /* Panel language from the sidebar menu: soft surface, 20px corner, heading font. */
            className="menu-soft-surface pointer-events-auto fixed bottom-4 right-4 flex w-[400px] max-w-[calc(100vw-2rem)] cursor-default select-none resize flex-col overflow-hidden rounded-[20px] border-0 p-2.5 font-heading ring-0"
          >
            <div className="flex items-center justify-between gap-2 px-1.5 pb-2 pt-0.5">
              <div className="flex min-w-0 flex-1 items-center gap-2">
                <HugeiconsIcon
                  icon={Globe02Icon}
                  strokeWidth={1.75}
                  className="size-icon shrink-0 text-nav-fg"
                />
                <span className="truncate text-ui-13p5 font-semibold tracking-[0.025em] text-nav-fg dark:tracking-[0.04em]">
                  API monitor
                </span>
                <span
                  className={cn(
                    "size-1.5 shrink-0 rounded-full",
                    serverStatus === "generating"
                      ? "animate-pulse bg-blue-500"
                      : serverStatus === "ready"
                        ? "bg-emerald-500"
                        : "bg-muted-foreground",
                  )}
                  aria-hidden={true}
                />
              </div>
              <div className="flex shrink-0 items-center gap-0.5">
                <div
                  onPointerDown={startDrag}
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

            <p className="truncate px-1.5 pb-2.5 text-ui-11p5 tracking-nav text-muted-foreground">
              {data?.active_model ?? "No model loaded"}
            </p>

            {/* Soft tile, as the Hub and Train pages group readouts. */}
            <div className="grid grid-cols-4 rounded-[14px] bg-muted/45 py-2.5 dark:bg-background/45">
              <StatCell
                label="Live"
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
                label="Avg"
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
                  No requests yet.
                </p>
              ) : (
                entries.slice(0, VISIBLE_ENTRIES).map((entry) => (
                  <div
                    key={entry.id}
                    className="flex h-9 min-w-0 items-center gap-2.5 rounded-[12px] px-3 transition-colors hover:bg-nav-surface-hover"
                  >
                    <span
                      className={cn(
                        "size-1.5 shrink-0 rounded-full",
                        statusDotClass(entry.status),
                      )}
                      aria-hidden={true}
                    />
                    <span className="shrink-0 truncate text-ui-12p5 font-medium tracking-nav text-nav-fg">
                      {isLifecycleEntry(entry)
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
                    <span className="shrink-0 text-ui-11 tabular-nums text-muted-foreground">
                      {formatDuration(entry.duration_ms)}
                    </span>
                  </div>
                ))
              )}
            </div>

            {/* Through to payloads, filters and per request tokens. */}
            <button
              type="button"
              onClick={() => {
                close();
                void navigate({ to: "/api-monitor" });
              }}
              className="mt-1 flex h-[33px] w-full items-center justify-center gap-[8.5px] rounded-full bg-muted/60 text-ui-13p5 font-medium tracking-nav text-nav-fg transition-colors hover:bg-nav-surface-hover focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring dark:bg-background/50"
            >
              <HugeiconsIcon
                icon={ArrowExpand01Icon}
                strokeWidth={1.75}
                className="size-icon shrink-0"
              />
              Expand to full monitor
            </button>

            {/* Closing silences this burst; this is the permanent off. */}
            <button
              type="button"
              onClick={() => {
                setAutoOpen(false);
                close();
              }}
              className="mt-1.5 w-full rounded-[12px] py-1 text-ui-11 tracking-nav text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            >
              Stop opening this automatically
            </button>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
}
