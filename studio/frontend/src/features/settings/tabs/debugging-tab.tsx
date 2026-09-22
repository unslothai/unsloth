// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  InputGroup,
  InputGroupAddon,
  InputGroupInput,
} from "@/components/ui/input-group";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useCopyFeedback } from "@/features/hub/hooks/use-copy-feedback";
import { formatBytes } from "@/features/hub";
import { useT } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { stripAnsi } from "@/lib/strip-ansi";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  Alert02Icon,
  ArrowDownDoubleIcon,
  Copy01Icon,
  Download01Icon,
  FolderOpenIcon,
  InformationCircleIcon,
  RefreshIcon,
  Search01Icon,
  Shield01Icon,
  TextWrapIcon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  type DebugLogSource,
  LogExportError,
  exportAllLogs,
  loadDebugLog,
  loadDebugLogSources,
  openLogsFolder,
  revealSavedArchive,
} from "../api/debug-logs";
import { SettingsRow } from "../components/settings-row";
import { SettingsSection } from "../components/settings-section";
import {
  DEFAULT_REFRESH_MODE,
  EMPTY_BUFFER,
  type LogBufferState,
  REFRESH_MODE_STORAGE_KEY,
  REQUEST_TIMEOUT_MS,
  type RefreshMode,
  applyLogChunk,
  isPageStale,
  nextDroppedState,
  parseRefreshMode,
  isRequestTimeout,
  pollDelayMs,
  withRequestTimeout,
} from "../lib/debug-log-buffer";
import { isAbort, isLogSourceGone } from "../lib/debug-log-error";

const MODES: RefreshMode[] = ["live", "3s", "manual"];

// Rescan cadence for log files that did not exist when the tab was opened.
// Slower than the poll: a directory walk rather than a tail read.
const SOURCE_RESCAN_MS = 10_000;

// how close to the bottom the pane must be scrolled for auto-follow to stay on
const FOLLOW_THRESHOLD_PX = 40;

function readStoredMode(): RefreshMode {
  if (typeof window === "undefined") return DEFAULT_REFRESH_MODE;
  try {
    return parseRefreshMode(
      window.localStorage.getItem(REFRESH_MODE_STORAGE_KEY),
    );
  } catch {
    return DEFAULT_REFRESH_MODE;
  }
}

function groupByFamily(
  sources: DebugLogSource[],
): { family: string; sources: DebugLogSource[] }[] {
  const groups = new Map<string, DebugLogSource[]>();
  for (const source of sources) {
    const group = groups.get(source.family);
    if (group) group.push(source);
    else groups.set(source.family, [source]);
  }
  return [...groups].map(([family, members]) => ({ family, sources: members }));
}

function NoticeStrip({
  tone,
  children,
  testId,
}: {
  tone: "warning" | "info";
  children: string;
  testId?: string;
}) {
  return (
    <p
      data-testid={testId}
      className={cn(
        "flex items-start gap-2 text-xs leading-snug",
        tone === "warning"
          ? "text-amber-600 dark:text-amber-400"
          : "text-muted-foreground",
      )}
    >
      {tone === "warning" ? (
        <HugeiconsIcon icon={Alert02Icon} className="mt-px size-3.5 shrink-0" />
      ) : null}
      {children}
    </p>
  );
}

export function DebuggingTab() {
  const t = useT();
  const [sources, setSources] = useState<DebugLogSource[]>([]);
  const [sourceId, setSourceId] = useState<string | null>(null);
  const [mode, setMode] = useState<RefreshMode>(readStoredMode);
  const [buffer, setBuffer] = useState<LogBufferState>(EMPTY_BUFFER);
  const [realpath, setRealpath] = useState<string | null>(null);
  // Where the backend says the logs live, for the folder button when no source
  // is selected yet, which is exactly the custom-home case that has no log.
  const [logRoot, setLogRoot] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [dropped, setDropped] = useState(false);
  // A burst larger than one response continues on the next poll, which in
  // manual mode never comes unless the user knows to ask for it.
  const [morePending, setMorePending] = useState(false);
  // File logging is off and an older session's log is still on disk, so the pane shows real content
  // that will never grow. Unsaid, a stale log is indistinguishable from a live one.
  const [staleSession, setStaleSession] = useState(false);
  // Each button tracks only its own request: the export takes seconds and must
  // not lock out "Show in folder", which is how the user reaches the result.
  const [exporting, setExporting] = useState(false);
  const [revealing, setRevealing] = useState(false);
  const [filter, setFilter] = useState("");
  const [wrap, setWrap] = useState(true);
  // mirrors pinnedRef for the jump-to-latest button; the ref stays the source of
  // truth so the scroll handler does not re-render the pane on every line
  const [following, setFollowing] = useState(true);
  const { copied, copy } = useCopyFeedback();
  const { copied: pathCopied, copy: copyPath } = useCopyFeedback();

  // In a ref as well as state: the poll loop must not restart per line arrived.
  const cursorRef = useRef<string | null>(null);
  // Counts source changes, so an in-flight request can tell its view moved.
  const selectionRef = useRef(0);
  // The selection in flight, not a bare flag: a poll for the newly picked source
  // must not be swallowed by a slow read of the one the user just left, or the
  // new pane stays empty (in manual mode, for good).
  const inFlightRef = useRef<number | null>(null);
  const paneRef = useRef<HTMLPreElement | null>(null);
  const pinnedRef = useRef(true);
  const lastSourceScanRef = useRef(Date.now());

  useEffect(() => {
    try {
      window.localStorage.setItem(REFRESH_MODE_STORAGE_KEY, mode);
    } catch {
      // A blocked localStorage must not stop the viewer working.
    }
  }, [mode]);

  const refreshSources = useCallback(
    async (options: { signal?: AbortSignal; reselect?: boolean } = {}) => {
      try {
        // Bounded like the tail read: the poll loop and its failure recovery
        // both await this, so an unanswered /sources would freeze both.
        const result = await withRequestTimeout(
          (signal) => loadDebugLogSources(signal),
          REQUEST_TIMEOUT_MS,
          options.signal,
        );
        setSources(result.sources);
        setLogRoot(result.logRoot);
        setSourceId((current) =>
          options.reselect
            ? result.defaultSourceId
            : (current ?? result.defaultSourceId),
        );
      } catch {
        // The log read reports the real reason; this just leaves the picker empty.
      }
    },
    [],
  );

  useEffect(() => {
    const controller = new AbortController();
    void refreshSources({ signal: controller.signal });
    return () => controller.abort();
  }, [refreshSources]);

  const onPollFailed = useCallback(
    async (error: unknown, signal?: AbortSignal) => {
      if (isAbort(error)) return;
      if (isRequestTimeout(error)) {
        // Not the raw message: the backstop duration is an internal number, and
        // the user needs the consequence.
        setNotice(t("settings.debugging.timeout"));
        return;
      }
      if (isLogSourceGone(error)) {
        // The id we hold is no longer enumerated (file removed, or pushed out of the per-family
        // window). The backend sends 404 so the picker rebuilds; without this the loop re-polls a
        // dead id forever. Reselecting the server's default terminates: it comes from the same
        // walk, and "nothing at all" is a 200 with a status, not another 404.
        cursorRef.current = null;
        await refreshSources({ signal, reselect: true });
        return;
      }
      setNotice((error as Error).message);
    },
    [refreshSources, t],
  );

  // The llama runner writes a NEW file per load attempt, so a list fetched at mount goes stale
  // exactly when it matters: fail a load with the tab open and that failure's log is not offered.
  const rescanSourcesIfStale = useCallback(
    async (signal?: AbortSignal) => {
      if (Date.now() - lastSourceScanRef.current < SOURCE_RESCAN_MS) return;
      lastSourceScanRef.current = Date.now();
      await refreshSources({ signal });
    },
    [refreshSources],
  );

  const poll = useCallback(
    async (signal?: AbortSignal) => {
      const selection = selectionRef.current;
      if (inFlightRef.current === selection) return;
      inFlightRef.current = selection;
      // Without the timeout a request that never settles pins inFlightRef
      // forever: every poll returns at the guard above and the pane freezes with
      // no error, since the catch never runs. A dropped tunnel does it.
      try {
        const page = await withRequestTimeout(
          (requestSignal) =>
            loadDebugLog({
              sourceId,
              cursor: cursorRef.current,
              signal: requestSignal,
            }),
          REQUEST_TIMEOUT_MS,
          signal,
        );
        // A manual refresh carries no abort signal, so one in flight across a
        // source switch would land the old file's lines under the new pick.
        if (
          isPageStale({
            requestSelection: selection,
            currentSelection: selectionRef.current,
            requestSourceId: sourceId,
            pageSourceId: page.sourceId,
          })
        )
          return;
        cursorRef.current = page.cursor;
        if (page.realpath) setRealpath(page.realpath);
        setDropped((previous) => nextDroppedState(previous, page));
        setMorePending(page.morePending);
        setStaleSession(page.fileLoggingDisabled);
        setNotice(
          page.status === "ok" || page.status === "empty"
            ? null
            : (page.reason ?? t(`settings.debugging.${page.status}` as never)),
        );
        setBuffer((previous) =>
          applyLogChunk(previous, {
            lines: page.lines,
            cursor: page.cursor,
            reset: page.reset,
          }),
        );
      } catch (error) {
        if (selection === selectionRef.current)
          await onPollFailed(error, signal);
      } finally {
        // Only if a poll for a newer selection has not taken the slot.
        if (inFlightRef.current === selection) inFlightRef.current = null;
      }
    },
    [onPollFailed, sourceId, t],
  );

  // Switching source starts a fresh read rather than appending to the old file.
  useEffect(() => {
    selectionRef.current += 1;
    cursorRef.current = null;
    setBuffer(EMPTY_BUFFER);
    setRealpath(null);
    // Every notice below describes the file being left, so all of them go with it. Clearing only
    // `dropped` let a failed first read on the new source keep claiming the OLD one's state, and in
    // manual mode nothing retries: the pane sat there calling a live log a frozen session.
    setDropped(false);
    setMorePending(false);
    setStaleSession(false);
    setNotice(null);
  }, [sourceId]);

  useEffect(() => {
    const controller = new AbortController();
    let timer: number | undefined;
    let stopped = false;

    // A self-scheduling timeout, not setInterval: the next poll is queued only
    // once the previous settled, so a slow link builds no backlog.
    const tick = async () => {
      if (stopped) return;
      if (
        typeof document === "undefined" ||
        document.visibilityState !== "hidden"
      ) {
        await rescanSourcesIfStale(controller.signal);
        await poll(controller.signal);
      }
      if (stopped) return;
      const delay = pollDelayMs(mode);
      if (delay !== null) timer = window.setTimeout(tick, delay);
    };

    void tick();
    return () => {
      stopped = true;
      controller.abort();
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [mode, poll, rescanSourcesIfStale]);

  // stripped once per line so the filter matches exactly what the pane shows
  const plainLines = useMemo(() => buffer.lines.map(stripAnsi), [buffer.lines]);
  const trimmedFilter = filter.trim().toLowerCase();
  const visibleLines = useMemo(
    () =>
      trimmedFilter
        ? plainLines.filter((line) =>
            line.toLowerCase().includes(trimmedFilter),
          )
        : plainLines,
    [plainLines, trimmedFilter],
  );

  const text = useMemo(() => visibleLines.join("\n"), [visibleLines]);

  const scrollToBottom = useCallback(() => {
    const pane = paneRef.current;
    if (!pane) return;
    pane.scrollTop = pane.scrollHeight;
    pinnedRef.current = true;
    setFollowing(true);
  }, []);

  // wrap changes the scroll height without changing the text, so it re-pins too
  useEffect(() => {
    const pane = paneRef.current;
    if (pane && pinnedRef.current) pane.scrollTop = pane.scrollHeight;
  }, [text, wrap]);

  const onScroll = useCallback(() => {
    const pane = paneRef.current;
    if (!pane) return;
    // Stop chasing the bottom once the user scrolls up, so a traceback stays
    // readable while the app keeps logging.
    const pinned =
      pane.scrollHeight - pane.scrollTop - pane.clientHeight <
      FOLLOW_THRESHOLD_PX;
    if (pinned !== pinnedRef.current) {
      pinnedRef.current = pinned;
      setFollowing(pinned);
    }
  }, []);

  const revealLogsFolder = useCallback(async () => {
    setRevealing(true);
    try {
      // The selected log's own path, else the root the backend reported. Either
      // resolves a custom UNSLOTH_STUDIO_HOME; open_logs_dir hard-codes
      // ~/.unsloth/studio and cannot.
      await openLogsFolder(realpath, logRoot);
    } catch (error) {
      toast.error(t("settings.debugging.openLogsFolderFailed"), {
        description: (error as Error).message,
      });
    } finally {
      setRevealing(false);
    }
  }, [t, realpath, logRoot]);

  const downloadAllLogs = useCallback(async () => {
    setExporting(true);
    try {
      const savedPath = await exportAllLogs();
      // A path only comes back on desktop. In a browser the file is wherever
      // that browser puts downloads, which we cannot name.
      if (savedPath) {
        toast.success(
          t("settings.debugging.downloadedTo", { path: savedPath }),
          {
            action: {
              label: t("settings.debugging.showInFolder"),
              // The folder the archive went to, not the one the logs came from.
              onClick: () => {
                void revealSavedArchive(savedPath).catch((error: unknown) => {
                  toast.error(t("settings.debugging.openLogsFolderFailed"), {
                    description: (error as Error).message,
                  });
                });
              },
            },
          },
        );
      } else {
        toast.success(t("settings.debugging.downloadedToBrowser"));
      }
    } catch (error) {
      const failure =
        error instanceof LogExportError ? error.failure : "failed";
      if (failure === "outdated") {
        toast.error(t("settings.debugging.exportTooOld"));
      } else if (failure === "forbidden") {
        toast.error(t("settings.debugging.exportForbidden"));
      } else {
        toast.error(t("settings.debugging.exportFailed"), {
          description: (error as Error).message,
        });
      }
    } finally {
      setExporting(false);
    }
  }, [t]);

  const groupedSources = useMemo(() => groupByFamily(sources), [sources]);
  // the selected file's path, else the log root the footer falls back to showing
  const shownPath = realpath ?? logRoot;
  const modeLabel = (candidate: RefreshMode) =>
    t(
      candidate === "live"
        ? "settings.debugging.modeLive"
        : candidate === "3s"
          ? "settings.debugging.modeInterval"
          : "settings.debugging.modeManual",
    );

  return (
    <div className="flex flex-col gap-6">
      <SettingsSection
        title={t("settings.debugging.logSection")}
        description={t("settings.debugging.sourceHint")}
      >
        <SettingsRow label={t("settings.debugging.source")}>
          <Select
            value={sourceId ?? ""}
            onValueChange={(value) => setSourceId(value || null)}
            disabled={sources.length === 0}
          >
            <SelectTrigger
              size="sm"
              data-testid="debug-log-source"
              aria-label={t("settings.debugging.source")}
              className="max-w-[22rem] font-mono text-ui-12"
            >
              <SelectValue placeholder={t("settings.debugging.missing")} />
            </SelectTrigger>
            <SelectContent align="end">
              {groupedSources.map((group) => (
                <SelectGroup key={group.family}>
                  <SelectLabel className="font-mono uppercase tracking-wide text-ui-10">
                    {group.family}
                  </SelectLabel>
                  {group.sources.map((source) => (
                    <SelectItem
                      key={source.id}
                      value={source.id}
                      className="font-mono text-ui-12"
                    >
                      <span className="flex items-center gap-2">
                        <span>{source.label}</span>
                        {source.isCurrent ? (
                          <span className="rounded-full bg-primary/10 px-1.5 py-px font-sans text-ui-10 font-medium text-primary">
                            {t("settings.debugging.currentSession")}
                          </span>
                        ) : null}
                        <span className="font-sans text-ui-10 text-muted-foreground tabular-nums">
                          {formatBytes(source.sizeBytes)}
                        </span>
                      </span>
                    </SelectItem>
                  ))}
                </SelectGroup>
              ))}
            </SelectContent>
          </Select>
        </SettingsRow>
        <SettingsRow label={t("settings.debugging.mode")}>
          <div className="flex items-center gap-2">
            <div
              role="radiogroup"
              aria-label={t("settings.debugging.mode")}
              className="hub-tab-toggle inline-flex h-8 items-center rounded-full"
            >
              {MODES.map((candidate) => {
                const active = candidate === mode;
                return (
                  <button
                    key={candidate}
                    type="button"
                    role="radio"
                    data-testid={`debug-log-mode-${candidate}`}
                    aria-checked={active}
                    onClick={() => setMode(candidate)}
                    className={cn(
                      "relative flex h-8 items-center rounded-full px-3 text-xs font-medium transition-colors",
                      active
                        ? "hub-tab-toggle-pill text-foreground"
                        : "text-muted-foreground hover:text-foreground",
                    )}
                  >
                    <span className="relative z-10">
                      {modeLabel(candidate)}
                    </span>
                  </button>
                );
              })}
            </div>
            <Button
              size="sm"
              variant="outline"
              disabled={mode !== "manual"}
              onClick={() => {
                void refreshSources();
                void poll();
              }}
            >
              <HugeiconsIcon icon={RefreshIcon} />
              {t("settings.debugging.refreshNow")}
            </Button>
          </div>
        </SettingsRow>
      </SettingsSection>

      <div
        data-settings-label={t("settings.debugging.path")}
        className="flex flex-col gap-3"
      >
        <div className="flex flex-wrap items-center gap-2">
          <InputGroup className="h-8 min-w-48 flex-1">
            <InputGroupAddon align="inline-start">
              <HugeiconsIcon
                icon={Search01Icon}
                strokeWidth={2}
                className="size-3.5 text-muted-foreground"
              />
            </InputGroupAddon>
            <InputGroupInput
              type="search"
              value={filter}
              onChange={(event) => setFilter(event.target.value)}
              placeholder={t("settings.debugging.filterPlaceholder")}
              aria-label={t("settings.debugging.filterPlaceholder")}
              data-testid="debug-log-filter"
              className="text-sm"
            />
          </InputGroup>
          <div className="ml-auto flex items-center gap-2">
            <span
              className="text-xs text-muted-foreground tabular-nums"
              data-testid="debug-log-line-count"
            >
              {trimmedFilter
                ? t("settings.debugging.filteredLineCount", {
                    shown: visibleLines.length,
                    total: buffer.lines.length,
                  })
                : t("settings.debugging.lineCount", {
                    count: buffer.lines.length,
                  })}
            </span>
            {/* a failed read is explained by the notice below; a live chip beside it would contradict it */}
            {notice ? null : (
              <span
                className={cn(
                  "inline-flex h-5 items-center gap-1.5 rounded-full px-2 text-xs font-medium",
                  staleSession
                    ? "bg-amber-500/10 text-amber-600 dark:text-amber-400"
                    : mode === "manual"
                      ? "bg-muted text-muted-foreground"
                      : "bg-primary/10 text-primary",
                )}
              >
                <span
                  className={cn(
                    "size-1.5 rounded-full bg-current",
                    !staleSession &&
                      mode !== "manual" &&
                      "motion-safe:animate-pulse",
                  )}
                />
                {staleSession
                  ? t("settings.debugging.statusStale")
                  : mode === "manual"
                    ? t("settings.debugging.statusPaused")
                    : t("settings.debugging.statusLive")}
              </span>
            )}
            <Tooltip>
              <TooltipTrigger asChild={true}>
                <Button
                  size="icon-sm"
                  variant="ghost"
                  aria-pressed={wrap}
                  aria-label={t("settings.debugging.wrapLines")}
                  onClick={() => setWrap((previous) => !previous)}
                  className={cn(
                    wrap ? "bg-muted text-foreground" : "text-muted-foreground",
                  )}
                >
                  <HugeiconsIcon icon={TextWrapIcon} />
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                {t("settings.debugging.wrapLines")}
              </TooltipContent>
            </Tooltip>
          </div>
        </div>

        {notice || dropped || staleSession || morePending ? (
          <div className="flex flex-col gap-1.5">
            {notice ? (
              <NoticeStrip tone="warning" testId="debug-log-notice">
                {notice}
              </NoticeStrip>
            ) : null}
            {dropped ? (
              <NoticeStrip tone="warning">
                {t("settings.debugging.droppedNotice")}
              </NoticeStrip>
            ) : null}
            {staleSession ? (
              <NoticeStrip tone="warning">
                {t("settings.debugging.staleSession")}
              </NoticeStrip>
            ) : null}
            {morePending ? (
              <NoticeStrip tone="info">
                {t("settings.debugging.morePending")}
              </NoticeStrip>
            ) : null}
          </div>
        ) : null}

        <div className="relative">
          {/* One text surface, not an element per line: this repaints on every
              poll and 1000 nodes per tick is what makes a log pane feel broken. */}
          <pre
            ref={paneRef}
            onScroll={onScroll}
            data-testid="debug-log-pane"
            className={cn(
              "h-[min(26rem,45vh)] w-full overflow-auto [overflow-anchor:none] rounded-xl border border-border/60 bg-muted/20 px-3.5 py-3 font-mono text-ui-11 leading-[1.55] text-foreground/90 dark:border-transparent dark:bg-[rgb(255_255_255_/_calc(0.04*var(--contrast-wash-gain,1)))]",
              wrap ? "whitespace-pre-wrap break-words" : "whitespace-pre",
              !text && "text-muted-foreground",
            )}
          >
            {text ||
              (trimmedFilter && buffer.lines.length > 0
                ? t("settings.debugging.noMatches")
                : t("settings.debugging.empty"))}
          </pre>
          {!following && text ? (
            <Button
              size="xs"
              variant="dark"
              onClick={scrollToBottom}
              data-testid="debug-log-jump-to-latest"
              className="absolute right-3 bottom-3 rounded-full shadow-md"
            >
              <HugeiconsIcon icon={ArrowDownDoubleIcon} />
              {t("settings.debugging.jumpToLatest")}
            </Button>
          ) : null}
        </div>

        <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
          <div className="flex min-w-0 flex-1 basis-48 items-center gap-1">
            <code
              className="min-w-0 truncate font-mono text-ui-11 text-muted-foreground"
              title={shownPath ?? undefined}
            >
              {shownPath ?? "-"}
            </code>
            <Tooltip>
              <TooltipTrigger asChild={true}>
                <Button
                  size="icon-xs"
                  variant="ghost"
                  aria-label={t("settings.debugging.pathCopy")}
                  disabled={!shownPath}
                  onClick={() => shownPath && copyPath(shownPath)}
                  className="text-muted-foreground"
                >
                  <HugeiconsIcon icon={pathCopied ? Tick02Icon : Copy01Icon} />
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                {t("settings.debugging.pathCopy")}
              </TooltipContent>
            </Tooltip>
          </div>
          <div className="flex shrink-0 flex-wrap items-center justify-end gap-1">
            <Button
              size="sm"
              variant="ghost"
              onClick={() => copy(text)}
              disabled={!text}
            >
              <HugeiconsIcon icon={copied ? Tick02Icon : Copy01Icon} />
              {t("settings.debugging.copyVisible")}
            </Button>
            <Button
              size="sm"
              variant="ghost"
              data-testid="debug-log-download-all"
              aria-busy={exporting}
              disabled={exporting}
              onClick={() => void downloadAllLogs()}
            >
              <HugeiconsIcon icon={Download01Icon} />
              {exporting
                ? t("settings.debugging.downloadingAllLogs")
                : t("settings.debugging.downloadAllLogs")}
            </Button>
            <Tooltip>
              <TooltipTrigger asChild={true}>
                <button
                  type="button"
                  data-testid="debug-log-export-note"
                  aria-label={t("settings.debugging.exportMaskedNote")}
                  className="flex size-8 shrink-0 cursor-help items-center justify-center rounded-full text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                >
                  <HugeiconsIcon
                    icon={InformationCircleIcon}
                    className="size-[var(--ui-icon-size-sm)]"
                  />
                </button>
              </TooltipTrigger>
              <TooltipContent className="max-w-[300px] text-ui-11 leading-snug">
                {t("settings.debugging.exportMaskedNote")}
              </TooltipContent>
            </Tooltip>
            {isTauri ? (
              <Button
                size="sm"
                variant="ghost"
                data-testid="debug-log-open-folder"
                aria-busy={revealing}
                disabled={revealing}
                onClick={() => void revealLogsFolder()}
              >
                <HugeiconsIcon icon={FolderOpenIcon} />
                {t("settings.debugging.openLogsFolder")}
              </Button>
            ) : null}
          </div>
        </div>

        <p className="flex items-center gap-2 text-xs text-muted-foreground">
          <HugeiconsIcon icon={Shield01Icon} className="size-3.5 shrink-0" />
          {t("settings.debugging.privacyNote")}
        </p>
      </div>
    </div>
  );
}
