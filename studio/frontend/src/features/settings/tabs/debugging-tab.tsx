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
import { ChevronDownDoubleStandardIcon } from "@/lib/chevron-icons";
import { stripAnsi } from "@/lib/strip-ansi";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import {
  Alert02Icon,
  Copy01Icon,
  Download01Icon,
  FolderOpenIcon,
  InformationCircleIcon,
  Refresh01Icon,
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
import {
  NO_PENDING_LOG_REQUEST,
  pendingLogRequestKey,
  useSettingsDialogStore,
} from "../stores/settings-dialog-store";

const MODES: RefreshMode[] = ["live", "3s", "manual"];

// Slower than the poll: a directory walk rather than a tail read.
const SOURCE_RESCAN_MS = 10_000;

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
        <HugeiconsIcon
          strokeWidth={1.75}
          icon={Alert02Icon}
          className="mt-px size-3.5 shrink-0"
        />
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
  const [logRoot, setLogRoot] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [dropped, setDropped] = useState(false);
  // In manual mode the continuation poll never comes unless the user asks.
  const [morePending, setMorePending] = useState(false);
  // File logging is off but an old log is on disk; say so, or it looks live.
  const [staleSession, setStaleSession] = useState(false);
  // Per-button state so the slow export does not lock out "Show in folder".
  const [exporting, setExporting] = useState(false);
  const [revealing, setRevealing] = useState(false);
  const [filter, setFilter] = useState("");
  const [wrap, setWrap] = useState(true);
  // Mirrors pinnedRef for the jump button; the ref stays the source of truth so scrolling
  // does not re-render the pane.
  const [following, setFollowing] = useState(true);
  const { copied, copy } = useCopyFeedback();
  const { copied: pathCopied, copy: copyPath } = useCopyFeedback();

  // A ref too, so the poll loop does not restart per line.
  const cursorRef = useRef<string | null>(null);
  const selectionRef = useRef(0);
  // The selection in flight, not a flag, so a slow read of the old source cannot swallow the new poll.
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

  // A response older than the last selecting one never overrides it.
  const sourceFetchSeqRef = useRef(0);
  const appliedSourceFetchRef = useRef(0);

  const refreshSources = useCallback(
    async (options: { signal?: AbortSignal; reselect?: boolean } = {}) => {
      const seq = ++sourceFetchSeqRef.current;
      try {
        // Bounded: the poll loop and its recovery both await /sources.
        const requestedFor = pendingLogRequestKey(
          useSettingsDialogStore.getState(),
        );
        const pendingPath =
          useSettingsDialogStore.getState().logSourcePathRequested;
        const result = await withRequestTimeout(
          (signal) => loadDebugLogSources(signal, pendingPath),
          REQUEST_TIMEOUT_MS,
          options.signal,
        );
        if (seq < appliedSourceFetchRef.current) return;
        setSources(result.sources);
        setLogRoot(result.logRoot);
        const dialog = useSettingsDialogStore.getState();
        const requested = dialog.logFamilyRequested;
        const byPath = result.matchedSourceId
          ? result.sources.find(
              (source) => source.id === result.matchedSourceId,
            )
          : undefined;
        const fromFailure =
          byPath ??
          (requested
            ? result.sources.find((source) => source.family === requested)
            : undefined);
        const stillTheSameRequest =
          pendingLogRequestKey(dialog) === requestedFor;
        if (fromFailure && stillTheSameRequest)
          useSettingsDialogStore.getState().consumeLogFamilyRequest();
        if (fromFailure && !stillTheSameRequest) return;
        appliedSourceFetchRef.current = seq;
        setSourceId((current) =>
          fromFailure
            ? fromFailure.id
            : options.reselect
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

  const pendingLogRequest = useSettingsDialogStore(pendingLogRequestKey);
  useEffect(() => {
    if (pendingLogRequest === NO_PENDING_LOG_REQUEST) return;
    const controller = new AbortController();
    void refreshSources({ signal: controller.signal });
    return () => controller.abort();
  }, [pendingLogRequest, refreshSources]);

  const onPollFailed = useCallback(
    async (error: unknown, signal?: AbortSignal) => {
      if (isAbort(error)) return;
      if (isRequestTimeout(error)) {
        // The backstop duration is internal; show the consequence.
        setNotice(t("settings.debugging.timeout"));
        return;
      }
      if (isLogSourceGone(error)) {
        // 404 means the id is no longer enumerated; reset so the loop does not poll a dead id.
        // Reselecting the default terminates, since "nothing at all" is a 200.
        cursorRef.current = null;
        await refreshSources({ signal, reselect: true });
        return;
      }
      setNotice((error as Error).message);
    },
    [refreshSources, t],
  );

  // The llama runner writes a new file per load attempt, so the mount-time list goes stale.
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
      // Without the timeout a hung request pins inFlightRef and the pane freezes silently.
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
        // A manual refresh has no abort signal and could land the old file's lines under the new pick.
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
        if (inFlightRef.current === selection) inFlightRef.current = null;
      }
    },
    [onPollFailed, sourceId, t],
  );

  useEffect(() => {
    selectionRef.current += 1;
    cursorRef.current = null;
    setBuffer(EMPTY_BUFFER);
    setRealpath(null);
    // Every notice describes the file being left, so reset all of them; manual mode never retries.
    setDropped(false);
    setMorePending(false);
    setStaleSession(false);
    setNotice(null);
  }, [sourceId]);

  useEffect(() => {
    const controller = new AbortController();
    let timer: number | undefined;
    let stopped = false;

    // Self-scheduling so a slow link builds no backlog.
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
      await openLogsFolder(logRoot, realpath);
    } catch (error) {
      toast.error(t("settings.debugging.openLogsFolderFailed"), {
        description: (error as Error).message,
      });
    } finally {
      setRevealing(false);
    }
  }, [t, logRoot, realpath]);

  const downloadAllLogs = useCallback(async () => {
    setExporting(true);
    try {
      const savedPath = await exportAllLogs();
      // Only desktop returns a path; a browser's download location is unknown.
      if (savedPath) {
        toast.success(
          t("settings.debugging.downloadedTo", { path: savedPath }),
          {
            action: {
              label: t("settings.debugging.showInFolder"),
              // The archive's folder, not the logs' folder.
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
    <div className="settings-page">
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
              className="max-w-[calc(22rem*var(--ui-space-scale,1))] font-mono text-ui-12"
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
                      <span className="flex min-w-0 items-center gap-2">
                        <span className="min-w-0 truncate">{source.label}</span>
                        {source.isCurrent ? (
                          <span className="shrink-0 rounded-full bg-primary/10 px-1.5 py-px font-sans text-ui-10 font-medium text-primary">
                            {t("settings.debugging.currentSession")}
                          </span>
                        ) : null}
                        <span className="shrink-0 font-sans text-ui-10 text-muted-foreground tabular-nums">
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
              <HugeiconsIcon strokeWidth={1.75} icon={Refresh01Icon} />
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
                  <HugeiconsIcon strokeWidth={1.75} icon={TextWrapIcon} />
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
              "h-[min(26rem,45vh)] w-full overflow-auto [overflow-anchor:none] scroll-rounded rounded-xl border border-border/60 bg-muted/20 px-3.5 py-3 font-mono text-ui-11 leading-[1.55] text-foreground/90 dark:border-transparent dark:bg-[rgb(255_255_255_/_calc(0.04*var(--contrast-wash-gain,1)))]",
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
              <HugeiconsIcon
                strokeWidth={1.75}
                icon={ChevronDownDoubleStandardIcon}
              />
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
                  <HugeiconsIcon
                    strokeWidth={1.75}
                    icon={pathCopied ? Tick02Icon : Copy01Icon}
                  />
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                {t("settings.debugging.pathCopy")}
              </TooltipContent>
            </Tooltip>
          </div>
          <div className="flex max-w-full shrink-0 flex-wrap items-center justify-end gap-1">
            <Button
              size="sm"
              variant="ghost"
              onClick={() => copy(text)}
              disabled={!text}
            >
              <HugeiconsIcon
                strokeWidth={1.75}
                icon={copied ? Tick02Icon : Copy01Icon}
              />
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
              <HugeiconsIcon strokeWidth={1.75} icon={Download01Icon} />
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
                    strokeWidth={1.75}
                    icon={InformationCircleIcon}
                    className="size-[var(--ui-icon-size-sm)]"
                  />
                </button>
              </TooltipTrigger>
              <TooltipContent className="max-w-[calc(300px*var(--ui-space-scale,1))] text-ui-11 leading-snug">
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
                <HugeiconsIcon strokeWidth={1.75} icon={FolderOpenIcon} />
                {t("settings.debugging.openLogsFolder")}
              </Button>
            ) : null}
          </div>
        </div>

        <p className="flex items-center gap-2 text-xs text-muted-foreground">
          <HugeiconsIcon
            strokeWidth={1.75}
            icon={Shield01Icon}
            className="size-3.5 shrink-0"
          />
          {t("settings.debugging.privacyNote")}
        </p>
      </div>
    </div>
  );
}
