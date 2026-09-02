// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useCopyFeedback } from "@/features/hub/hooks/use-copy-feedback";
import { useT } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { isDownloadCancelled } from "@/lib/native-files";
import { stripAnsi } from "@/lib/strip-ansi";
import { toast } from "@/lib/toast";
import {
  Copy01Icon,
  Download01Icon,
  FolderOpenIcon,
  RefreshIcon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  type DebugLogSource,
  exportDebugLogs,
  loadDebugLog,
  loadDebugLogSources,
  openDebugLogsFolder,
} from "../api/debug-logs";
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

export function DebuggingTab() {
  const t = useT();
  const [sources, setSources] = useState<DebugLogSource[]>([]);
  const [sourceId, setSourceId] = useState<string | null>(null);
  const [mode, setMode] = useState<RefreshMode>(readStoredMode);
  const [buffer, setBuffer] = useState<LogBufferState>(EMPTY_BUFFER);
  const [notice, setNotice] = useState<string | null>(null);
  const [dropped, setDropped] = useState(false);
  // A burst larger than one response continues on the next poll, which in
  // manual mode never comes unless the user knows to ask for it.
  const [morePending, setMorePending] = useState(false);
  // File logging is off and an older session's log is still on disk, so the
  // pane shows real content that will never grow. Unsaid, a stale log is
  // indistinguishable from a live one.
  const [staleSession, setStaleSession] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [openingFolder, setOpeningFolder] = useState(false);
  const { copied, copy } = useCopyFeedback();
  const selectedRealpath = useMemo(
    () => sources.find((source) => source.id === sourceId)?.realpath ?? null,
    [sourceId, sources],
  );

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
        // The id we hold is no longer enumerated (file removed, or pushed out of
        // the per-family window). The backend sends 404 so the picker rebuilds;
        // without this the loop re-polls a dead id forever. Reselecting the
        // server's default terminates: it comes from the same walk, and
        // "nothing at all" is a 200 with a status, not another 404.
        cursorRef.current = null;
        await refreshSources({ signal, reselect: true });
        return;
      }
      setNotice((error as Error).message);
    },
    [refreshSources, t],
  );

  // The llama runner writes a NEW file per load attempt, so a list fetched at
  // mount goes stale exactly when it matters: fail a load with the tab open and
  // that failure's log is not offered.
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
    // Every notice below describes the file being left, so all of them go with
    // it. Clearing only `dropped` let a failed first read on the new source keep
    // claiming the OLD one's state, and in manual mode nothing retries: the pane
    // sat there calling a live log a frozen session.
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

  const text = useMemo(
    () => stripAnsi(buffer.lines.join("\n")),
    [buffer.lines],
  );

  useEffect(() => {
    const pane = paneRef.current;
    if (pane && pinnedRef.current) pane.scrollTop = pane.scrollHeight;
  }, [text]);

  const onScroll = useCallback(() => {
    const pane = paneRef.current;
    if (!pane) return;
    // Stop chasing the bottom once the user scrolls up, so a traceback stays
    // readable while the app keeps logging.
    pinnedRef.current =
      pane.scrollHeight - pane.scrollTop - pane.clientHeight < 40;
  }, []);

  const onExport = useCallback(async () => {
    setExporting(true);
    try {
      await exportDebugLogs();
    } catch (error) {
      if (!isDownloadCancelled(error)) {
        toast.error(t("settings.debugging.exportError"), {
          description: error instanceof Error ? error.message : String(error),
        });
      }
    } finally {
      setExporting(false);
    }
  }, [t]);

  const onOpenFolder = useCallback(async () => {
    setOpeningFolder(true);
    try {
      await openDebugLogsFolder(selectedRealpath);
    } catch (error) {
      toast.error(t("settings.debugging.openFolderError"), {
        description: error instanceof Error ? error.message : String(error),
      });
    } finally {
      setOpeningFolder(false);
    }
  }, [selectedRealpath, t]);

  return (
    <div className="flex flex-col gap-5">
      <SettingsSection
        title={t("settings.debugging.logSection")}
        description={t("settings.debugging.sourceHint")}
      >
        <div
          data-testid="debug-log-config"
          data-layout="flat"
          className="mt-4 grid gap-4"
        >
          <div className="grid gap-1.5">
            <span
              id="debug-log-source-label"
              className="text-xs font-medium text-foreground"
            >
              {t("settings.debugging.source")}
            </span>
            <Select
              value={sourceId ?? ""}
              onValueChange={(value) => setSourceId(value || null)}
              disabled={sources.length === 0}
            >
              <SelectTrigger
                id="debug-log-source"
                data-testid="debug-log-source"
                aria-labelledby="debug-log-source-label"
                className="w-full rounded-lg"
              >
                <SelectValue placeholder="-" />
              </SelectTrigger>
              <SelectContent align="start" className="max-h-72">
                {sources.map((source) => (
                  <SelectItem key={source.id} value={source.id}>
                    {source.family} / {source.label}
                    {source.isCurrent ? " *" : ""}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div
            data-testid="debug-log-location"
            className="flex min-w-0 items-center gap-3 py-1"
          >
            <div className="flex min-w-0 flex-1 items-baseline gap-3">
              <p className="shrink-0 text-ui-11 font-medium text-muted-foreground">
                {t("settings.debugging.path")}
              </p>
              <code
                className="min-w-0 flex-1 truncate text-xs text-foreground/80"
                title={selectedRealpath ?? undefined}
              >
                {selectedRealpath ?? "-"}
              </code>
            </div>
            <Button
              size="sm"
              variant="ghost"
              className="shrink-0"
              disabled={!selectedRealpath}
              aria-label={t("settings.debugging.pathCopy")}
              onClick={() => selectedRealpath && copy(selectedRealpath)}
            >
              <HugeiconsIcon
                icon={copied ? Tick02Icon : Copy01Icon}
                className="size-3.5"
              />
              {t("settings.debugging.pathCopy")}
            </Button>
          </div>

          <div
            data-settings-label={t("settings.debugging.actions")}
            data-testid="debug-log-actions"
            className="flex flex-wrap items-center justify-between gap-3 border-t border-border/50 pt-4 dark:border-white/[0.08]"
          >
            <div className="grid gap-0.5">
              <p className="text-xs font-medium text-foreground">
                {t("settings.debugging.actions")}
              </p>
              <p className="max-w-md text-ui-11 leading-relaxed text-muted-foreground">
                {t("settings.debugging.exportPrivacyNote")}
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <Button
                size="sm"
                variant="outline"
                disabled={sources.length === 0 || exporting}
                onClick={() => void onExport()}
              >
                <HugeiconsIcon icon={Download01Icon} className="size-3.5" />
                {exporting
                  ? t("settings.debugging.exportingLogs")
                  : t("settings.debugging.exportLogs")}
              </Button>
              {isTauri ? (
                <Button
                  size="sm"
                  variant="outline"
                  disabled={
                    sources.length === 0 || !selectedRealpath || openingFolder
                  }
                  onClick={() => void onOpenFolder()}
                >
                  <HugeiconsIcon icon={FolderOpenIcon} className="size-3.5" />
                  {openingFolder
                    ? t("settings.debugging.openingFolder")
                    : t("settings.debugging.openFolder")}
                </Button>
              ) : null}
            </div>
          </div>
        </div>
      </SettingsSection>

      <section
        data-settings-label={t("settings.debugging.refreshSection")}
        className="flex flex-col gap-3"
      >
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="font-heading text-base font-semibold text-foreground">
            {t("settings.debugging.refreshSection")}
          </h2>
          <div
            data-testid="debug-log-refresh-controls"
            className="flex flex-wrap items-center gap-2"
          >
            <span className="text-ui-11 font-medium text-muted-foreground">
              {t("settings.debugging.mode")}
            </span>
            <div className="flex overflow-hidden rounded-full border border-border/60 bg-muted/20 p-0.5 dark:border-transparent dark:bg-white/[0.05]">
              {MODES.map((candidate) => (
                <button
                  key={candidate}
                  type="button"
                  data-testid={`debug-log-mode-${candidate}`}
                  aria-pressed={candidate === mode}
                  onClick={() => setMode(candidate)}
                  className={
                    candidate === mode
                      ? "rounded-full bg-primary px-3 py-1 text-ui-11 font-medium text-primary-foreground shadow-sm"
                      : "rounded-full px-3 py-1 text-ui-11 text-muted-foreground transition-colors hover:text-foreground"
                  }
                >
                  {t(
                    candidate === "live"
                      ? "settings.debugging.modeLive"
                      : candidate === "3s"
                        ? "settings.debugging.modeInterval"
                        : "settings.debugging.modeManual",
                  )}
                </button>
              ))}
            </div>
            {mode === "manual" ? (
              <Button
                size="sm"
                variant="outline"
                onClick={() => {
                  void refreshSources();
                  void poll();
                }}
              >
                <HugeiconsIcon icon={RefreshIcon} className="size-3.5" />
                {t("settings.debugging.refreshNow")}
              </Button>
            ) : null}
          </div>
        </div>

        {notice ? (
          <p className="text-xs text-amber-500" data-testid="debug-log-notice">
            {notice}
          </p>
        ) : null}
        {dropped ? (
          <p className="text-xs text-amber-500">
            {t("settings.debugging.droppedNotice")}
          </p>
        ) : null}
        {morePending ? (
          <p className="text-xs text-muted-foreground">
            {t("settings.debugging.morePending")}
          </p>
        ) : null}
        {staleSession ? (
          <p className="text-xs text-amber-500">
            {t("settings.debugging.staleSession")}
          </p>
        ) : null}
        {/* One text surface, not an element per line: this repaints on every
            poll and 1000 nodes per tick is what makes a log pane feel broken. */}
        <pre
          ref={paneRef}
          onScroll={onScroll}
          data-testid="debug-log-pane"
          className="h-80 w-full overflow-auto [overflow-anchor:none] whitespace-pre-wrap break-words rounded-lg border border-border/40 bg-black/90 p-4 font-mono text-ui-11 leading-[1.5] text-emerald-200/90"
        >
          {text || t("settings.debugging.empty")}
        </pre>
        <div className="flex items-center justify-between gap-2">
          <p className="text-xs text-muted-foreground">
            {t("settings.debugging.privacyNote")}
          </p>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => copy(text)}
            disabled={!text}
          >
            <HugeiconsIcon icon={Copy01Icon} className="size-3.5" />
            {t("settings.debugging.copyVisible")}
          </Button>
        </div>
      </section>
    </div>
  );
}
