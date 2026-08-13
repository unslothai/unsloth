// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { useCopyFeedback } from "@/features/hub/hooks/use-copy-feedback";
import { useT } from "@/i18n";
import { stripAnsi } from "@/lib/strip-ansi";
import { Tick02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  type DebugLogSource,
  loadDebugLog,
  loadDebugLogSources,
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
  pollDelayMs,
  withRequestTimeout,
} from "../lib/debug-log-buffer";
import { isAbort, isLogSourceGone } from "../lib/debug-log-error";

const MODES: RefreshMode[] = ["live", "3s", "manual"];

// How often the picker rescans for log files that did not exist when the tab
// was opened. Slower than the poll, since it is a directory walk rather than a
// tail read.
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
  const [realpath, setRealpath] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [dropped, setDropped] = useState(false);
  const { copied, copy } = useCopyFeedback();

  // The buffer is read inside the poll loop, which must not restart whenever a
  // line arrives, so the cursor lives in a ref as well as in state.
  const cursorRef = useRef<string | null>(null);
  // Counts source changes, so a request that is already in flight can tell that
  // the view it was started for has since been replaced.
  const selectionRef = useRef(0);
  // The selection a request is in flight for, rather than a bare flag: a poll
  // for the newly picked source must not be swallowed by a slow read of the one
  // the user just left, or the new pane stays empty (in manual mode, for good).
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
        // Bounded like the tail read: the poll loop awaits this before polling,
        // and the failure recovery awaits it inside the poll's own catch, so an
        // unanswered /sources would otherwise freeze both.
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
        // The log read below reports the real reason; a failed source list just
        // means the picker stays empty.
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
      if (isLogSourceGone(error)) {
        // The id we hold is no longer enumerated: the file was removed, or a
        // run of failed load attempts pushed it out of the per-family window.
        // The backend sends 404 precisely so the picker rebuilds itself;
        // without this the loop re-polls a dead id once a second forever.
        // Reselecting the server's default also ends the loop, because that is
        // recomputed from the same walk, and "nothing at all" is a 200 with a
        // status rather than another 404.
        cursorRef.current = null;
        await refreshSources({ signal, reselect: true });
        return;
      }
      setNotice((error as Error).message);
    },
    [refreshSources],
  );

  // The llama runner writes a NEW file per load attempt, so a list fetched once
  // at mount goes stale exactly when it matters: open the tab, then fail a
  // load, and the log for that failure is not offered. Rescanning is three
  // directory listings, so it runs on its own slower cadence than the tail
  // poll.
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
      // A request that never settles would otherwise pin inFlightRef forever:
      // the timer keeps firing, every poll returns at the guard above, and the
      // pane freezes with no error because the catch never runs. A suspended
      // laptop or a dropped tunnel is enough.
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
        // A manual refresh carries no abort signal, so one still in flight when
        // the user switches source would otherwise land the old file's lines,
        // cursor and path under the new pick.
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
    setDropped(false);
  }, [sourceId]);

  useEffect(() => {
    const controller = new AbortController();
    let timer: number | undefined;
    let stopped = false;

    // A self-scheduling timeout, not setInterval: the next poll is only queued
    // once the previous one has settled, so a slow link (a remote tunnel) can
    // never build a backlog of overlapping requests.
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
    // Stop chasing the bottom once the user scrolls up: reading a traceback
    // while the app is still logging was the whole complaint.
    pinnedRef.current =
      pane.scrollHeight - pane.scrollTop - pane.clientHeight < 40;
  }, []);

  return (
    <div className="flex flex-col gap-6">
      <SettingsSection
        title={t("settings.debugging.logSection")}
        description={t("settings.debugging.sourceHint")}
      >
        <SettingsRow label={t("settings.debugging.source")}>
          <select
            data-testid="debug-log-source"
            className="max-w-[22rem] rounded-md border border-border/60 bg-background px-2 py-1 text-xs"
            value={sourceId ?? ""}
            onChange={(event) => setSourceId(event.target.value || null)}
          >
            {sources.map((source) => (
              <option key={source.id} value={source.id}>
                {source.family} / {source.label}
                {source.isCurrent ? " *" : ""}
              </option>
            ))}
          </select>
        </SettingsRow>
        <SettingsRow label={t("settings.debugging.path")} alignTop={true}>
          <div className="flex items-center gap-2">
            <code className="max-w-[22rem] truncate text-ui-11 text-muted-foreground">
              {realpath ?? "-"}
            </code>
            <Button
              size="sm"
              variant="ghost"
              disabled={!realpath}
              onClick={() => realpath && copy(realpath)}
            >
              {copied ? (
                <HugeiconsIcon icon={Tick02Icon} className="size-3.5" />
              ) : (
                t("settings.debugging.pathCopy")
              )}
            </Button>
          </div>
        </SettingsRow>
      </SettingsSection>

      <SettingsSection title={t("settings.debugging.refreshSection")}>
        <SettingsRow label={t("settings.debugging.mode")}>
          <div className="flex items-center gap-2">
            <div className="flex overflow-hidden rounded-md border border-border/60">
              {MODES.map((candidate) => (
                <button
                  key={candidate}
                  type="button"
                  data-testid={`debug-log-mode-${candidate}`}
                  onClick={() => setMode(candidate)}
                  className={
                    candidate === mode
                      ? "bg-primary px-2 py-1 text-ui-11 text-primary-foreground"
                      : "px-2 py-1 text-ui-11 text-muted-foreground"
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
            <Button
              size="sm"
              variant="outline"
              disabled={mode !== "manual"}
              onClick={() => {
                void refreshSources();
                void poll();
              }}
            >
              {t("settings.debugging.refreshNow")}
            </Button>
          </div>
        </SettingsRow>
      </SettingsSection>

      <div className="flex flex-col gap-2">
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
        {/* One text surface, not an element per line: this repaints on every
            poll and 1000 nodes per tick is what makes a log pane feel broken. */}
        <pre
          ref={paneRef}
          onScroll={onScroll}
          data-testid="debug-log-pane"
          className="h-72 w-full overflow-auto [overflow-anchor:none] whitespace-pre-wrap break-words rounded-lg border border-border/40 bg-black/85 p-3 font-mono text-ui-11 leading-[1.45] text-emerald-200/90"
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
            {t("settings.debugging.copyVisible")}
          </Button>
        </div>
      </div>
    </div>
  );
}
