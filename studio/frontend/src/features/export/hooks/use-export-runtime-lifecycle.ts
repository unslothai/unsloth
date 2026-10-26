// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { hasAuthToken } from "@/features/auth";
import { useEffect } from "react";
import {
  fetchExportLogs,
  getExportStatus,
  streamExportLogs,
} from "../api/export-api";
import { useExportRuntimeStore } from "../stores/export-runtime-store";

const STATUS_POLL_INTERVAL_MS = 5000;
// Idle: only catches a run started in another tab, so slow and skipped while hidden.
const IDLE_STATUS_POLL_INTERVAL_MS = 30_000;
const STREAM_RECONNECT_DELAY_MS = 600;
const LOG_POLL_INTERVAL_MS = 750;

/**
 * Global export driver mounted at the app root: hydrates status, streams logs, and keeps
 * the stream alive across the load -> export boundary. The sequence itself is runExport.
 */
export function useExportRuntimeLifecycle(): void {
  useEffect(() => {
    let disposed = false;
    let openingStream = false;
    let streamController: AbortController | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let logPolling = false;
    let logPollTimer: ReturnType<typeof setTimeout> | null = null;
    let statusPolling = false;
    let statusTimer: ReturnType<typeof setTimeout> | null = null;

    const store = useExportRuntimeStore;

    const clearReconnect = () => {
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
    };

    const stopStream = () => {
      clearReconnect();
      if (streamController) {
        streamController.abort();
        streamController = null;
      }
      stopLogPolling();
      store.getState().setConnected(false);
    };

    // ── JSON log polling (tunnel-safe fallback) ────────────────────────── Runs for the whole
    // Polling runs alongside SSE; over a Cloudflare tunnel the SSE is buffered, so polls fill
    // the log. A successful poll marks the stream connected.
    const pollLogsOnce = async () => {
      if (disposed || !store.getState().isExporting) return;
      try {
        const res = await fetchExportLogs(store.getState().lastSeq);
        if (disposed) return;
        store.getState().setConnected(true);
        if (res.entries.length > 0) {
          store.getState().appendLogs(res.entries);
        }
      } catch {
        // Transient poll failure: the next poll (or the SSE) will catch up.
      }
    };

    const logPollLoop = async () => {
      if (disposed || !logPolling) return;
      await pollLogsOnce();
      if (disposed || !logPolling) return;
      logPollTimer = setTimeout(() => {
        void logPollLoop();
      }, LOG_POLL_INTERVAL_MS);
    };

    const startLogPolling = () => {
      if (logPolling || disposed) return;
      logPolling = true;
      void logPollLoop();
    };

    function stopLogPolling() {
      logPolling = false;
      if (logPollTimer) {
        clearTimeout(logPollTimer);
        logPollTimer = null;
      }
    }

    const ensureStream = async () => {
      if (
        disposed ||
        openingStream ||
        streamController ||
        !store.getState().isExporting
      ) {
        return;
      }

      clearReconnect();
      openingStream = true;
      const controller = new AbortController();
      streamController = controller;

      try {
        await streamExportLogs({
          signal: controller.signal,
          since: store.getState().lastSeq,
          onOpen: () => store.getState().setConnected(true),
          onEvent: (event) => {
            if (event.event === "log" && event.entry) {
              store.getState().appendLog(event.entry, event.id ?? undefined);
            }
          },
        });
      } catch {
        // fetch-level failure: fall through to the reconnect below.
      } finally {
        openingStream = false;
        if (streamController === controller) {
          streamController = null;
        }
        // Do not clear `connected`: tunnel SSE reconnects would flap it. The poll owns the flag.

        if (
          !disposed &&
          !controller.signal.aborted &&
          store.getState().isExporting
        ) {
          reconnectTimer = setTimeout(() => {
            void ensureStream();
          }, STREAM_RECONNECT_DELAY_MS);
        }
      }
    };

    const scheduleStatusPoll = () => {
      if (disposed) return;
      if (statusTimer) clearTimeout(statusTimer);
      const { isExporting, backendActive } = store.getState();
      const active = isExporting || backendActive;
      statusTimer = setTimeout(
        () => {
          statusTimer = null;
          if (!active && document.hidden) return;
          void pollStatus();
        },
        active ? STATUS_POLL_INTERVAL_MS : IDLE_STATUS_POLL_INTERVAL_MS,
      );
    };

    const pollStatus = async () => {
      if (statusPolling) return;
      if (!hasAuthToken()) {
        scheduleStatusPoll();
        return;
      }
      statusPolling = true;
      try {
        const status = await getExportStatus();
        if (disposed) return;
        store.getState().applyBackendStatus(status);
        if (store.getState().isExporting) {
          void ensureStream();
          startLogPolling();
        }
      } catch {
        // ignore transient status failures
      } finally {
        statusPolling = false;
        scheduleStatusPoll();
      }
    };

    const onVisibilityChange = () => {
      if (document.hidden || statusPolling || disposed) return;
      if (statusTimer) clearTimeout(statusTimer);
      statusTimer = null;
      void pollStatus();
    };

    // React to isExporting flipping (run start / terminal) without needing the
    // subscribeWithSelector middleware: the base subscribe fires on every change.
    let prevExporting = store.getState().isExporting;
    const unsubscribe = store.subscribe((state) => {
      if (state.isExporting === prevExporting) return;
      prevExporting = state.isExporting;
      if (!statusPolling) scheduleStatusPoll();
      if (state.isExporting) {
        void ensureStream();
        startLogPolling();
      } else {
        stopStream();
      }
    });

    void pollStatus();
    if (store.getState().isExporting) {
      void ensureStream();
      startLogPolling();
    }

    document.addEventListener("visibilitychange", onVisibilityChange);

    return () => {
      disposed = true;
      if (statusTimer) clearTimeout(statusTimer);
      document.removeEventListener("visibilitychange", onVisibilityChange);
      unsubscribe();
      stopStream();
    };
  }, []);
}
