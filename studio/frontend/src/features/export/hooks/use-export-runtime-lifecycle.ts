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

    const pollStatus = async () => {
      if (!hasAuthToken()) return;
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
      }
    };

    // The base subscribe fires on every change, so no subscribeWithSelector is needed.
    let prevExporting = store.getState().isExporting;
    const unsubscribe = store.subscribe((state) => {
      if (state.isExporting === prevExporting) return;
      prevExporting = state.isExporting;
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

    const statusTimer = setInterval(() => {
      void pollStatus();
    }, STATUS_POLL_INTERVAL_MS);

    return () => {
      disposed = true;
      clearInterval(statusTimer);
      unsubscribe();
      stopStream();
    };
  }, []);
}
