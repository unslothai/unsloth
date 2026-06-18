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
// JSON log poll cadence. The SSE stream is the low-latency path on localhost,
// but Cloudflare quick tunnels (`--secure`) buffer `text/event-stream`, so this
// poll is the transport that actually delivers logs over the tunnel. Short
// JSON responses are never buffered, so this works through any proxy.
const LOG_POLL_INTERVAL_MS = 750;

/**
 * Global export runtime driver, mounted once at the app root so it runs on every
 * route (the inline export panel lives on /export, but the run must stay live and
 * visible from any tab). It:
 *   - hydrates `is_export_active` from the backend on mount / reload,
 *   - streams the worker log SSE into the store while a run is active, and
 *   - keeps that stream alive across the load -> export phase boundary so a
 *     transient `complete` between per-phase POSTs does not strand the panel on
 *     "Waiting for worker output..." (the premature-complete bug).
 *
 * The actual export sequence is driven by the store's `runExport` action (a
 * detached promise), so it survives navigation independently of this hook.
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

    // ── JSON log polling (tunnel-safe fallback) ──────────────────────────
    // Runs for the whole active run, in parallel with the SSE stream. On
    // localhost the SSE delivers first and these polls are de-duped away by
    // seq; over a Cloudflare tunnel the SSE is buffered and these polls are
    // what actually fill the log panel. A successful poll marks the stream
    // "connected" so the panel shows "streaming" rather than "connecting...".
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
            // `complete` / `error` end this connection (streamExportLogs returns);
            // the finally block reconnects while the run is still in flight.
          },
        });
      } catch {
        // fetch-level failure: fall through to the reconnect below.
      } finally {
        openingStream = false;
        if (streamController === controller) {
          streamController = null;
        }
        // Do NOT clear `connected` here: over a Cloudflare tunnel the SSE drops
        // and reconnects repeatedly (buffered / premature complete), which used
        // to flap the indicator back to "connecting...". The log poll owns the
        // connected flag for the duration of the run; stopStream clears it when
        // the run actually ends.

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

    // React to isExporting flipping (run start / terminal) without needing the
    // subscribeWithSelector middleware: the base subscribe fires on every change.
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
