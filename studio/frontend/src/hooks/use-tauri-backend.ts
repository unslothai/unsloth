// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState, useCallback, useRef } from "react";
import { isTauri, setApiBase } from "@/lib/api-base";

type BackendStatus =
  | "checking"
  | "not-installed"
  | "installing"
  | "install-error"
  | "starting"
  | "running"
  | "stopped"
  | "error";

export function useTauriBackend() {
  const [status, setStatus] = useState<BackendStatus>("checking");
  const statusRef = useRef<BackendStatus>(status);
  const [logs, setLogs] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  // Guard against double startServer calls
  const startingRef = useRef(false);
  // Track the discovered port from server-port event
  const portRef = useRef<number | null>(null);

  // Keep ref in sync for event listener closures
  useEffect(() => {
    statusRef.current = status;
  }, [status]);

  async function checkInstallAndStart() {
    const { invoke } = await import("@tauri-apps/api/core");

    // Check for existing running server first
    const existingPort = await invoke<number | null>("find_existing_server");
    if (existingPort) {
      setApiBase(existingPort);
      setStatus("running");
      return;
    }

    const installed = await invoke<boolean>("check_install_status");
    if (!installed) {
      setStatus("not-installed");
      return;
    }
    setStatus("starting");
    await startServer();
  }

  async function startServer() {
    // Prevent double-start race condition
    if (startingRef.current) return;
    startingRef.current = true;

    try {
      const { invoke } = await import("@tauri-apps/api/core");
      await invoke("start_server", { port: 8888 });

      // Wait for the server-port event (usually arrives within ~2s) before
      // falling back to scanning the full port range. This avoids 20 unnecessary
      // health checks per poll iteration.
      for (let i = 0; i < 120; i++) {
        if (portRef.current) {
          const healthy = await invoke<boolean>("check_health", {
            port: portRef.current,
          });
          if (healthy) {
            setApiBase(portRef.current);
            setStatus("running");
            startingRef.current = false;
            return;
          }
        } else if (i >= 4) {
          // Only scan the full range after ~2s (4 x 500ms) if the server-port
          // event hasn't arrived yet. This is the rare fallback path.
          for (let p = 8888; p <= 8907; p++) {
            const healthy = await invoke<boolean>("check_health", { port: p });
            if (healthy) {
              setApiBase(p);
              setStatus("running");
              startingRef.current = false;
              return;
            }
          }
        }
        await new Promise((r) => setTimeout(r, 500));
      }
      setStatus("error");
      setError("Backend did not start within 60 seconds");
    } catch (e) {
      // "Backend is already running" is not an error — just start polling
      const msg = String(e);
      if (msg.includes("already running")) {
        // Already started by another path, just wait for health
        startingRef.current = false;
        return;
      }
      setStatus("error");
      setError(msg);
    }
    startingRef.current = false;
  }

  async function stopServer() {
    const { invoke } = await import("@tauri-apps/api/core");
    await invoke("stop_server");
    startingRef.current = false;
    setStatus("stopped");
  }

  async function startInstall() {
    setStatus("installing");
    setLogs([]);
    setError(null);
    const { invoke } = await import("@tauri-apps/api/core");
    try {
      await invoke("start_install");
      // Install completed — this is the ONLY path that starts the server
      // after install. The install-complete event listener does NOT call
      // startServer() to avoid a double-start race condition.
      setStatus("starting");
      await startServer();
    } catch (e) {
      setStatus("install-error");
      setError(String(e));
    }
  }

  const retry = useCallback(() => {
    setError(null);
    setLogs([]);
    startingRef.current = false;
    portRef.current = null;
    checkInstallAndStart();
  }, []);

  const retryInstall = useCallback(() => {
    setError(null);
    setLogs([]);
    setStatus("not-installed");
  }, []);

  // Initial check on mount
  useEffect(() => {
    if (!isTauri) {
      setStatus("running");
      return;
    }
    checkInstallAndStart();
  }, []);

  // Listen for Tauri events
  useEffect(() => {
    if (!isTauri) return;
    const cleanup: (() => void)[] = [];

    import("@tauri-apps/api/event").then(({ listen }) => {
      listen<string>("install-progress", (e) => {
        setLogs((prev) => [...prev.slice(-499), e.payload]);
      }).then((u) => cleanup.push(u));

      // install-complete is informational only — does NOT trigger startServer.
      // The invoke("start_install") success path handles that to avoid races.
      listen<void>("install-complete", () => {
        // Just update logs; the invoke path handles the transition
      }).then((u) => cleanup.push(u));

      listen<string>("install-failed", (e) => {
        setError(e.payload);
        setStatus("install-error");
      }).then((u) => cleanup.push(u));

      listen<number>("server-port", (e) => {
        portRef.current = e.payload;
        setApiBase(e.payload);
      }).then((u) => cleanup.push(u));

      listen<void>("server-crashed", () => {
        startingRef.current = false;
        setStatus("error");
        setError("Server stopped unexpectedly");
      }).then((u) => cleanup.push(u));

      listen<string>("server-log", (e) => {
        setLogs((prev) => [...prev.slice(-499), e.payload]);
      }).then((u) => cleanup.push(u));

      listen<void>("tray-toggle-server", () => {
        if (statusRef.current === "running") {
          stopServer();
        } else if (
          statusRef.current === "stopped" ||
          statusRef.current === "error"
        ) {
          retry();
        }
      }).then((u) => cleanup.push(u));
    });

    return () => {
      cleanup.forEach((fn) => fn());
    };
  }, []);

  return {
    status,
    logs,
    error,
    startServer,
    stopServer,
    startInstall,
    retry,
    retryInstall,
  };
}
