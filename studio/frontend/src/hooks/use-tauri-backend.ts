// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState, useCallback, useRef } from "react";
import { isTauri, setApiBase } from "@/lib/api-base";

type BackendStatus =
  | "checking"
  | "not-installed"
  | "installing"
  | "install-error"
  | "needs-elevation"
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
  // Guard against React Strict Mode double-mount
  const mountedRef = useRef(false);
  // Track the discovered port from server-port event
  const portRef = useRef<number | null>(null);
  const [currentStepIndex, setCurrentStepIndex] = useState(-1);
  const [elevationPackages, setElevationPackages] = useState<string[]>([]);
  const [progressDetail, setProgressDetail] = useState<string | null>(null);
  // Track seen step names to deduplicate (Strict Mode, event replay, etc.)
  const seenStepsRef = useRef(new Set<string>());

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
    setCurrentStepIndex(-1);
    setProgressDetail(null);
    seenStepsRef.current.clear();
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
    setCurrentStepIndex(-1);
    setProgressDetail(null);
    setElevationPackages([]);
    seenStepsRef.current.clear();
    checkInstallAndStart();
  }, []);

  const retryInstall = useCallback(() => {
    setError(null);
    setLogs([]);
    setStatus("not-installed");
  }, []);

  const approveElevation = useCallback(async () => {
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      await invoke("install_system_packages", { packages: elevationPackages });
      // Packages installed successfully, retry the full install
      setCurrentStepIndex(-1);
      setProgressDetail(null);
      await startInstall();
    } catch (e) {
      setStatus("install-error");
      setError(String(e));
    }
  }, [elevationPackages]);

  // Initial check on mount (guarded against Strict Mode double-mount)
  useEffect(() => {
    if (mountedRef.current) return;
    mountedRef.current = true;

    if (!isTauri) {
      setStatus("running");
      return;
    }
    // Close native splash immediately when React renders
    import("@tauri-apps/api/core").then(({ invoke }) => {
      invoke("close_splashscreen");
    });
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
        setCurrentStepIndex(999); // all steps done
      }).then((u) => cleanup.push(u));

      listen<string>("install-step", (e) => {
        const stepName = e.payload;
        if (seenStepsRef.current.has(stepName)) return; // deduplicate
        seenStepsRef.current.add(stepName);
        setCurrentStepIndex((prev) => prev + 1);
        setProgressDetail(null);
      }).then((u) => cleanup.push(u));

      listen<string[]>("install-needs-elevation", (e) => {
        setElevationPackages(e.payload);
        setStatus("needs-elevation");
      }).then((u) => cleanup.push(u));

      listen<string>("install-progress-detail", (e) => {
        setProgressDetail(e.payload);
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
    status, logs, error,
    currentStepIndex, progressDetail, elevationPackages,
    startServer, stopServer, startInstall,
    retry, retryInstall, approveElevation,
  };
}
