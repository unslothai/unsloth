// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState, useCallback, useRef } from "react";
import { isTauri, setApiBase } from "@/lib/api-base";
import {
  clearTauriAuthFailure,
  getTauriAuthFailure,
} from "@/features/auth";

export type BackendStatus =
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
  // True when we attached to a server we didn't spawn (can't stop it)
  const [isExternalServer, setIsExternalServer] = useState(false);
  const externalPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const externalPollAbortedRef = useRef(false);
  const authFailureRef = useRef<string | null>(getTauriAuthFailure());

  function setBackendStatus(nextStatus: BackendStatus) {
    if (authFailureRef.current) return;
    setStatus(nextStatus);
  }

  function setBackendError(
    nextError: string,
    nextStatus: BackendStatus = "error",
  ) {
    if (authFailureRef.current) return;
    setStatus(nextStatus);
    setError(nextError);
  }

  function clearBackendError() {
    if (authFailureRef.current) return;
    setError(null);
  }

  function setRunningStatus() {
    setBackendStatus("running");
  }

  function setAuthFailure(detail: string) {
    authFailureRef.current = detail;
    setStatus("error");
    setError(detail);
  }

  function clearAuthFailure() {
    authFailureRef.current = null;
    clearTauriAuthFailure();
  }

  function stopExternalServerPoll() {
    externalPollAbortedRef.current = true;
    if (externalPollRef.current) {
      clearInterval(externalPollRef.current);
      externalPollRef.current = null;
    }
  }

  function startExternalServerPoll(port: number) {
    stopExternalServerPoll();
    externalPollAbortedRef.current = false;
    let failures = 0;
    externalPollRef.current = setInterval(async () => {
      if (externalPollAbortedRef.current) return;
      try {
        const { invoke } = await import("@tauri-apps/api/core");
        const healthy = await invoke<boolean>("check_health", { port });
        if (externalPollAbortedRef.current) return;
        if (healthy) {
          failures = 0;
        } else {
          failures++;
        }
      } catch {
        if (externalPollAbortedRef.current) return;
        failures++;
      }
      if (failures >= 3) {
        stopExternalServerPoll();
        setIsExternalServer(false);
        setBackendError("External server is no longer responding");
      }
    }, 15_000);
  }

  // Keep ref in sync for event listener closures
  useEffect(() => {
    statusRef.current = status;
  }, [status]);

  async function checkInstallAndStart() {
    try {
      const { invoke } = await import("@tauri-apps/api/core");

      // Check for existing running server first
      const existingPort = await invoke<number | null>("find_existing_server");
      if (existingPort) {
        setApiBase(existingPort);
        setIsExternalServer(true);
        setRunningStatus();
        // Monitor external server — we can't get Rust-side crash events for it
        startExternalServerPoll(existingPort);
        return;
      }

      const installed = await invoke<boolean>("check_install_status");
      if (!installed) {
        setBackendStatus("not-installed");
        return;
      }
      setBackendStatus("starting");
      await startServer();
    } catch (e) {
      setBackendError(String(e));
    }
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
            setRunningStatus();
            startingRef.current = false;
            return;
          }
        } else if (i >= 4) {
          // Only scan the full range after ~2s (4 x 500ms) if the server-port
          // event hasn't arrived yet. This is the rare fallback path.
          for (let p = 8888; p <= 8908; p++) {
            const healthy = await invoke<boolean>("check_health", { port: p });
            if (healthy) {
              setApiBase(p);
              setRunningStatus();
              startingRef.current = false;
              return;
            }
          }
        }
        await new Promise((r) => setTimeout(r, 500));
      }
      const message = !portRef.current
        ? "Could not start the server — all ports 8888–8908 may be in use. " +
          "Close other Unsloth instances or free a port and try again."
        : "Server started but is not responding. Check the logs for details.";
      setBackendError(message);
    } catch (e) {
      const msg = String(e);
      if (msg.includes("already running")) {
        // Backend exists but start_server rejected. Give Rust time to reap
        // the old process (stdout reader sets child=None), then retry via
        // find_existing_server which will attach to the running backend.
        startingRef.current = false;
        setBackendStatus("starting");
        await new Promise((r) => setTimeout(r, 2000));
        checkInstallAndStart();
        return;
      }
      setBackendError(msg);
    }
    startingRef.current = false;
  }

  async function stopServer() {
    if (isExternalServer) {
      // We attached to a server we didn't spawn — can't kill it,
      // just disconnect the UI.
      startingRef.current = false;
      setIsExternalServer(false);
      stopExternalServerPoll();
      setBackendStatus("stopped");
      return;
    }
    const { invoke } = await import("@tauri-apps/api/core");
    await invoke("stop_server");
    startingRef.current = false;
    setBackendStatus("stopped");
  }

  async function startInstall() {
    setCurrentStepIndex(-1);
    setProgressDetail(null);
    seenStepsRef.current.clear();
    setBackendStatus("installing");
    setLogs([]);
    clearBackendError();
    const { invoke } = await import("@tauri-apps/api/core");
    try {
      await invoke("start_install");
      // Install completed — this is the ONLY path that starts the server
      // after install. The install-complete event listener does NOT call
      // startServer() to avoid a double-start race condition.
      setBackendStatus("starting");
      await startServer();
    } catch (e) {
      const msg = String(e);
      // NEEDS_ELEVATION is not a real error — the Rust side also emits
      // install-needs-elevation which sets needs-elevation status.
      // Don't race with it by setting install-error here.
      if (msg.includes("NEEDS_ELEVATION")) return;
      setBackendError(msg, "install-error");
    }
  }

  const retry = useCallback(() => {
    clearAuthFailure();
    setError(null);
    setLogs([]);
    startingRef.current = false;
    portRef.current = null;
    setCurrentStepIndex(-1);
    setProgressDetail(null);
    setElevationPackages([]);
    setIsExternalServer(false);
    stopExternalServerPoll();
    seenStepsRef.current.clear();
    checkInstallAndStart();
  }, []);

  const retryInstall = useCallback(() => {
    clearBackendError();
    setLogs([]);
    setBackendStatus("not-installed");
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
      setBackendError(String(e), "install-error");
    }
  }, [elevationPackages]);

  // Initial check on mount (guarded against Strict Mode double-mount)
  useEffect(() => {
    if (mountedRef.current) return;
    mountedRef.current = true;

    if (!isTauri) {
      setRunningStatus();
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
        setBackendStatus("needs-elevation");
      }).then((u) => cleanup.push(u));

      listen<string>("install-progress-detail", (e) => {
        setProgressDetail(e.payload);
      }).then((u) => cleanup.push(u));

      listen<string>("install-failed", (e) => {
        setBackendError(e.payload, "install-error");
      }).then((u) => cleanup.push(u));

      listen<number>("server-port", (e) => {
        portRef.current = e.payload;
        setApiBase(e.payload);
      }).then((u) => cleanup.push(u));

      listen<void>("server-crashed", () => {
        startingRef.current = false;
        setBackendError("Server stopped unexpectedly");
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

    const onAuthFailed = (event: Event) => {
      const detail =
        event instanceof CustomEvent && typeof event.detail === "string"
          ? event.detail
          : "Desktop authentication failed. Restart Unsloth Studio or run `unsloth studio reset-password`.";
      setAuthFailure(detail);
    };
    window.addEventListener("tauri-auth-failed", onAuthFailed);
    const authFailure = getTauriAuthFailure();
    if (authFailure) setAuthFailure(authFailure);
    cleanup.push(() =>
      window.removeEventListener("tauri-auth-failed", onAuthFailed),
    );

    return () => {
      cleanup.forEach((fn) => fn());
      stopExternalServerPoll();
    };
  }, []);

  return {
    status, logs, error, isExternalServer,
    currentStepIndex, progressDetail, elevationPackages,
    startServer, stopServer, startInstall,
    retry, retryInstall, approveElevation,
  };
}
