// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState, useCallback, useRef } from "react";
import { isTauri, setApiBase } from "@/lib/api-base";
import {
  copySupportDiagnostics,
  type CopySupportDiagnosticsResult,
} from "@/lib/tauri-diagnostics";
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
  | "repairing"
  | "repair-error"
  | "starting"
  | "running"
  | "stopped"
  | "error";

type DesktopPreflightDisposition =
  | "not_installed"
  | "managed_ready"
  | "managed_stale"
  | "owned_ready"
  | "owned_stale"
  | "attached_ready"
  | "external_conflict";

interface DesktopPreflightResult {
  disposition: DesktopPreflightDisposition;
  reason: string | null;
  port: number | null;
  can_auto_repair: boolean;
  managed_bin: string | null;
}

const MANAGED_STARTUP_POLL_MS = 500;

type TauriInvoke = typeof import("@tauri-apps/api/core").invoke;
type ManagedStartupResult =
  | { status: "ready"; port: number }
  | { status: "aborted" };

function wait(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function externalConflictMessage(preflight: DesktopPreflightResult) {
  if (preflight.reason === "desktop_owned_backend_active") {
    return preflight.port
      ? `A desktop-owned Studio server for this install is already running on port ${preflight.port}. Quit the other desktop app instance, then try again.`
      : "A desktop-owned Studio server for this install is already running. Quit the other desktop app instance, then try again.";
  }

  if (preflight.reason === "desktop_owned_backend_starting") {
    return "The desktop-owned Studio backend is still starting. Wait a moment, then try again.";
  }

  if (preflight.reason?.startsWith("desktop_owned_backend_unmanageable:")) {
    return preflight.port
      ? `A desktop-owned Studio backend on port ${preflight.port} cannot be safely controlled by this desktop app. Stop that backend, then reopen Studio.`
      : "A desktop-owned Studio backend cannot be safely controlled by this desktop app. Stop that backend, then reopen Studio.";
  }

  return preflight.port
    ? `A Studio server for this install is already running from a terminal on port ${preflight.port}. Stop that server, or run \`unsloth studio update\` from that terminal before using the desktop app.`
    : "A Studio server for this install is already running from a terminal. Stop that server, or run `unsloth studio update` from that terminal before using the desktop app.";
}

async function waitForManagedServerReady(
  invoke: TauriInvoke,
  getPort: () => number | null,
  shouldContinue: () => boolean,
): Promise<ManagedStartupResult> {
  while (true) {
    if (!shouldContinue()) {
      return { status: "aborted" };
    }

    const port = getPort();
    if (port === null) {
      await wait(MANAGED_STARTUP_POLL_MS);
      continue;
    }

    const healthy = await invoke<boolean>("check_health", { port });
    if (!shouldContinue()) {
      return { status: "aborted" };
    }
    if (healthy && getPort() === port) {
      return { status: "ready", port };
    }

    await wait(MANAGED_STARTUP_POLL_MS);
  }
}

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
  const elevationResumeRef = useRef<"install" | "repair" | null>(null);
  const [tauriEventsReady, setTauriEventsReady] = useState(!isTauri);

  function setBackendStatus(nextStatus: BackendStatus) {
    if (authFailureRef.current) return;
    statusRef.current = nextStatus;
    setStatus(nextStatus);
  }

  function setBackendError(
    nextError: string,
    nextStatus: BackendStatus = "error",
  ) {
    if (authFailureRef.current) return;
    statusRef.current = nextStatus;
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
    statusRef.current = "error";
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

      const preflight = await invoke<DesktopPreflightResult>("desktop_preflight");
      switch (preflight.disposition) {
        case "attached_ready": {
          if (!preflight.port) {
            setBackendError("Desktop preflight found a backend without a port.");
            return;
          }
          setApiBase(preflight.port);
          portRef.current = preflight.port;
          setIsExternalServer(true);
          setRunningStatus();
          startExternalServerPoll(preflight.port);
          return;
        }
        case "owned_ready":
          if (!preflight.port) {
            setBackendError("Desktop preflight found an owned backend without a port.");
            return;
          }
          setApiBase(preflight.port);
          portRef.current = preflight.port;
          setIsExternalServer(false);
          stopExternalServerPoll();
          setRunningStatus();
          return;
        case "managed_ready":
          setIsExternalServer(false);
          stopExternalServerPoll();
          setBackendStatus("starting");
          await startManagedServer();
          return;
        case "owned_stale":
        case "managed_stale":
          setIsExternalServer(false);
          stopExternalServerPoll();
          if (preflight.can_auto_repair) {
            await startRepair();
          } else {
            setBackendError(
              preflight.disposition === "owned_stale"
                ? "Desktop-owned Studio backend is too old for this desktop app. Run `unsloth studio update`, then restart Studio."
                : "Managed Studio install is too old. Run `unsloth studio update`.",
            );
          }
          return;
        case "external_conflict":
          setIsExternalServer(false);
          stopExternalServerPoll();
          setBackendError(externalConflictMessage(preflight));
          return;
        case "not_installed":
          setBackendStatus("not-installed");
          return;
      }
    } catch (e) {
      setBackendError(String(e));
    }
  }

  async function startManagedServer() {
    // Prevent double-start race condition
    if (startingRef.current) {
      return;
    }
    startingRef.current = true;
    portRef.current = null;

    try {
      const { invoke } = await import("@tauri-apps/api/core");
      // backend/run.py keeps the existing 8888-8908 fallback via
      // server-port/TAURI_PORT.
      await invoke("start_managed_server", { port: 8888 });

      // Wait for the owned backend's server-port event. Do not attach to an
      // external backend if the managed start does not report a port.
      const startupResult = await waitForManagedServerReady(
        invoke,
        () => portRef.current,
        () => startingRef.current,
      );

      if (startupResult.status === "ready") {
        setApiBase(startupResult.port);
        setRunningStatus();
        startingRef.current = false;
        return;
      }

      if (startupResult.status === "aborted") {
        return;
      }

    } catch (e) {
      const msg = String(e);
      if (msg.includes("already running")) {
        startingRef.current = false;
        setBackendError(
          "Managed server is already running but did not report a port. Restart Studio and try again.",
        );
        return;
      }
      setBackendError(msg);
    }
    startingRef.current = false;
  }

  async function startRepair() {
    elevationResumeRef.current = null;
    setCurrentStepIndex(-1);
    setProgressDetail(null);
    seenStepsRef.current.clear();
    startingRef.current = false;
    portRef.current = null;
    setIsExternalServer(false);
    stopExternalServerPoll();
    setLogs([]);
    clearBackendError();
    setBackendStatus("repairing");

    const { invoke } = await import("@tauri-apps/api/core");
    try {
      await invoke("start_managed_repair");

      setBackendStatus("starting");
      elevationResumeRef.current = null;
      await startManagedServer();
    } catch (e) {
      const msg = String(e);
      if (msg.includes("NEEDS_ELEVATION")) return;
      setBackendError(msg, "repair-error");
    }
  }

  async function startServer() {
    setBackendStatus("starting");
    await startManagedServer();
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
    elevationResumeRef.current = null;
    setCurrentStepIndex(-1);
    setProgressDetail(null);
    seenStepsRef.current.clear();
    setBackendStatus("installing");
    setLogs([]);
    clearBackendError();
    const { invoke } = await import("@tauri-apps/api/core");
    try {
      await invoke("start_install");
      // Install completed — start the managed backend we just installed.
      // Do not run the general preflight here: it can attach to an unrelated
      // already-running CLI/backend server before launching our managed one.
      // The install-complete event listener does NOT call startServer() to
      // avoid a double-start race condition.
      setBackendStatus("starting");
      elevationResumeRef.current = null;
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
    elevationResumeRef.current = null;
    setIsExternalServer(false);
    stopExternalServerPoll();
    seenStepsRef.current.clear();
    checkInstallAndStart();
  }, []);

  const retryInstall = useCallback(async () => {
    const resume = elevationResumeRef.current;
    if (resume) {
      try {
        const { invoke } = await import("@tauri-apps/api/core");
        await invoke("cancel_pending_elevation");
      } catch (error) {
        console.warn("Failed to record elevation cancellation", error);
      }
    }
    elevationResumeRef.current = null;
    clearBackendError();
    setLogs([]);
    setElevationPackages([]);
    if (resume === "repair") {
      setBackendError("Repair canceled before system packages were installed.", "repair-error");
      return;
    }
    setBackendStatus("not-installed");
  }, []);

  const approveElevation = useCallback(async () => {
    const resume = elevationResumeRef.current ?? "install";
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      await invoke("install_system_packages", { packages: elevationPackages });
      // Packages installed successfully, resume the flow that requested them.
      setCurrentStepIndex(-1);
      setProgressDetail(null);
      elevationResumeRef.current = null;
      if (resume === "repair") {
        await startRepair();
      } else {
        await startInstall();
      }
    } catch (e) {
      setBackendError(String(e), resume === "repair" ? "repair-error" : "install-error");
    }
  }, [elevationPackages]);

  const copyDiagnostics = useCallback((): Promise<CopySupportDiagnosticsResult> => {
    const currentStatus = statusRef.current;
    const flow =
      currentStatus === "repairing" ||
      currentStatus === "repair-error" ||
      (currentStatus === "needs-elevation" && elevationResumeRef.current === "repair")
        ? "repair"
        : currentStatus === "installing" ||
            currentStatus === "install-error" ||
            currentStatus === "not-installed" ||
            currentStatus === "needs-elevation"
          ? "install"
          : "backend";

    return copySupportDiagnostics({
      status: currentStatus,
      error,
      currentStepIndex,
      progressDetail,
      elevationPackages,
      lastUiLogLines: logs,
      flow,
    });
  }, [currentStepIndex, elevationPackages, error, logs, progressDetail]);

  // Initial check on mount after Tauri event listeners are registered.
  useEffect(() => {
    if (!tauriEventsReady || mountedRef.current) return;
    mountedRef.current = true;

    if (!isTauri) {
      setRunningStatus();
      return;
    }
    checkInstallAndStart();
  }, [tauriEventsReady]);

  // Listen for Tauri events
  useEffect(() => {
    if (!isTauri) return;
    const cleanup: (() => void)[] = [];
    let disposed = false;

    import("@tauri-apps/api/event").then(({ listen }) => {
      const registrations: Promise<void>[] = [];
      function register<T>(
        event: string,
        handler: Parameters<typeof listen<T>>[1],
      ) {
        registrations.push(
          listen<T>(event, handler).then((unlisten) => {
            if (disposed) {
              unlisten();
            } else {
              cleanup.push(unlisten);
            }
          }),
        );
      }

      register<string>("install-progress", (e) => {
        setLogs((prev) => [...prev.slice(-499), e.payload]);
      });

      // install-complete is informational only — does NOT trigger startServer.
      // The invoke("start_install") success path handles that to avoid races.
      register<void>("install-complete", () => {
        setCurrentStepIndex(999); // all steps done
      });

      register<string>("install-step", (e) => {
        const stepName = e.payload;
        if (seenStepsRef.current.has(stepName)) return; // deduplicate
        seenStepsRef.current.add(stepName);
        setCurrentStepIndex((prev) => prev + 1);
        setProgressDetail(null);
      });

      register<string[]>("install-needs-elevation", (e) => {
        elevationResumeRef.current = "install";
        setElevationPackages(e.payload);
        setBackendStatus("needs-elevation");
      });

      register<string>("install-progress-detail", (e) => {
        setProgressDetail(e.payload);
      });

      register<string>("install-failed", (e) => {
        setBackendError(e.payload, "install-error");
      });

      register<string>("repair-progress", (e) => {
        setLogs((prev) => [...prev.slice(-499), e.payload]);
      });

      register<string[]>("repair-needs-elevation", (e) => {
        elevationResumeRef.current = "repair";
        setElevationPackages(e.payload);
        setBackendStatus("needs-elevation");
      });

      register<void>("repair-complete", () => {
        if (statusRef.current !== "repairing") return;
        setProgressDetail("Repair complete");
      });

      register<string>("repair-failed", (e) => {
        if (statusRef.current !== "repairing") return;
        setBackendError(e.payload, "repair-error");
      });

      register<number>("server-port", (e) => {
        portRef.current = e.payload;
        setApiBase(e.payload);
      });

      register<void>("server-crashed", () => {
        startingRef.current = false;
        setBackendError("Server stopped unexpectedly");
      });

      register<string>("server-log", (e) => {
        setLogs((prev) => [...prev.slice(-499), e.payload]);
      });

      register<void>("tray-toggle-server", () => {
        if (statusRef.current === "running") {
          stopServer();
        } else if (
          statusRef.current === "stopped" ||
          statusRef.current === "error"
        ) {
          retry();
        }
      });

      Promise.all(registrations)
        .then(() => {
          if (!disposed) setTauriEventsReady(true);
        })
        .catch((error) => {
          if (!disposed) setBackendError(String(error));
        });
    }).catch((error) => {
      if (!disposed) setBackendError(String(error));
    });

    const onAuthFailed = (event: Event) => {
      const detail =
        event instanceof CustomEvent && typeof event.detail === "string"
          ? event.detail
          : "Desktop authentication failed. Update or repair the managed Studio install, then restart Studio.";
      setAuthFailure(detail);
    };
    window.addEventListener("tauri-auth-failed", onAuthFailed);
    const authFailure = getTauriAuthFailure();
    if (authFailure) setAuthFailure(authFailure);
    cleanup.push(() =>
      window.removeEventListener("tauri-auth-failed", onAuthFailed),
    );

    return () => {
      disposed = true;
      cleanup.forEach((fn) => fn());
      stopExternalServerPoll();
    };
  }, []);

  return {
    status, logs, error, isExternalServer,
    currentStepIndex, progressDetail, elevationPackages,
    startServer, stopServer, startInstall,
    retry, retryInstall, approveElevation, copyDiagnostics,
  };
}
