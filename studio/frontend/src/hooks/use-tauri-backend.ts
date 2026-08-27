// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  useEffect,
  useState,
  useCallback,
  useRef,
  useSyncExternalStore,
} from "react";
import { isTauri, setApiBase } from "@/lib/api-base";
import { preflightStaleMessage } from "@/hooks/backend-preflight-message";
import {
  copySupportDiagnostics,
  type CopySupportDiagnosticsResult,
} from "@/lib/tauri-diagnostics";
import {
  clearTauriAuthFailure,
  getTauriAuthFailure,
} from "@/features/auth";
import {
  APP_CLOSING_CANCELLED_EVENT,
  APP_CLOSING_EVENT,
  clearAppClosing,
  isAppClosing,
  markAppClosing,
  subscribeAppClosing,
} from "@/components/tauri/closing-signal";
import {
  INITIAL_STARTUP_MESSAGE,
  SERVER_STARTUP_MESSAGE,
  startupMessageFromLog,
  type StartupMessage,
} from "@/components/tauri/startup-messages";
import {
  clearServerStopIntent,
  hasServerStopIntent,
  markServerStopIntent,
} from "./server-stop-intent";

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

function syncTrayStatus(status: BackendStatus) {
  if (!isTauri) return;
  import("@tauri-apps/api/core")
    .then(({ invoke }) => invoke("set_tray_server_status", { status }))
    .catch(() => {});
}

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
      ? `A desktop-owned Unsloth server for this install is already running on port ${preflight.port}. Quit the other desktop app instance, then try again.`
      : "A desktop-owned Unsloth server for this install is already running. Quit the other desktop app instance, then try again.";
  }

  if (preflight.reason === "desktop_owned_backend_starting") {
    return "The desktop-owned Unsloth backend is still starting. Wait a moment, then try again.";
  }

  // A backend we cannot attribute to this install no longer reaches here: the
  // launch steps over its port. Only a mutation still refuses, and that message
  // comes from external_conflict_message in commands.rs.

  if (preflight.reason?.startsWith("desktop_owned_backend_unmanageable:")) {
    return preflight.port
      ? `A desktop-owned Unsloth backend on port ${preflight.port} cannot be safely controlled by this desktop app. Stop that backend, then reopen Unsloth.`
      : "A desktop-owned Unsloth backend cannot be safely controlled by this desktop app. Stop that backend, then reopen Unsloth.";
  }

  return preflight.port
    ? `An Unsloth server for this install is already running from a terminal on port ${preflight.port}. Stop that server, or run \`unsloth studio update\` from that terminal before using the desktop app.`
    : "An Unsloth server for this install is already running from a terminal. Stop that server, or run `unsloth studio update` from that terminal before using the desktop app.";
}

async function waitForManagedServerPort(
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

    return { status: "ready", port };
  }
}

export function useTauriBackend() {
  const [status, setStatus] = useState<BackendStatus>("checking");
  const statusRef = useRef<BackendStatus>(status);
  const [logs, setLogs] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  // Guard against double startServer calls
  const startingRef = useRef(false);
  // Guard against double stopServer calls
  const stoppingRef = useRef(false);
  // Guard against React Strict Mode double-mount
  const mountedRef = useRef(false);
  // Track the discovered port from server-port event
  const portRef = useRef<number | null>(null);
  // Set once server-start-timeout has reported a stalled startup. commands.rs's
  // health watchdog kills that same portless backend ~30 s later and emits a
  // payload-free server-crashed, so without this the log tail the timeout carried
  // is replaced by "Server stopped unexpectedly" on an unattended error screen.
  // Cleared whenever a start attempt begins or a validated port arrives.
  const startTimedOutRef = useRef(false);
  const [currentStepIndex, setCurrentStepIndex] = useState(-1);
  const [elevationPackages, setElevationPackages] = useState<string[]>([]);
  const [progressDetail, setProgressDetail] = useState<string | null>(null);
  const [startupMessage, setStartupMessage] = useState<StartupMessage>(
    INITIAL_STARTUP_MESSAGE,
  );
  // Track seen step names to deduplicate (Strict Mode, event replay, etc.)
  const seenStepsRef = useRef(new Set<string>());
  // True when we attached to a server we didn't spawn (can't stop it)
  const [isExternalServer, setIsExternalServer] = useState(false);
  const externalPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const externalPollAbortedRef = useRef(false);
  const authFailureRef = useRef<string | null>(getTauriAuthFailure());
  const elevationResumeRef = useRef<"install" | "repair" | null>(null);
  // Whether the repair in flight was asked to skip straight to the installer. Read back by
  // approveElevation, which restarts the repair after the system packages land.
  const forcedRepairRef = useRef(false);
  const [tauriEventsReady, setTauriEventsReady] = useState(!isTauri);
  // Read through rather than mirrored into state: the app-closing listener is registered
  // inside the long event effect below, which cannot reach a setState from this render.
  const closing = useSyncExternalStore(subscribeAppClosing, isAppClosing);

  function setBackendStatus(nextStatus: BackendStatus) {
    if (authFailureRef.current) return;
    statusRef.current = nextStatus;
    setStatus(nextStatus);
    syncTrayStatus(nextStatus);
  }

  function setBackendError(
    nextError: string,
    nextStatus: BackendStatus = "error",
  ) {
    if (authFailureRef.current) return;
    statusRef.current = nextStatus;
    setStatus(nextStatus);
    setError(nextError);
    syncTrayStatus(nextStatus);
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
    syncTrayStatus("error");
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
    // Honor a persisted stop before preflight: the native command side-effects
    // (it can adopt a still-reaping backend, reset the intentional-stop flag,
    // and arm a watchdog that later fires server-crashed over this screen).
    if (hasServerStopIntent()) {
      setBackendStatus("stopped");
      return;
    }
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
          setStartupMessage(SERVER_STARTUP_MESSAGE);
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
          setStartupMessage(SERVER_STARTUP_MESSAGE);
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
              preflightStaleMessage(preflight.disposition, preflight.reason),
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
    // Ahead of the re-entry guard: a start the user asked for retires the stop they
    // asked for earlier, whether or not this particular call goes on to do the work.
    clearServerStopIntent();
    // Prevent double-start race condition
    if (startingRef.current) {
      return;
    }
    startingRef.current = true;
    setStartupMessage(INITIAL_STARTUP_MESSAGE);
    portRef.current = null;
    startTimedOutRef.current = false;

    try {
      const { invoke } = await import("@tauri-apps/api/core");
      // backend/run.py keeps the 8888-8908 fallback via server-port/TAURI_PORT.
      await invoke("start_managed_server", { port: 8888 });

      // Rust emits server-port only after validating the desktop-owned process.
      // Treat that as the UI handoff point instead of doing a second health poll.
      const startupResult = await waitForManagedServerPort(
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
          "Managed server is already running but did not report a port. Restart Unsloth and try again.",
        );
        return;
      }
      setBackendError(msg);
    }
    startingRef.current = false;
  }

  // `forceInstaller` runs the bundled installer without trying `studio update` first. The
  // automatic callers leave it off: an out-of-date venv is the common case and the update is
  // the cheap fix. Settings' manual "Repair installation" turns it on, because an update
  // reuses the environment it finds -- a managed venv whose PyTorch was replaced by a CPU-only
  // wheel comes back from a successful update still CPU-only, and only the installer
  // re-selects the torch index.
  async function startRepair(options?: { forceInstaller?: boolean }) {
    const forceInstaller = options?.forceInstaller ?? false;
    // Survives the elevation round trip: approveElevation resumes by calling this again, and
    // resuming without the flag would run the update the caller deliberately skipped.
    forcedRepairRef.current = forceInstaller;
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
      await invoke("start_managed_repair", { forceInstaller });

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

  // One stop at a time. The tray toggle branches on statusRef, which stays "running" until
  // the invoke resolves, so a second tray Stop otherwise runs a second shutdown against the
  // backend the first is still taking down. Mirrors the startingRef guard on the start path.
  async function stopServer() {
    if (stoppingRef.current) return;
    stoppingRef.current = true;
    try {
      await runStopServer();
    } finally {
      stoppingRef.current = false;
    }
  }

  async function runStopServer() {
    if (isExternalServer) {
      // We attached to a server we didn't spawn: can't kill it, just disconnect the UI.
      startingRef.current = false;
      setIsExternalServer(false);
      stopExternalServerPoll();
      markServerStopIntent();
      setBackendStatus("stopped");
      return;
    }
    const { invoke } = await import("@tauri-apps/api/core");
    // Record intent before the await: reaping can block ~15s and a reload
    // mid-await would lose the marker. Roll back if the stop fails.
    markServerStopIntent();
    try {
      await invoke("stop_server");
    } catch (e) {
      clearServerStopIntent();
      throw e;
    }
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
      // Install done: start the managed backend we just installed. Don't run the
      // general preflight here, it can attach to an unrelated running CLI/backend
      // before launching ours. The install-complete listener does NOT call
      // startServer() to avoid a double-start race.
      setBackendStatus("starting");
      elevationResumeRef.current = null;
      await startServer();
    } catch (e) {
      const msg = String(e);
      // NEEDS_ELEVATION is not a real error: the Rust side also emits
      // install-needs-elevation (sets needs-elevation status). Don't race with it
      // by setting install-error here.
      if (msg.includes("NEEDS_ELEVATION")) return;
      setBackendError(msg, "install-error");
    }
  }

  const retry = useCallback(() => {
    clearAuthFailure();
    clearServerStopIntent();
    setError(null);
    setLogs([]);
    startingRef.current = false;
    portRef.current = null;
    startTimedOutRef.current = false;
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
        await startRepair({ forceInstaller: forcedRepairRef.current });
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

      // install-complete is informational only; does NOT trigger startServer. The
      // invoke("start_install") success path handles that to avoid races.
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
        // A validated port means startup finished after all, so a later crash is
        // a real crash and deserves the generic message.
        startTimedOutRef.current = false;
        setApiBase(e.payload);
      });

      register<void>("server-crashed", () => {
        startingRef.current = false;
        // Startup already timed out and left a message naming the backend's last
        // output. That is strictly more actionable than this one, and the kill it
        // reports is the timeout's own consequence, so keep the detail.
        if (startTimedOutRef.current) return;
        setBackendError("Server stopped unexpectedly");
      });

      // A backend that hangs never closes stdout, so server-crashed never fires and the
      // startup screen would otherwise spin forever. Payload carries the backend's tail.
      register<string>("server-start-timeout", (e) => {
        startingRef.current = false;
        startTimedOutRef.current = true;
        setBackendError(e.payload || "The Unsloth backend did not start in time");
      });

      register<string>("server-log", (e) => {
        setLogs((prev) => [...prev.slice(-499), e.payload]);
        setStartupMessage((current) => startupMessageFromLog(current, e.payload));
      });

      // Reaping the backend blocks Rust's quit thread for up to ~15s. Cover the window
      // for that, or it reads as a freeze.
      register<void>(APP_CLOSING_EVENT, () => {
        markAppClosing();
      });

      register<void>(APP_CLOSING_CANCELLED_EVENT, () => {
        clearAppClosing();
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
          : "Desktop authentication failed. Update or repair the managed Unsloth install, then restart Unsloth.";
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
    status, logs, error, isExternalServer, closing,
    currentStepIndex, progressDetail, startupMessage, elevationPackages,
    startServer, stopServer, startInstall,
    retry, retryInstall, approveElevation, copyDiagnostics,
    // Exported for the manual "Repair installation" action in Settings. It is the same
    // function startup uses, so a manual repair renders the same repairing screen and
    // restarts the backend afterwards rather than leaving it stopped.
    startRepair,
  };
}
