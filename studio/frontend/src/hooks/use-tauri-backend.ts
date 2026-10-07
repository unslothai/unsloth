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
import {
  MANAGED_ENVIRONMENT_BUSY,
  MANAGED_ENVIRONMENT_UPDATING,
  preflightStaleMessage,
  runtimeRepairFailureMessage,
  runtimeRepairRecurrenceMessage,
  isLlamaRuntimeReason,
} from "@/hooks/backend-preflight-message";
import {
  recordRuntimeRepair,
  wasRuntimeRepairedRecently,
} from "@/hooks/runtime-repair-history";
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
  UPDATE_STARTUP_MESSAGE,
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
const MANAGED_ENVIRONMENT_POLL_MS = 5_000;
// Five minutes; bounds only a gate held outside this app, e.g. a terminal update at a prompt.
const MANAGED_ENVIRONMENT_WAIT_POLLS = 60;

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

  // Unattributable backends are stepped over at launch; only a mutation refuses, with
  // external_conflict_message from commands.rs.

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
  const startingRef = useRef(false);
  const stoppingRef = useRef(false);
  const mountedRef = useRef(false);
  const portRef = useRef<number | null>(null);
  // commands.rs's watchdog kills a timed-out backend ~30 s later with a payload-free server-crashed;
  // this keeps the timeout's log tail on screen.
  const startTimedOutRef = useRef(false);
  const [currentStepIndex, setCurrentStepIndex] = useState(-1);
  const [elevationPackages, setElevationPackages] = useState<string[]>([]);
  const [progressDetail, setProgressDetail] = useState<string | null>(null);
  const [startupMessage, setStartupMessage] = useState<StartupMessage>(
    INITIAL_STARTUP_MESSAGE,
  );
  // Dedupe step names (Strict Mode, event replay).
  const seenStepsRef = useRef(new Set<string>());
  // Attached to a server we did not spawn, so we cannot stop it.
  const [isExternalServer, setIsExternalServer] = useState(false);
  const externalPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const externalPollAbortedRef = useRef(false);
  const environmentWaitRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const environmentWaitPollsRef = useRef(0);
  const authFailureRef = useRef<string | null>(getTauriAuthFailure());
  const elevationResumeRef = useRef<"install" | "repair" | null>(null);
  // approveElevation restarts the repair with this after system packages land.
  const forcedRepairRef = useRef(false);
  const repairReasonRef = useRef<string | null>(null);
  // Retry may repair a recently repaired runtime; only the automatic launch-time repair is held.
  const allowHeldRuntimeRepairRef = useRef(false);
  // One preflight and one repair at a time; rapid Retry clicks raced the installer.
  const preflightInFlightRef = useRef(false);
  const repairInFlightRef = useRef(false);
  const [tauriEventsReady, setTauriEventsReady] = useState(!isTauri);
  // Read through: the listener lives in an effect that cannot reach this render's setState.
  const closing = useSyncExternalStore(subscribeAppClosing, isAppClosing);

  function setBackendStatus(nextStatus: BackendStatus) {
    if (authFailureRef.current) return;
    // A native install or repair outlives a reload; do not let the next poll overwrite its state.
    stopManagedEnvironmentWait();
    statusRef.current = nextStatus;
    setStatus(nextStatus);
    syncTrayStatus(nextStatus);
  }

  function setBackendError(
    nextError: string,
    nextStatus: BackendStatus = "error",
  ) {
    if (authFailureRef.current) return;
    stopManagedEnvironmentWait();
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

  useEffect(() => {
    statusRef.current = status;
  }, [status]);

  function stopManagedEnvironmentWait() {
    if (environmentWaitRef.current) {
      clearTimeout(environmentWaitRef.current);
      environmentWaitRef.current = null;
    }
    environmentWaitPollsRef.current = 0;
  }

  function waitForManagedEnvironment(bounded: boolean) {
    // setBackendStatus is a no-op here, so the poll would run on behind the error.
    if (authFailureRef.current) return;
    if (bounded && environmentWaitPollsRef.current >= MANAGED_ENVIRONMENT_WAIT_POLLS) {
      setBackendError(
        "Another Unsloth install or update, such as `unsloth studio update` in a terminal, is still running. Retry once it finishes.",
      );
      return;
    }
    // Read before setBackendStatus, which resets the count.
    const polls = bounded ? environmentWaitPollsRef.current + 1 : 0;
    setStartupMessage(UPDATE_STARTUP_MESSAGE);
    setBackendStatus("starting");
    environmentWaitPollsRef.current = polls;
    environmentWaitRef.current = setTimeout(() => {
      environmentWaitRef.current = null;
      void checkInstallAndStart();
    }, MANAGED_ENVIRONMENT_POLL_MS);
  }

  async function checkInstallAndStart() {
    // Before preflight: the native command can adopt a reaping backend and arm a crash watchdog.
    if (hasServerStopIntent()) {
      setBackendStatus("stopped");
      return;
    }
    if (preflightInFlightRef.current) return;
    preflightInFlightRef.current = true;
    const allowHeldRuntimeRepair = allowHeldRuntimeRepairRef.current;
    allowHeldRuntimeRepairRef.current = false;
    // A later call may hold the flag by the finally; clearing it unowned lets a third preflight in.
    let ownsPreflight = true;
    const releasePreflight = () => {
      if (!ownsPreflight) return;
      ownsPreflight = false;
      preflightInFlightRef.current = false;
    };
    try {
      const { invoke } = await import("@tauri-apps/api/core");

      const preflight = await invoke<DesktopPreflightResult>("desktop_preflight");
      // Held longer, it swallows the Retry server-start-timeout offers.
      releasePreflight();
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
          if (
            preflight.reason === MANAGED_ENVIRONMENT_BUSY ||
            preflight.reason === MANAGED_ENVIRONMENT_UPDATING
          ) {
            allowHeldRuntimeRepairRef.current = allowHeldRuntimeRepair;
            waitForManagedEnvironment(preflight.reason === MANAGED_ENVIRONMENT_BUSY);
            return;
          }
          if (preflight.can_auto_repair) {
            if (!allowHeldRuntimeRepair && wasRuntimeRepairedRecently(preflight.reason)) {
              setBackendError(runtimeRepairRecurrenceMessage());
            } else {
              await startRepair({ preflightReason: preflight.reason });
            }
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
    } finally {
      releasePreflight();
    }
  }

  async function startManagedServer() {
    // Before the re-entry guard: a requested start retires the earlier stop.
    clearServerStopIntent();
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

      // Rust emits server-port only after validating the process; no second health poll needed.
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

  // `forceInstaller` skips `studio update`, which would keep a CPU-only PyTorch; only manual repair
  // sets it.
  async function startRepair(options?: { forceInstaller?: boolean; preflightReason?: string | null }) {
    if (repairInFlightRef.current) return;
    repairInFlightRef.current = true;
    let ownsRepair = true;
    const releaseRepair = () => {
      if (!ownsRepair) return;
      ownsRepair = false;
      repairInFlightRef.current = false;
    };
    try {
      await runRepair(options, releaseRepair);
    } finally {
      releaseRepair();
    }
  }

  async function runRepair(
    options?: { forceInstaller?: boolean; preflightReason?: string | null },
    releaseRepair: () => void = () => {},
  ) {
    const forceInstaller = options?.forceInstaller ?? false;
    // Survives the elevation round trip.
    forcedRepairRef.current = forceInstaller;
    repairReasonRef.current = options?.preflightReason ?? null;
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
      recordRuntimeRepair(repairReasonRef.current);
      // Held across the following start, it swallows server-start-timeout's Retry.
      releaseRepair();
    } catch (e) {
      const msg = String(e);
      if (msg.includes("NEEDS_ELEVATION")) return;
      setBackendError(
        isLlamaRuntimeReason(repairReasonRef.current) ? runtimeRepairFailureMessage(msg) : msg,
        "repair-error",
      );
      return;
    }
    setBackendStatus("starting");
    elevationResumeRef.current = null;
    await startManagedServer();
  }

  async function startServer() {
    setBackendStatus("starting");
    await startManagedServer();
  }

  // statusRef stays "running" until the invoke resolves, so a second tray Stop would double-shutdown.
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
      startingRef.current = false;
      setIsExternalServer(false);
      stopExternalServerPoll();
      markServerStopIntent();
      setBackendStatus("stopped");
      return;
    }
    const { invoke } = await import("@tauri-apps/api/core");
    // Record intent before the await (reaping can block ~15s); roll back if the stop fails.
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
      // Skip the general preflight, which can attach to an unrelated backend; install-complete does not
      // call startServer, to avoid a double start.
      setBackendStatus("starting");
      elevationResumeRef.current = null;
      await startServer();
    } catch (e) {
      const msg = String(e);
      // Rust also emits install-needs-elevation; do not race it with install-error.
      if (msg.includes("NEEDS_ELEVATION")) return;
      setBackendError(msg, "install-error");
    }
  }

  const retry = useCallback(() => {
    // Retry on a FORCED repair re-runs it: the transactional installer restores the old environment, so
    // a preflight would restart the same CPU-only backend.
    const resumeForcedRepair =
      statusRef.current === "repair-error" && forcedRepairRef.current;
    forcedRepairRef.current = false;
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
    stopManagedEnvironmentWait();
    seenStepsRef.current.clear();
    if (resumeForcedRepair) {
      void startRepair({ forceInstaller: true });
      return;
    }
    allowHeldRuntimeRepairRef.current = true;
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
      setCurrentStepIndex(-1);
      setProgressDetail(null);
      elevationResumeRef.current = null;
      if (resume === "repair") {
        await startRepair({
          forceInstaller: forcedRepairRef.current,
          preflightReason: repairReasonRef.current,
        });
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

  // After the Tauri listeners are registered.
  useEffect(() => {
    if (!tauriEventsReady || mountedRef.current) return;
    mountedRef.current = true;

    if (!isTauri) {
      setRunningStatus();
      return;
    }
    checkInstallAndStart();
  }, [tauriEventsReady]);

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

      // Informational only; start_install's success path starts the server, to avoid races.
      register<void>("install-complete", () => {
        setCurrentStepIndex(999);
      });

      register<string>("install-step", (e) => {
        const stepName = e.payload;
        if (seenStepsRef.current.has(stepName)) return;
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
        setBackendError(
          isLlamaRuntimeReason(repairReasonRef.current)
            ? runtimeRepairFailureMessage(e.payload)
            : e.payload,
          "repair-error",
        );
      });

      register<number>("server-port", (e) => {
        portRef.current = e.payload;
        // A validated port means startup finished, so later crashes get the generic message.
        startTimedOutRef.current = false;
        setApiBase(e.payload);
      });

      register<void>("server-crashed", () => {
        startingRef.current = false;
        // The timeout message with the backend's output is more actionable than this one.
        if (startTimedOutRef.current) return;
        setBackendError("Server stopped unexpectedly");
      });

      // A hung backend never closes stdout, so server-crashed never fires; payload carries its tail.
      register<string>("server-start-timeout", (e) => {
        startingRef.current = false;
        startTimedOutRef.current = true;
        setBackendError(e.payload || "The Unsloth backend did not start in time");
      });

      register<string>("server-log", (e) => {
        setLogs((prev) => [...prev.slice(-499), e.payload]);
        setStartupMessage((current) => startupMessageFromLog(current, e.payload));
      });

      // Reaping blocks Rust's quit thread up to ~15s; cover it or it looks frozen.
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
      stopManagedEnvironmentWait();
    };
  }, []);

  return {
    status, logs, error, isExternalServer, closing,
    currentStepIndex, progressDetail, startupMessage, elevationPackages,
    startServer, stopServer, startInstall,
    retry, retryInstall, approveElevation, copyDiagnostics,
    // Same function as startup, so manual repair shows the same screen and restarts the backend.
    startRepair,
  };
}
