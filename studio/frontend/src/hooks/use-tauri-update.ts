// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useRef, useState } from "react";
import { isTrainingStartPending, useTrainingRuntimeStore } from "@/features/training";
import { apiUrl, isTauri } from "@/lib/api-base";
import {
  copySupportDiagnostics,
  type CopySupportDiagnosticsResult,
} from "@/lib/tauri-diagnostics";
import {
  adoptStagedUpdate,
  cancelStagedUpdate,
  checkDesktopUpdate,
  desktopUpdateBundleStatus,
  discardStagedUpdate,
  downloadDesktopUpdate,
  installDesktopUpdate,
  stagedUpdateStatus,
  startStagedUpdate,
  waitForDesktopUpdateDownload,
  type DesktopUpdateMetadata,
} from "@/lib/tauri-updater";
import { toast } from "@/lib/toast";
import {
  INITIAL_PREPARATION,
  backendIdle,
  desktopDownloadDecision,
  preparationStatus,
  restartPlan,
  sameUpdateVersion,
  settleWithin,
  stagingDecision,
  waitForBackendIdle,
  type UpdatePreparation,
} from "@/lib/update-preparation";

export type UpdateStatus =
  | "idle"
  | "checking"
  | "available"
  | "preparing"
  | "ready"
  | "updating-backend"
  | "downloading"
  | "installing"
  | "error";

export interface UpdateInfo {
  version: string;
  currentVersion: string;
  // Backend release this build pins, which preflight checks against.
  pypiVersion?: string;
  // latest.json's `notes`: a static download blurb, the same every release.
  // Kept as updater metadata, not shown; the popup fetches the real notes.
  body?: string;
  date?: string;
}

export type UpdatePhase =
  | "backend"
  | "shell_download"
  | "shell_install"
  | "recovered_after_shell_failure";

export type DesktopUpdatePolicyMode = "in_app" | "manual_linux_package";

interface DesktopUpdatePolicy {
  mode: DesktopUpdatePolicyMode;
  releasePageBaseUrl: string;
  releaseTagPrefix: string;
}

interface ManualUpdateInfo {
  version: string;
  currentVersion: string;
  pypiVersion?: string | null;
  body?: string;
  date?: string;
}

/** `pypi_version` from latest.json, which the updater passes through raw. */
function rawPypiVersion(raw: Record<string, unknown>): string | undefined {
  const value = raw.pypi_version;
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

export interface RetainedUpdateFailure {
  error: string;
  phase: UpdatePhase;
  progress: number;
  logs: string[];
}

const DEFAULT_UPDATE_POLICY: DesktopUpdatePolicy = {
  mode: "in_app",
  releasePageBaseUrl: "https://github.com/unslothai/unsloth/releases/tag/",
  releaseTagPrefix: "v",
};

const PREPARATION_STATUSES: ReadonlySet<UpdateStatus> = new Set([
  "available",
  "preparing",
  "ready",
]);

// Desktop quit never fires beforeunload, and only the renderer sees the shell installer.
function publishShellUpdateActive(active: boolean): void {
  if (!isTauri) return;
  void import("@tauri-apps/api/core")
    .then(({ invoke }) =>
      invoke("set_renderer_activity", { kind: "shell_update", active }),
    )
    .catch(() => {});
}

const UPDATE_VERSION_RE = /^v?\d+\.\d+\.\d+(?:(?:[-+][0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)|(?:\.(?:post|dev|rc)\d*)|(?:(?:post|dev|rc|a|b)\d*))?$/;

function normalizeUpdateVersion(version: string): string | null {
  const trimmed = version.trim();
  if (!UPDATE_VERSION_RE.test(trimmed)) return null;
  return trimmed.startsWith("v") ? trimmed.slice(1) : trimmed;
}

function manualReleasePageUrl(
  policy: DesktopUpdatePolicy,
  version: string,
): string | null {
  const normalized = normalizeUpdateVersion(version);
  if (!normalized) return null;
  return `${policy.releasePageBaseUrl}${policy.releaseTagPrefix}${normalized}`;
}

async function fetchHealth(): Promise<{ inference_active?: boolean } | null> {
  return settleWithin(async (signal) => {
    const res = await fetch(apiUrl("/api/health"), { signal });
    if (!res.ok) return null;
    return (await res.json()) as { inference_active?: boolean };
  }, null);
}

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function useTauriUpdate(isExternalServer = false) {
  const [status, setStatus] = useState<UpdateStatus>("idle");
  const statusRef = useRef<UpdateStatus>("idle");
  const [info, setInfo] = useState<UpdateInfo | null>(null);
  const infoRef = useRef<UpdateInfo | null>(null);
  const [hasChecked, setHasChecked] = useState(false);
  const [checkError, setCheckError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const progressRef = useRef(0);
  const [logs, setLogs] = useState<string[]>([]);
  const logsRef = useRef<string[]>([]);
  const [phase, setPhase] = useState<UpdatePhase | null>(null);
  const phaseRef = useRef<UpdatePhase | null>(null);
  const [dismissed, setDismissed] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastFailure, setLastFailure] = useState<RetainedUpdateFailure | null>(null);
  const [updatePolicy, setUpdatePolicy] = useState<DesktopUpdatePolicy>(DEFAULT_UPDATE_POLICY);
  const [preparation, setPreparation] = useState<UpdatePreparation>(INITIAL_PREPARATION);
  const preparationRef = useRef<UpdatePreparation>(INITIAL_PREPARATION);
  const preparingVersionRef = useRef<string | null>(null);
  const updateRef = useRef<DesktopUpdateMetadata | null>(null);
  const checkedRef = useRef(false);
  const startupScheduledRef = useRef(false);
  const checkingRef = useRef(false);
  const updatingRef = useRef(false);
  // Windows kill-on-close: false once a re-arm has failed, and every path that
  // starts a backend has to check it or the orphan risk comes straight back.
  const cleanupRearmedRef = useRef(true);
  // A webview reload rebuilds this hook with the ref back at its initial value
  // while the native job may still be disarmed, so the first gate of a mount
  // asks the native side instead of trusting it.
  const cleanupCheckedRef = useRef(false);

  async function resumeCleanup(): Promise<boolean> {
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      await invoke("resume_desktop_update_cleanup");
      cleanupRearmedRef.current = true;
    } catch (e) {
      console.error("Could not re-arm crash cleanup after a failed update:", e);
      cleanupRearmedRef.current = false;
    }
    return cleanupRearmedRef.current;
  }

  function updateStatus(next: UpdateStatus) {
    statusRef.current = next;
    setStatus(next);
  }

  function replaceInfo(nextInfo: UpdateInfo | null) {
    infoRef.current = nextInfo;
    setInfo(nextInfo);
  }

  function offerUpdate(nextInfo: UpdateInfo) {
    const isNewOffer = infoRef.current?.version !== nextInfo.version;
    replaceInfo(nextInfo);
    if (isNewOffer) {
      // Only a version the user has not seen may reopen the banner.
      setLastFailure(null);
      setError(null);
      setDismissed(false);
    }
    updateStatus("available");
  }

  function replaceLogs(nextLogs: string[]) {
    logsRef.current = nextLogs;
    setLogs(nextLogs);
  }

  function appendLog(line: string) {
    setLogs((prev) => {
      const next = [...prev.slice(-499), line];
      logsRef.current = next;
      return next;
    });
  }

  function setUpdateProgress(nextProgress: number) {
    progressRef.current = nextProgress;
    setProgress(nextProgress);
  }

  function setUpdatePhase(nextPhase: UpdatePhase | null) {
    phaseRef.current = nextPhase;
    setPhase(nextPhase);
  }

  function retainFailure(
    nextError: string,
    nextPhase: UpdatePhase = phaseRef.current ?? "backend",
  ) {
    const failure = {
      error: nextError,
      phase: nextPhase,
      progress: progressRef.current,
      logs: logsRef.current,
    };
    setLastFailure(failure);
    return failure;
  }

  function patchPreparation(patch: Partial<UpdatePreparation>) {
    const next = { ...preparationRef.current, ...patch };
    preparationRef.current = next;
    setPreparation(next);
    if (!PREPARATION_STATUSES.has(statusRef.current)) return;
    const derived = preparationStatus(next);
    if (derived === "ready" && statusRef.current !== "ready") {
      setDismissed(false);
    }
    updateStatus(derived);
  }

  function resetPreparation() {
    preparingVersionRef.current = null;
    preparationRef.current = INITIAL_PREPARATION;
    setPreparation(INITIAL_PREPARATION);
  }

  /** `resolved` is false when the policy is a fail-safe guess, not the real answer. */
  async function resolveUpdatePolicy(): Promise<{
    policy: DesktopUpdatePolicy;
    resolved: boolean;
  }> {
    if (!isTauri) return { policy: DEFAULT_UPDATE_POLICY, resolved: true };
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      const policy = await invoke<DesktopUpdatePolicy>("desktop_update_policy");
      setUpdatePolicy(policy);
      return { policy, resolved: true };
    } catch (e) {
      console.warn("Desktop update policy check failed:", e);
      const failSafePolicy: DesktopUpdatePolicy = {
        ...DEFAULT_UPDATE_POLICY,
        mode: "manual_linux_package",
      };
      setUpdatePolicy(failSafePolicy);
      return { policy: failSafePolicy, resolved: false };
    }
  }

  async function checkManualUpdate(policy: DesktopUpdatePolicy) {
    if (policy.mode !== "manual_linux_package") return false;
    const { invoke } = await import("@tauri-apps/api/core");
    const manualUpdate = await invoke<ManualUpdateInfo | null>(
      "check_desktop_manual_update",
    );
    if (!manualUpdate) return false;
    updateRef.current = null;
    offerUpdate({
      version: manualUpdate.version,
      currentVersion: manualUpdate.currentVersion,
      pypiVersion: manualUpdate.pypiVersion ?? undefined,
      body: manualUpdate.body,
      date: manualUpdate.date,
    });
    return true;
  }

  async function openManualUpdatePage(policy: DesktopUpdatePolicy, version: string) {
    const url = manualReleasePageUrl(policy, version);
    if (!url) {
      throw new Error(`Invalid desktop update version: ${version}`);
    }
    const { openUrl } = await import("@tauri-apps/plugin-opener");
    await openUrl(url);
  }

  async function checkForUpdate() {
    if (checkingRef.current || updatingRef.current) return;
    // A manual check covers startup, so the delayed timer must not repeat it.
    checkedRef.current = true;
    checkingRef.current = true;
    setCheckError(null);
    updateStatus("checking");

    try {
      const { policy, resolved } = await resolveUpdatePolicy();

      if (policy.mode === "manual_linux_package") {
        // Self-gates on the real target_os, so it is authoritative even if policy is a guess.
        if (await checkManualUpdate(policy)) return;
        if (resolved) {
          // latest.json has no deb/rpm key, so the in-app updater would offer an
          // AppImage this install cannot apply. Stop instead.
          updateRef.current = null;
          await clearPreparedBackendUpdate();
          replaceInfo(null);
          updateStatus("idle");
          return;
        }
        // Guessed policy, no manual offer: macOS, Windows and AppImage land here
        // and do have an in-app path. Fall through to it.
      }

      const update = await checkDesktopUpdate();
      if (update) {
        updateRef.current = update;
        offerUpdate({
          version: update.version,
          currentVersion: update.currentVersion,
          pypiVersion: rawPypiVersion(update.rawJson),
          body: update.body,
          date: update.date,
        });
      } else {
        updateRef.current = null;
        await clearPreparedBackendUpdate();
        replaceInfo(null);
        resetPreparation();
        updateStatus("idle");
      }
    } catch (e) {
      console.error("Update check failed:", e);
      setCheckError(String(e));
      updateStatus(infoRef.current ? preparationStatus(preparationRef.current) : "idle");
    } finally {
      checkingRef.current = false;
      setHasChecked(true);
    }
  }

  async function clearPreparedBackendUpdate() {
    const staged = await stagedUpdateStatus();
    if (staged.staging) await cancelStagedUpdate();
    await discardStagedUpdate();
  }
  // Startup owns one delayed check; this ref keeps a per-render function out of
  // the empty dep list. The closure reads only refs/setState, so it cannot stale.
  const initialCheckRef = useRef(checkForUpdate);

  useEffect(() => {
    if (!isTauri || startupScheduledRef.current) return;
    startupScheduledRef.current = true;

    const timer = setTimeout(() => {
      if (checkedRef.current) return;
      void initialCheckRef.current();
    }, 5000);
    return () => clearTimeout(timer);
  }, []);

  async function prepareUpdate(version: string, policy: DesktopUpdatePolicy) {
    if (!isTauri || policy.mode !== "in_app" || isExternalServer) return;
    if (preparingVersionRef.current === version) {
      if (preparationRef.current.shell !== "failed") return;
      patchPreparation({ shell: "pending", shellProgress: 0 });
      await prepareShell(version);
      return;
    }
    preparingVersionRef.current = version;
    preparationRef.current = INITIAL_PREPARATION;
    setPreparation(INITIAL_PREPARATION);
    patchPreparation({});

    await Promise.allSettled([prepareShell(version), prepareBackend(version)]);
  }

  async function ensureBundleDownloaded(): Promise<void> {
    setUpdatePhase("shell_download");
    updateStatus("downloading");
    setUpdateProgress(0);
    const version = updateRef.current?.version;
    if (!version) throw new Error("No desktop update has been checked.");
    for (;;) {
      const decision = desktopDownloadDecision(await desktopUpdateBundleStatus(), version);
      if (decision === "ready") break;
      if (decision === "wait") {
        await waitForDesktopUpdateDownload(version, setUpdateProgress, () => false);
      } else {
        await downloadDesktopUpdate(version, setUpdateProgress);
      }
    }
    patchPreparation({ shell: "done", shellProgress: 100 });
  }

  async function prepareShell(version: string) {
    try {
      for (;;) {
        const bundle = await desktopUpdateBundleStatus();
        const decision = desktopDownloadDecision(bundle, version);
        if (decision === "ready") {
          patchPreparation({ shell: "done", shellProgress: 100 });
          return;
        }
        patchPreparation({ shell: "downloading" });
        if (decision === "wait") {
          await waitForDesktopUpdateDownload(
            version,
            (percent) => {
              if (preparingVersionRef.current !== version) return;
              patchPreparation({ shellProgress: percent });
            },
            () => preparingVersionRef.current !== version,
          );
          if (preparingVersionRef.current !== version) return;
          continue;
        }
        await downloadDesktopUpdate(version, (percent) => {
          if (preparingVersionRef.current !== version) return;
          patchPreparation({ shellProgress: percent });
        });
        if (preparingVersionRef.current !== version) return;
      }
    } catch (e) {
      console.warn("Background app download failed:", e);
      if (preparingVersionRef.current !== version) return;
      patchPreparation({ shell: "failed" });
    }
  }

  async function prepareBackend(version: string) {
    try {
      const staged = await stagedUpdateStatus();
      const decision = stagingDecision({
        inApp: true,
        isExternalServer,
        offeredVersion: version,
        staged,
      });
      if (decision === "already-ready") {
        patchPreparation({ backend: "ready" });
        return;
      }
      if (decision === "skip") {
        patchPreparation({ backend: "skipped" });
        return;
      }
      if (decision === "adopt" || decision === "wait") {
        patchPreparation({ backend: "staging" });
        const settled = await adoptStagedUpdate(
          appendLog,
          () => preparingVersionRef.current !== version,
        );
        if (preparingVersionRef.current !== version) return;
        if (decision === "wait" || !sameUpdateVersion(settled.shellVersion, version)) {
          await prepareBackend(version);
          return;
        }
        patchPreparation({ backend: settled.state === "ready" ? "ready" : "failed" });
        return;
      }
      patchPreparation({ backend: "waiting" });
      const outcome = await waitForBackendIdle({
        cancelled: () => preparingVersionRef.current !== version,
        probe: async () =>
          backendIdle(
            await fetchHealth(),
            isTrainingStartPending(useTrainingRuntimeStore.getState()),
          ),
        sleep: wait,
        now: () => Date.now(),
      });
      if (outcome === "cancelled") return;
      if (outcome === "timeout") {
        // "skipped" still settles the offer into ready, and restartPlan falls back
        // to the classic update because the backend was never staged.
        patchPreparation({ backend: "skipped" });
        return;
      }
      patchPreparation({ backend: "staging" });
      await startStagedUpdate(appendLog);
      if (preparingVersionRef.current !== version) return;
      const settled = await stagedUpdateStatus();
      if (settled.state !== "ready" || !sameUpdateVersion(settled.shellVersion, version)) {
        await prepareBackend(version);
        return;
      }
      patchPreparation({ backend: "ready" });
    } catch (e) {
      console.warn("Background backend preparation failed:", e);
      if (preparingVersionRef.current !== version) return;
      patchPreparation({ backend: "failed" });
    }
  }

  async function installUpdate() {
    if (updatingRef.current) return;
    updatingRef.current = true;

    const cleanups: (() => void)[] = [];
    try {
      // A retry re-enters here, and start_backend_update spawns an
      // environment-mutating child of its own.
      if (!(await crashCleanupReady())) return;
      const { policy } = await resolveUpdatePolicy();
      if (policy.mode === "manual_linux_package") {
        const version = info?.version ?? updateRef.current?.version;
        if (!version) return;
        try {
          await openManualUpdatePage(policy, version);
          setDismissed(true);
          setError(null);
        } catch (manualError) {
          const msg = String(manualError);
          setError(msg);
          toast.error("Could not open release page", { description: msg });
        }
        return;
      }

      const update = updateRef.current;
      if (!update) return;
      if (statusRef.current === "available") {
        void prepareUpdate(update.version, policy);
        return;
      }

      const { invoke } = await import("@tauri-apps/api/core");
      const bundleReady = (await desktopUpdateBundleStatus()).downloaded;
      const plan = bundleReady ? restartPlan(preparationRef.current) : "classic";
      const staging = preparationRef.current.backend === "staging";
      preparingVersionRef.current = null;
      if (staging) {
        await cancelStagedUpdate().catch(() => {});
      }

      if (plan === "fast") {
        setUpdatePhase("shell_install");
        updateStatus("installing");
        setError(null);
        // Swallowing this leaves the old backend alive: its pid record then defers
        // the staged activation at the next launch, so the new shell would come up
        // against the old backend while reporting a finished update.
        try {
          await invoke("stop_server");
        } catch (stopError) {
          throw new Error(`Could not stop the backend before restarting: ${stopError}`);
        }
      } else {
        setUpdatePhase("backend");
        updateStatus("updating-backend");
        replaceLogs([]);
        setUpdateProgress(0);
        setError(null);
        setCheckError(null);
        setLastFailure(null);
        setDismissed(false);

        const { listen } = await import("@tauri-apps/api/event");
        const unlistenProgress = await listen<string>(
          "update-progress",
          (e) => {
            appendLog(e.payload);
          },
        );
        cleanups.push(unlistenProgress);

        const backendResult = await new Promise<"complete" | string>(
          (resolve) => {
            listen<void>("update-complete", () => resolve("complete")).then(
              (u) => cleanups.push(u),
            );
            listen<string>("update-failed", (e) =>
              resolve(e.payload),
            ).then((u) => cleanups.push(u));

            invoke("start_backend_update").catch((e) => resolve(String(e)));
          },
        );

        if (backendResult !== "complete") {
          retainFailure(backendResult, "backend");
          setError(backendResult);
          updateStatus("error");
          return;
        }

        await ensureBundleDownloaded();
        setUpdatePhase("shell_install");
        updateStatus("installing");
      }

      // `update::is_update_running` is already false here, and quitting mid-install
      // leaves a half-updated app.
      publishShellUpdateActive(true);
      try {
        await installDesktopUpdate();
      } catch (installError) {
        // Failed or cancelled: we keep running, so the cleanup the pre-exit hook
        // stood down has to come back.
        await resumeCleanup();
        throw installError;
      } finally {
        publishShellUpdateActive(false);
      }

      // Deliberately NOT re-arming kill-on-close before the restart: relaunch()
      // starts the replacement as a child, so it inherits this job, and re-arming
      // would make this process kill it on the way out.
      // The whole handoff is inside the recovery scope: anything that throws here
      // leaves this process running with cleanup still stood down.
      try {
        // relaunch() re-execs with the original argv, so flag the inherited --hidden as not a
        // login start. It only fails when there is a --hidden to suppress, so let it stop the
        // restart.
        await invoke("mark_in_app_relaunch");
        const { relaunch } = await import("@tauri-apps/plugin-process");
        await relaunch();
      } catch (relaunchError) {
        // No replacement process, so the marker would outlive it and unhide a later login start.
        await invoke("clear_in_app_relaunch").catch(() => {});
        // Still this process, so the cleanup has to come back after all.
        await resumeCleanup();
        throw relaunchError;
      }
    } catch (e) {
      console.error("Update failed:", e);
      const msg = String(e);

      // Shell update failed, so restart the backend on the updated code.
      if (phaseRef.current === "shell_download" || phaseRef.current === "shell_install") {
        // A backend started under a job that still has kill-on-close disabled is
        // the orphan this PR exists to prevent, so retry the re-arm and stop here
        // if it will not take.
        if (!(await crashCleanupReady())) {
          retainFailure(msg, phaseRef.current ?? "shell_install");
          return;
        }
        try {
          const { invoke } = await import("@tauri-apps/api/core");
          await invoke("start_server", { port: 8888 });
          retainFailure(msg, "recovered_after_shell_failure");
          toast.error("App update failed", {
            description:
              "Backend was updated. Copy diagnostics from the update banner if you need support.",
          });
          setError(null);
          updateStatus("idle");
          setDismissed(false);
          setUpdatePhase("recovered_after_shell_failure");
        } catch {
          retainFailure(msg, phaseRef.current ?? "shell_install");
          setError(msg);
          updateStatus("error");
        }
      } else {
        retainFailure(msg, phaseRef.current ?? "backend");
        setError(msg);
        updateStatus("error");
      }
    } finally {
      updatingRef.current = false;
      cleanup(cleanups);
    }
  }

  async function retryUpdate() {
    updatingRef.current = false;
    await installUpdate();
  }

  /** Every path that starts a child has to clear this first. */
  async function crashCleanupReady(): Promise<boolean> {
    if (!cleanupCheckedRef.current) {
      cleanupCheckedRef.current = true;
      try {
        const { invoke } = await import("@tauri-apps/api/core");
        cleanupRearmedRef.current = await invoke<boolean>("desktop_update_cleanup_armed");
      } catch {
        // On the desktop this is the one answer we cannot assume: fail closed
        // and let the gate below re-arm. In the browser there is no job at all.
        cleanupRearmedRef.current = !isTauri;
      }
    }
    if (cleanupRearmedRef.current) return true;
    if (await resumeCleanup()) return true;
    setError(
      "Crash cleanup could not be re-armed. Restart Unsloth before continuing.",
    );
    updateStatus("error");
    return false;
  }

  async function skipAndRestart() {
    const skippedError = error;
    // Same gate as the recovery path: this is offered on every error, so it is
    // the other way a user could start a backend under a disarmed job.
    if (!(await crashCleanupReady())) return;
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      await invoke("start_server", { port: 8888 });
      if (skippedError) {
        retainFailure(skippedError, phaseRef.current ?? "recovered_after_shell_failure");
        setDismissed(false);
      } else {
        setDismissed(true);
      }
      updateStatus("idle");
      setError(null);
      replaceLogs([]);
    } catch (e) {
      const msg = String(e);
      retainFailure(msg, phaseRef.current ?? "backend");
      setError(msg);
      updateStatus("error");
    }
  }

  function dismiss() {
    setDismissed(true);
  }

  function copyDiagnostics(): Promise<CopySupportDiagnosticsResult> {
    const failure = lastFailure;
    return copySupportDiagnostics({
      status: failure ? "error" : status,
      error: failure?.error ?? error,
      lastUiLogLines: failure?.logs ?? logs,
      flow: "update",
      updatePhase: failure?.phase ?? phase,
      updateProgress: failure?.progress ?? progress,
    });
  }

  // Install target for Linux packages that cannot self-update.
  const manualReleaseUrl =
    updatePolicy.mode === "manual_linux_package" && info
      ? manualReleasePageUrl(updatePolicy, info.version)
      : null;
  // Release page for the offered version, on every platform, for the notes link.
  const releasePageUrl = info ? manualReleasePageUrl(updatePolicy, info.version) : null;

  return {
    status,
    info,
    hasChecked,
    checkError,
    progress,
    logs,
    dismissed,
    error,
    phase,
    lastFailure,
    preparation,
    isExternalServer,
    updatePolicyMode: updatePolicy.mode,
    manualReleaseUrl,
    releasePageUrl,
    checkForUpdate,
    installUpdate,
    retryUpdate,
    skipAndRestart,
    dismiss,
    copyDiagnostics,
  };
}

export type TauriUpdateController = ReturnType<typeof useTauriUpdate>;

function cleanup(fns: (() => void)[]) {
  for (const fn of fns) {
    try {
      fn();
    } catch {
      // ignore
    }
  }
}
