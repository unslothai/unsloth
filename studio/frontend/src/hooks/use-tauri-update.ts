// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useRef, useState } from "react";
import { isTauri } from "@/lib/api-base";
import {
  copySupportDiagnostics,
  type CopySupportDiagnosticsResult,
} from "@/lib/tauri-diagnostics";
import {
  adoptPrefetch,
  cancelPrefetch,
  checkDesktopUpdate,
  desktopUpdateBundleStatus,
  discardPrefetch,
  downloadDesktopUpdate,
  installDesktopUpdate,
  listenDesktopUpdateDownload,
  prefetchStatus,
  sameUpdateVersion,
  startPrefetch,
  type DesktopUpdateMetadata,
} from "@/lib/tauri-updater";
import { toast } from "@/lib/toast";
import {
  INITIAL_PREPARATION,
  desktopDownloadDecision,
  prefetchDecision,
  preparationStatus,
  type UpdatePreparation,
} from "@/lib/update-preparation";

export type UpdateStatus =
  | "idle"
  | "checking"
  | "available"
  // "Update now" was pressed and the app bundle and the wheels are being
  // fetched in the background. Nothing is installed yet and nothing is stopped.
  | "preparing"
  // Everything that could be fetched ahead of time is fetched. Restarting now
  // runs the ordinary update, which finds its downloads already done.
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
  // latest.json's `notes`: a static download blurb. Kept as metadata, not shown.
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
const STARTUP_UPDATE_CHECK_DELAY_MS = 5000;
const PERIODIC_UPDATE_CHECK_INTERVAL_MS = 60 * 60 * 1000;
const BUNDLE_DOWNLOAD_POLL_MS = 500;
// A native download that stalls without clearing the flag would otherwise hold the update forever.
const BUNDLE_DOWNLOAD_WAIT_MS = 10 * 60 * 1000;

// Desktop quit never fires beforeunload, and only the renderer sees the shell installer.
function publishShellUpdateActive(active: boolean): void {
  if (!isTauri) return;
  void import("@tauri-apps/api/core")
    .then(({ invoke }) =>
      invoke("set_renderer_activity", { kind: "shell_update", active }),
    )
    .catch(() => {});
}

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
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
  // The offer the background work belongs to. Every step re-reads it and gives
  // up when it has moved, so a newer offer cannot be finished by an older run.
  const preparingVersionRef = useRef<string | null>(null);
  const updateRef = useRef<DesktopUpdateMetadata | null>(null);
  const checkedRef = useRef(false);
  const lastCheckAtRef = useRef<number | null>(null);
  const checkingRef = useRef(false);
  const updatingRef = useRef(false);
  // Windows kill-on-close: false once a re-arm has failed, and every path that starts a backend must check it.
  // A webview reload resets this ref while the native job may still be disarmed, so the first gate asks natively.
  const cleanupRearmedRef = useRef(true);
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
      setLastFailure(null);
      setError(null);
      setDismissed(false);
    }
    if (preparingVersionRef.current === nextInfo.version) {
      // The hourly recheck re-offers the version already being prepared. Putting
      // it back to "available" would replace a pill that says "Update ready" with
      // an Update button, and the second press would prepare it all over again.
      updateStatus(preparationStatus(preparationRef.current));
      return;
    }
    if (isNewOffer && preparingVersionRef.current !== null) {
      // Whatever was being prepared is for a version nobody is offering any more.
      void restartPreparationFor(nextInfo.version);
      return;
    }
    // An hourly re-offer of the version already on show must not reopen a dismissed banner.
    updateStatus("available");
  }

  /** What the current offer looks like: preparing only if something is preparing it. */
  function offeredStatus(): UpdateStatus {
    const offered = infoRef.current;
    if (!offered) return "idle";
    return preparingVersionRef.current === offered.version
      ? preparationStatus(preparationRef.current)
      : "available";
  }

  function patchPreparation(patch: Partial<UpdatePreparation>) {
    const next = { ...preparationRef.current, ...patch };
    preparationRef.current = next;
    setPreparation(next);
    // The status is a function of the preparation, so it is derived here rather
    // than set by each step, which is how the two used to drift apart.
    updateStatus(preparationStatus(next));
  }

  function resetPreparation() {
    preparationRef.current = INITIAL_PREPARATION;
    setPreparation(INITIAL_PREPARATION);
  }

  /** A newer offer arrived mid-preparation: drop the old work and start again. */
  async function restartPreparationFor(version: string) {
    preparingVersionRef.current = null;
    resetPreparation();
    updateStatus("available");
    try {
      await cancelPrefetch();
    } catch (e) {
      console.warn("Could not stop the background preparation:", e);
    }
    // The user already asked for an update; the version changing underneath is
    // not a reason to make them ask again.
    void prepareUpdate(version);
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
    lastCheckAtRef.current = Date.now();
    checkingRef.current = true;
    setCheckError(null);
    updateStatus("checking");

    try {
      const { policy, resolved } = await resolveUpdatePolicy();

      if (policy.mode === "manual_linux_package") {
        // Self-gates on the real target_os, so it is authoritative even if policy is a guess.
        if (await checkManualUpdate(policy)) return;
        if (resolved) {
          // latest.json has no deb/rpm key, so the in-app updater would offer an AppImage this install cannot apply.
          updateRef.current = null;
          await clearPreparedUpdate();
          replaceInfo(null);
          updateStatus("idle");
          return;
        }
        // Guessed policy, no manual offer: macOS, Windows and AppImage do have an in-app path.
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
        await clearPreparedUpdate();
        replaceInfo(null);
        updateStatus("idle");
      }
    } catch (e) {
      console.error("Update check failed:", e);
      setCheckError(String(e));
      updateStatus(offeredStatus());
    } finally {
      checkingRef.current = false;
      setHasChecked(true);
    }
  }

  function checkForUpdateWhenSafe() {
    // Recovery owns version-specific state until the user settles it.
    if (statusRef.current === "error") return;
    void checkForUpdate();
  }

  const scheduledCheckRef = useRef(checkForUpdateWhenSafe);

  useEffect(() => {
    if (!isTauri) return;

    const startupTimer = setTimeout(() => {
      if (checkedRef.current) return;
      scheduledCheckRef.current();
    }, STARTUP_UPDATE_CHECK_DELAY_MS);
    const periodicTimer = setInterval(() => {
      scheduledCheckRef.current();
    }, PERIODIC_UPDATE_CHECK_INTERVAL_MS);
    const checkWhenVisibleAndDue = () => {
      if (document.hidden) return;
      const lastCheckAt = lastCheckAtRef.current;
      if (lastCheckAt === null) return;
      const elapsed = Date.now() - lastCheckAt;
      if (
        elapsed >= 0 &&
        elapsed < PERIODIC_UPDATE_CHECK_INTERVAL_MS
      ) {
        return;
      }
      scheduledCheckRef.current();
    };
    window.addEventListener("focus", checkWhenVisibleAndDue);
    document.addEventListener("visibilitychange", checkWhenVisibleAndDue);

    return () => {
      clearTimeout(startupTimer);
      clearInterval(periodicTimer);
      window.removeEventListener("focus", checkWhenVisibleAndDue);
      document.removeEventListener("visibilitychange", checkWhenVisibleAndDue);
    };
  }, []);

  /** No update is on offer any more, so the disk copy is holding space for nothing. */
  async function clearPreparedUpdate(): Promise<void> {
    preparingVersionRef.current = null;
    resetPreparation();
    if (!isTauri) return;
    try {
      // discard_prefetch stops a running child first, so no separate cancel.
      await discardPrefetch();
    } catch (e) {
      console.warn("Could not discard the prepared update:", e);
    }
  }

  /**
   * Fetch everything the restart would otherwise fetch, with the app still
   * running. Both halves are independent and neither can fail the other, so they
   * settle rather than race: a failed app download puts the offer back to the
   * ordinary Update button, and a failed prefetch just means the restart
   * downloads its own wheels.
   */
  async function prepareUpdate(version: string): Promise<void> {
    if (!isTauri || isExternalServer) return;
    if (preparingVersionRef.current === version) {
      // Already preparing. A retry only makes sense for the half that failed.
      if (preparationRef.current.shell !== "failed") return;
      patchPreparation({ shell: "pending", shellProgress: 0 });
      await prepareShell(version);
      return;
    }
    preparingVersionRef.current = version;
    resetPreparation();
    patchPreparation({});

    await Promise.allSettled([prepareShell(version), prepareBackend(version)]);
  }

  async function prepareShell(version: string): Promise<void> {
    // A download already in flight reports on the same event as one this renderer
    // starts, so the wait below shows real progress instead of a still bar.
    // Attached only when there is something to listen to, released either way.
    let unlisten: (() => void) | null = null;
    const waitUntil = Date.now() + BUNDLE_DOWNLOAD_WAIT_MS;
    try {
      for (;;) {
        const decision = desktopDownloadDecision(
          await desktopUpdateBundleStatus(),
          version,
        );
        if (preparingVersionRef.current !== version) return;
        if (decision === "ready") {
          patchPreparation({ shell: "done", shellProgress: 100 });
          return;
        }
        patchPreparation({ shell: "downloading" });
        if (decision === "wait" && Date.now() < waitUntil) {
          // A native download this renderer did not start; the shell refuses a
          // second one, so wait it out rather than fail the preparation on it.
          if (!unlisten) {
            unlisten = await listenDesktopUpdateDownload(version, (percent) => {
              if (preparingVersionRef.current !== version) return;
              patchPreparation({ shellProgress: percent });
            });
            if (preparingVersionRef.current !== version) return;
          }
          await wait(BUNDLE_DOWNLOAD_POLL_MS);
          if (preparingVersionRef.current !== version) return;
          continue;
        }
        // Either nothing is in flight, or the wait above ran out: a native
        // download that stalls without ever clearing the flag would otherwise
        // hold the offer at "preparing" with no Restart button and no way back.
        // Handing it to download_desktop_update either takes the download over
        // or refuses, and a refusal puts the plain Update button back.
        //
        // downloadDesktopUpdate verifies the bundle it produced and throws when
        // it is missing or the wrong version, so one successful call is the whole
        // download. Looping back to re-read the status instead would spin forever
        // against a native side that keeps reporting nothing downloaded.
        await downloadDesktopUpdate(version, (percent) => {
          if (preparingVersionRef.current !== version) return;
          patchPreparation({ shellProgress: percent });
        });
        if (preparingVersionRef.current !== version) return;
        patchPreparation({ shell: "done", shellProgress: 100 });
        return;
      }
    } catch (e) {
      console.warn("Background app download failed:", e);
      if (preparingVersionRef.current !== version) return;
      patchPreparation({ shell: "failed" });
    } finally {
      unlisten?.();
    }
  }

  async function prepareBackend(version: string): Promise<void> {
    try {
      const decision = prefetchDecision({
        inApp: true,
        isExternalServer,
        offeredVersion: version,
        prefetch: await prefetchStatus(),
      });
      if (preparingVersionRef.current !== version) return;
      if (decision === "skip") {
        patchPreparation({ backend: "skipped" });
        return;
      }
      if (decision === "already-ready") {
        patchPreparation({ backend: "ready" });
        return;
      }
      patchPreparation({ backend: "prefetching" });
      if (decision === "adopt") {
        const settled = await adoptPrefetch(
          () => preparingVersionRef.current !== version,
        );
        if (preparingVersionRef.current !== version) return;
        patchPreparation({
          backend: settled.state === "none" || settled.state === "stale"
            ? "failed"
            : "ready",
        });
        return;
      }
      if (decision === "restart") {
        await cancelPrefetch().catch(() => {});
        if (preparingVersionRef.current !== version) return;
      }
      const outcome = await startPrefetch(version, appendLog);
      if (preparingVersionRef.current !== version) return;
      patchPreparation({
        backend:
          outcome === "ready"
            ? "ready"
            : // A backend without the command, or a run somebody else owns:
              // neither is a fault, and both leave the restart able to proceed.
              outcome === "failed"
              ? "failed"
              : "skipped",
      });
    } catch (e) {
      console.warn("Background update preparation failed:", e);
      if (preparingVersionRef.current !== version) return;
      patchPreparation({ backend: "failed" });
    }
  }

  async function ensureBundleDownloaded(): Promise<void> {
    setUpdatePhase("shell_download");
    updateStatus("downloading");
    setUpdateProgress(0);
    const version = updateRef.current?.version;
    if (!version) throw new Error("No desktop update has been checked.");
    // Attached only once a download is in flight and released whichever way the wait ends.
    let unlisten: (() => void) | null = null;
    const waitUntil = Date.now() + BUNDLE_DOWNLOAD_WAIT_MS;
    try {
      for (;;) {
        // A bundle retained by an earlier attempt is reused, so a retry usually stops here.
        const bundle = await desktopUpdateBundleStatus();
        if (bundle.downloaded && sameUpdateVersion(bundle.version, version)) {
          setUpdateProgress(100);
          return;
        }
        // A webview reload leaves the native download running with no listener, and a second one is refused.
        if (!bundle.downloading) break;
        if (Date.now() >= waitUntil) break;
        if (!unlisten) {
          unlisten = await listenDesktopUpdateDownload(version, setUpdateProgress);
        }
        await new Promise((resolve) =>
          setTimeout(resolve, BUNDLE_DOWNLOAD_POLL_MS),
        );
      }
    } finally {
      unlisten?.();
    }
    await downloadDesktopUpdate(version, setUpdateProgress);
  }

  async function installUpdate() {
    if (updatingRef.current) return;
    // "preparing" has no install to be in the middle of, and its own press is
    // ignored by the button being disabled.
    if (statusRef.current === "preparing") return;
    updatingRef.current = true;

    const cleanups: (() => void)[] = [];
    try {
      // A retry re-enters here, and start_backend_update spawns a mutating child of its own.
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
        // First press: fetch in the background and leave the app running. The
        // offer becomes "Restart", which is the press that installs.
        void prepareUpdate(update.version);
        return;
      }

      const { invoke } = await import("@tauri-apps/api/core");
      // From here on this IS the update, so nothing may be preparing beside it.
      // start_backend_update stops a running prefetch too; asking first keeps the
      // renderer's own record straight and makes the stop deterministic in tests.
      preparingVersionRef.current = null;
      await cancelPrefetch().catch(() => {});
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

      // `update::is_update_running` is already false here, and quitting mid-install leaves a half-updated app.
      publishShellUpdateActive(true);
      try {
        await installDesktopUpdate();
      } catch (installError) {
        // Failed or cancelled: we keep running, so the stood-down cleanup has to come back.
        await resumeCleanup();
        throw installError;
      } finally {
        publishShellUpdateActive(false);
      }

      // Deliberately NOT re-arming kill-on-close before the restart: relaunch() starts the replacement as a child,
      // which inherits this job. The handoff stays in the recovery scope, or a throw leaves cleanup stood down.
      try {
        // relaunch() re-execs with the original argv, so flag the inherited --hidden as not a login start.
        await invoke("mark_in_app_relaunch");
        const { relaunch } = await import("@tauri-apps/plugin-process");
        await relaunch();
      } catch (relaunchError) {
        // No replacement process, so the marker would outlive it and unhide a later login start.
        await invoke("clear_in_app_relaunch").catch(() => {});
        await resumeCleanup();
        throw relaunchError;
      }
    } catch (e) {
      console.error("Update failed:", e);
      const msg = String(e);

      if (phaseRef.current === "shell_download" || phaseRef.current === "shell_install") {
        // A backend started under a job with kill-on-close disabled is the orphan this prevents.
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
        // The one answer we cannot assume on the desktop: fail closed and let the gate re-arm.
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
    // Same gate as the recovery path: this is offered on every error.
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

  const manualReleaseUrl =
    updatePolicy.mode === "manual_linux_package" && info
      ? manualReleasePageUrl(updatePolicy, info.version)
      : null;
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
    preparation,
    lastFailure,
    isExternalServer,
    updatePolicyMode: updatePolicy.mode,
    manualReleaseUrl,
    releasePageUrl,
    checkForUpdate,
    prepareUpdate,
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
