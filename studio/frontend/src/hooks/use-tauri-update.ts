// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useRef, useState } from "react";
import { isTauri } from "@/lib/api-base";
import {
  copySupportDiagnostics,
  type CopySupportDiagnosticsResult,
} from "@/lib/tauri-diagnostics";
import { checkDesktopUpdate } from "@/lib/tauri-updater";
import { toast } from "@/lib/toast";

export type UpdateStatus =
  | "idle"
  | "checking"
  | "available"
  | "updating-backend"
  | "downloading"
  | "installing"
  | "error";

export interface UpdateInfo {
  version: string;
  currentVersion: string;
  // Backend release this build pins; CHANGELOG.md is keyed by it, not the SemVer.
  pypiVersion?: string;
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
  releaseTagPrefix: "desktop-v",
};

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

export function useTauriUpdate(isExternalServer = false) {
  const [status, setStatus] = useState<UpdateStatus>("idle");
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
  const updateRef = useRef<Awaited<ReturnType<typeof checkDesktopUpdate>>>(null);
  const checkedRef = useRef(false);
  const startupScheduledRef = useRef(false);
  const checkingRef = useRef(false);
  const updatingRef = useRef(false);

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
    setStatus("available");
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
    checkingRef.current = true;
    setCheckError(null);
    setStatus("checking");

    try {
      const { policy, resolved } = await resolveUpdatePolicy();

      if (policy.mode === "manual_linux_package") {
        // Self-gates on the real target_os, so it is authoritative even if policy is a guess.
        if (await checkManualUpdate(policy)) return;
        if (resolved) {
          // latest.json has no deb/rpm key, so the in-app updater would offer an
          // AppImage this install cannot apply. Stop instead.
          updateRef.current = null;
          replaceInfo(null);
          setStatus("idle");
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
        replaceInfo(null);
        setStatus("idle");
      }
    } catch (e) {
      console.error("Update check failed:", e);
      setCheckError(String(e));
      setStatus(infoRef.current ? "available" : "idle");
    } finally {
      checkingRef.current = false;
      setHasChecked(true);
    }
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

  async function installUpdate() {
    if (updatingRef.current) return;
    updatingRef.current = true;

    const cleanups: (() => void)[] = [];

    try {
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

      setUpdatePhase("backend");
      setStatus("updating-backend");
      replaceLogs([]);
      setUpdateProgress(0);
      setError(null);
      setCheckError(null);
      setLastFailure(null);
      setDismissed(false);

      const { listen } = await import("@tauri-apps/api/event");
      const { invoke } = await import("@tauri-apps/api/core");

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
        setStatus("error");
        return;
      }

      setUpdatePhase("shell_download");
      setStatus("downloading");
      setUpdateProgress(0);

      let downloaded = 0;
      let contentLength = 0;
      // `update::is_update_running` is already false here, and quitting mid-install
      // leaves a half-updated app.
      publishShellUpdateActive(true);
      try {
        await update.downloadAndInstall((event) => {
          switch (event.event) {
            case "Started":
              contentLength = event.data.contentLength ?? 0;
              break;
            case "Progress":
              downloaded += event.data.chunkLength;
              if (contentLength > 0) {
                setUpdateProgress(
                  Math.round((downloaded / contentLength) * 100),
                );
              }
              break;
            case "Finished":
              setUpdatePhase("shell_install");
              setStatus("installing");
              break;
          }
        });
      } finally {
        // Success, failure and cancel alike: the installer is no longer running.
        publishShellUpdateActive(false);
      }

      const { relaunch } = await import("@tauri-apps/plugin-process");
      await relaunch();
    } catch (e) {
      console.error("Update failed:", e);
      const msg = String(e);

      // Shell update failed, so restart the backend on the updated code.
      if (phaseRef.current === "shell_download" || phaseRef.current === "shell_install") {
        try {
          const { invoke } = await import("@tauri-apps/api/core");
          await invoke("start_server", { port: 8888 });
          retainFailure(msg, "recovered_after_shell_failure");
          toast.error("App update failed", {
            description:
              "Backend was updated. Copy diagnostics from the update banner if you need support.",
          });
          setError(null);
          setStatus("idle");
          setDismissed(false);
          setUpdatePhase("recovered_after_shell_failure");
        } catch {
          retainFailure(msg, phaseRef.current ?? "shell_install");
          setError(msg);
          setStatus("error");
        }
      } else {
        retainFailure(msg, phaseRef.current ?? "backend");
        setError(msg);
        setStatus("error");
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

  async function skipAndRestart() {
    const skippedError = error;
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      await invoke("start_server", { port: 8888 });
      if (skippedError) {
        retainFailure(skippedError, phaseRef.current ?? "recovered_after_shell_failure");
        setDismissed(false);
      } else {
        setDismissed(true);
      }
      setStatus("idle");
      setError(null);
      replaceLogs([]);
    } catch (e) {
      const msg = String(e);
      retainFailure(msg, phaseRef.current ?? "backend");
      setError(msg);
      setStatus("error");
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
