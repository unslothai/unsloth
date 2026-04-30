// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useRef, useState } from "react";
import { isTauri } from "@/lib/api-base";
import {
  copySupportDiagnostics,
  type CopySupportDiagnosticsResult,
} from "@/lib/tauri-diagnostics";
import { toast } from "sonner";

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
  body?: string;
  date?: string;
}

export type UpdatePhase =
  | "backend"
  | "shell_download"
  | "shell_install"
  | "recovered_after_shell_failure";

export interface RetainedUpdateFailure {
  error: string;
  phase: UpdatePhase;
  progress: number;
  logs: string[];
}

export function useTauriUpdate(isExternalServer = false) {
  const [status, setStatus] = useState<UpdateStatus>("idle");
  const [info, setInfo] = useState<UpdateInfo | null>(null);
  const [progress, setProgress] = useState(0);
  const progressRef = useRef(0);
  const [logs, setLogs] = useState<string[]>([]);
  const logsRef = useRef<string[]>([]);
  const [phase, setPhase] = useState<UpdatePhase | null>(null);
  const phaseRef = useRef<UpdatePhase | null>(null);
  const [dismissed, setDismissed] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastFailure, setLastFailure] = useState<RetainedUpdateFailure | null>(null);
  const updateRef = useRef<Awaited<
    ReturnType<typeof import("@tauri-apps/plugin-updater").check>
  > | null>(null);
  const checkedRef = useRef(false);
  const updatingRef = useRef(false);

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

  useEffect(() => {
    if (!isTauri || checkedRef.current) return;
    checkedRef.current = true;

    async function checkForUpdate() {
      setStatus("checking");
      try {
        const { check } = await import("@tauri-apps/plugin-updater");
        const update = await check();
        if (update) {
          updateRef.current = update;
          setInfo({
            version: update.version,
            currentVersion: update.currentVersion,
            body: update.body,
            date: update.date,
          });
          setStatus("available");
        } else {
          setStatus("idle");
        }
      } catch (e) {
        console.error("Update check failed:", e);
        setStatus("idle");
      }
    }

    const timer = setTimeout(checkForUpdate, 5000);
    return () => clearTimeout(timer);
  }, []);

  async function installUpdate() {
    const update = updateRef.current;
    if (!update || updatingRef.current) return;
    updatingRef.current = true;

    const cleanups: (() => void)[] = [];

    try {
      // ── Step 1: Backend update ──
      setUpdatePhase("backend");
      setStatus("updating-backend");
      replaceLogs([]);
      setUpdateProgress(0);
      setError(null);
      setLastFailure(null);
      setDismissed(false);

      const { listen } = await import("@tauri-apps/api/event");
      const { invoke } = await import("@tauri-apps/api/core");

      // Listen for backend update progress
      const unlistenProgress = await listen<string>(
        "update-progress",
        (e) => {
          appendLog(e.payload);
        },
      );
      cleanups.push(unlistenProgress);

      // Wait for complete or failed
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
        updatingRef.current = false;
        cleanup(cleanups);
        return;
      }

      // ── Step 2: Shell update ──
      setUpdatePhase("shell_download");
      setStatus("downloading");
      setUpdateProgress(0);

      let downloaded = 0;
      let contentLength = 0;
      await update.downloadAndInstall((event) => {
        switch (event.event) {
          case "Started":
            contentLength = event.data.contentLength ?? 0;
            break;
          case "Progress":
            downloaded += event.data.chunkLength;
            if (contentLength > 0) {
              setUpdateProgress(Math.round((downloaded / contentLength) * 100));
            }
            break;
          case "Finished":
            setUpdatePhase("shell_install");
            setStatus("installing");
            break;
        }
      });

      // ── Step 3: Relaunch ──
      const { relaunch } = await import("@tauri-apps/plugin-process");
      await relaunch();
    } catch (e) {
      console.error("Update failed:", e);
      const msg = String(e);

      // Shell update failed — restart backend on updated code
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

  return {
    status,
    info,
    progress,
    logs,
    dismissed,
    error,
    phase,
    lastFailure,
    isExternalServer,
    installUpdate,
    retryUpdate,
    skipAndRestart,
    dismiss,
    copyDiagnostics,
  };
}

function cleanup(fns: (() => void)[]) {
  for (const fn of fns) {
    try {
      fn();
    } catch {
      // ignore
    }
  }
}
