// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useRef, useState } from "react";
import { isTauri } from "@/lib/api-base";
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

export function useTauriUpdate(isExternalServer = false) {
  const [status, setStatus] = useState<UpdateStatus>("idle");
  const [info, setInfo] = useState<UpdateInfo | null>(null);
  const [progress, setProgress] = useState(0);
  const [logs, setLogs] = useState<string[]>([]);
  const [dismissed, setDismissed] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const updateRef = useRef<Awaited<
    ReturnType<typeof import("@tauri-apps/plugin-updater").check>
  > | null>(null);
  const checkedRef = useRef(false);
  const updatingRef = useRef(false);

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
    let phase: "backend" | "shell" = "backend";

    try {
      // ── Step 1: Backend update ──
      setStatus("updating-backend");
      setLogs([]);
      setError(null);
      setDismissed(false);

      const { listen } = await import("@tauri-apps/api/event");
      const { invoke } = await import("@tauri-apps/api/core");

      // Listen for backend update progress
      const unlistenProgress = await listen<string>(
        "update-progress",
        (e) => {
          setLogs((prev) => [...prev.slice(-499), e.payload]);
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
        setError(backendResult);
        setStatus("error");
        updatingRef.current = false;
        cleanup(cleanups);
        return;
      }

      // ── Step 2: Shell update ──
      phase = "shell";
      setStatus("downloading");
      setProgress(0);

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
              setProgress(Math.round((downloaded / contentLength) * 100));
            }
            break;
          case "Finished":
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
      if (phase === "shell") {
        try {
          const { invoke } = await import("@tauri-apps/api/core");
          await invoke("start_server", { port: 8888 });
          toast.error("App update failed", {
            description:
              "Backend was updated. The app update will be retried on next launch.",
          });
          setStatus("idle");
          setDismissed(true);
        } catch {
          setError(msg);
          setStatus("error");
        }
      } else {
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
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      await invoke("start_server", { port: 8888 });
      setStatus("idle");
      setError(null);
      setLogs([]);
      setDismissed(true);
    } catch (e) {
      setError(String(e));
      setStatus("error");
    }
  }

  function dismiss() {
    setDismissed(true);
  }

  return {
    status,
    info,
    progress,
    logs,
    dismissed,
    error,
    isExternalServer,
    installUpdate,
    retryUpdate,
    skipAndRestart,
    dismiss,
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
