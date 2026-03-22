// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState, useCallback } from "react";
import { isTauri, setApiBase } from "@/lib/api-base";

type BackendStatus =
  | "checking"
  | "not-installed"
  | "installing"
  | "starting"
  | "running"
  | "stopped"
  | "error";

export function useTauriBackend() {
  const [status, setStatus] = useState<BackendStatus>("checking");
  const [logs, setLogs] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

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
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      await invoke("start_server", { port: 8888 });
      // Poll health - scan port range since backend may auto-increment
      for (let i = 0; i < 120; i++) {
        for (let p = 8888; p <= 8899; p++) {
          const healthy = await invoke<boolean>("check_health", { port: p });
          if (healthy) {
            setApiBase(p);
            setStatus("running");
            return;
          }
        }
        await new Promise((r) => setTimeout(r, 500));
      }
      setStatus("error");
      setError("Backend did not start within 60 seconds");
    } catch (e) {
      setStatus("error");
      setError(String(e));
    }
  }

  async function stopServer() {
    const { invoke } = await import("@tauri-apps/api/core");
    await invoke("stop_server");
    setStatus("stopped");
  }

  async function startInstall() {
    setStatus("installing");
    setLogs([]);
    setError(null);
    const { invoke } = await import("@tauri-apps/api/core");
    try {
      await invoke("start_install");
    } catch (e) {
      setStatus("error");
      setError(String(e));
    }
  }

  const retry = useCallback(() => {
    setError(null);
    setLogs([]);
    checkInstallAndStart();
  }, []);

  // Initial check on mount
  useEffect(() => {
    if (!isTauri) {
      setStatus("running");
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

      listen<void>("install-complete", () => {
        setStatus("starting");
        startServer();
      }).then((u) => cleanup.push(u));

      listen<string>("install-failed", (e) => {
        setError(e.payload);
        setStatus("error");
      }).then((u) => cleanup.push(u));

      listen<number>("server-port", (e) => {
        setApiBase(e.payload);
      }).then((u) => cleanup.push(u));

      listen<void>("server-crashed", () => {
        setStatus("error");
        setError("Server stopped unexpectedly");
      }).then((u) => cleanup.push(u));

      listen<string>("server-log", (e) => {
        setLogs((prev) => [...prev.slice(-499), e.payload]);
      }).then((u) => cleanup.push(u));
    });

    return () => {
      cleanup.forEach((fn) => fn());
    };
  }, []);

  return { status, logs, error, startServer, stopServer, startInstall, retry };
}
