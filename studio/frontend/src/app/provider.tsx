// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { StartupScreen } from "@/components/tauri/startup-screen";
import { UpdateBanner } from "@/components/tauri/update-banner";
import { UpdateScreen } from "@/components/tauri/update-screen";
import {
  WindowTitlebar,
  shouldUseCustomWindowTitlebar,
} from "@/components/tauri/window-titlebar";
import { Toaster } from "@/components/ui/sonner";
import { getTauriAuthFailure, tauriAutoAuth } from "@/features/auth";
import { useTauriBackend, type BackendStatus } from "@/hooks/use-tauri-backend";
import { useTauriUpdate } from "@/hooks/use-tauri-update";
import { isTauri } from "@/lib/api-base";
import { useRouterState } from "@tanstack/react-router";
import { ThemeProvider } from "next-themes";
import { useEffect, useRef, useState, type ReactNode } from "react";

interface AppProviderProps {
  children: ReactNode;
}

// ---------------------------------------------------------------------------
// Tauri window helpers (only imported in Tauri mode)
// ---------------------------------------------------------------------------

type TauriWindowMode = "setup" | "app";

async function showSetupWindow(): Promise<void> {
  const { getCurrentWindow } = await import("@tauri-apps/api/window");
  const win = getCurrentWindow();
  await win.center();
  await win.show();
}

async function applyAppWindowLayout(): Promise<void> {
  const { getCurrentWindow, currentMonitor, LogicalSize } = await import("@tauri-apps/api/window");
  const win = getCurrentWindow();
  const monitor = await currentMonitor();

  if (monitor) {
    // Convert physical pixels to logical using scale factor
    const scale = monitor.scaleFactor;
    const screenW = monitor.size.width / scale;
    const screenH = monitor.size.height / scale;

    // Target: 75% of screen width, golden ratio height, capped at min 900x600
    const targetW = Math.max(900, Math.round(screenW * 0.75));
    const targetH = Math.max(600, Math.round(targetW / 1.618));
    // Don't exceed screen height
    const finalH = Math.min(targetH, Math.round(screenH * 0.85));
    const finalW = targetW;

    await win.setSize(new LogicalSize(finalW, finalH));
  }

  // Apply constraints and finalize without animating through intermediate sizes
  await win.setSizeConstraints({ minWidth: 900, minHeight: 600 });
  await win.setResizable(true);
  await win.center();
  await win.show();
}

async function showWindowFallback(): Promise<void> {
  const { getCurrentWindow } = await import("@tauri-apps/api/window");
  const win = getCurrentWindow();
  await win.setResizable(true);
  await win.show();
}

function getTauriWindowMode(
  status: BackendStatus,
  hasEnteredAppMode: boolean,
): TauriWindowMode | null {
  switch (status) {
    case "checking":
      return null;
    case "not-installed":
    case "installing":
    case "install-error":
    case "needs-elevation":
    case "repairing":
    case "repair-error":
      return "setup";
    case "starting":
    case "running":
    case "stopped":
      return "app";
    case "error":
      return hasEnteredAppMode ? "app" : "setup";
  }
}

// ---------------------------------------------------------------------------
// TauriWrapper
// ---------------------------------------------------------------------------

function TauriUpdateLayer({ isExternalServer }: { isExternalServer: boolean }) {
  const update = useTauriUpdate(isExternalServer);
  const isUpdating =
    update.status === "updating-backend" ||
    update.status === "downloading" ||
    update.status === "installing" ||
    (update.status === "error" && !update.dismissed);

  if (isUpdating) {
    return (
      <UpdateScreen
        status={update.status}
        logs={update.logs}
        progress={update.progress}
        error={update.error}
        onRetry={update.retryUpdate}
        onSkipRestart={update.skipAndRestart}
        onCopyDiagnostics={update.copyDiagnostics}
      />
    );
  }

  return (
    <UpdateBanner
      status={update.status}
      info={update.info}
      dismissed={update.dismissed}
      lastFailure={update.lastFailure}
      isExternalServer={isExternalServer}
      onInstall={update.installUpdate}
      onDismiss={update.dismiss}
      onCopyDiagnostics={update.copyDiagnostics}
    />
  );
}

const HIDDEN_TITLEBAR_SIDEBAR_ROUTES = new Set([
  "/onboarding",
  "/login",
  "/change-password",
  "/signup",
]);

function TauriWrapper({ children }: { children: ReactNode }) {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const {
    status, logs, error, isExternalServer,
    currentStepIndex, progressDetail, elevationPackages,
    startInstall, retry, retryInstall, approveElevation, copyDiagnostics,
  } = useTauriBackend();

  const appliedWindowModeRef = useRef<TauriWindowMode | null>(null);
  const hasEnteredAppModeRef = useRef(false);
  const [desktopAuthReady, setDesktopAuthReady] = useState(!isTauri);
  const [desktopAuthRetry, setDesktopAuthRetry] = useState(0);

  // Keep the Tauri window hidden during preflight, then show it centered in setup
  // mode or apply the final app layout in one instant step.
  useEffect(() => {
    if (!isTauri) return;

    const nextMode = getTauriWindowMode(status, hasEnteredAppModeRef.current);
    if (!nextMode || appliedWindowModeRef.current === nextMode) return;

    appliedWindowModeRef.current = nextMode;
    if (nextMode === "app") hasEnteredAppModeRef.current = true;

    const applyWindowMode = nextMode === "setup" ? showSetupWindow : applyAppWindowLayout;
    applyWindowMode().catch(async () => {
      // On failure, at minimum make the window visible and resizable so user can fix manually.
      try {
        await showWindowFallback();
      } catch { /* swallow — window may still be functional */ }
    });
  }, [status]);

  useEffect(() => {
    if (!isTauri) {
      setDesktopAuthReady(true);
      return;
    }
    if (status !== "running") {
      setDesktopAuthReady(false);
      setDesktopAuthRetry(0);
      return;
    }

    let disposed = false;
    setDesktopAuthReady(false);
    tauriAutoAuth({ force: true }).then((authenticated) => {
      if (disposed) return;
      if (authenticated) {
        setDesktopAuthReady(true);
        return;
      }
      if (!getTauriAuthFailure()) {
        window.setTimeout(() => {
          if (!disposed) setDesktopAuthRetry((value) => value + 1);
        }, 500);
      }
    });

    return () => { disposed = true; };
  }, [status, desktopAuthRetry]);

  if (!isTauri) return <>{children}</>;

  const showApp = status === "running" && desktopAuthReady;
  const startupStatus = status === "running" ? "starting" : status;
  const startupProgressDetail =
    status === "running" && !desktopAuthReady
      ? "Signing in to desktop session..."
      : progressDetail;

  const content = showApp ? (
    <>
      <TauriUpdateLayer isExternalServer={isExternalServer} />
      {children}
    </>
  ) : (
    <StartupScreen
      status={startupStatus}
      logs={logs}
      error={error}
      currentStepIndex={currentStepIndex}
      progressDetail={startupProgressDetail}
      elevationPackages={elevationPackages}
      onInstall={startInstall}
      onRetry={retry}
      onRetryInstall={retryInstall}
      onApproveElevation={approveElevation}
      onStartServer={retry}
      onCopyDiagnostics={copyDiagnostics}
    />
  );

  if (!shouldUseCustomWindowTitlebar()) return content;

  const showSidebarSurface =
    showApp && !HIDDEN_TITLEBAR_SIDEBAR_ROUTES.has(pathname);

  return (
    <div className="flex h-dvh min-h-0 flex-col overflow-hidden bg-background [--studio-titlebar-height:34px]">
      <WindowTitlebar showSidebarSurface={showSidebarSurface} />
      <div className="min-h-0 flex-1 overflow-hidden">
        {content}
      </div>
    </div>
  );
}

export function AppProvider({ children }: AppProviderProps) {
  return (
    <ThemeProvider attribute="class" defaultTheme="light">
      <TauriWrapper>
        {children}
      </TauriWrapper>
      <Toaster position="top-right" visibleToasts={2} expand={true} />
    </ThemeProvider>
  );
}
