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
import { WebUpdateBanner } from "@/components/web/update-banner";
import { getTauriAuthFailure, tauriAutoAuth } from "@/features/auth";
import { NativeIntentDrain } from "@/features/native-intents/native-intent-drain";
import { useTauriBackend, type BackendStatus } from "@/hooks/use-tauri-backend";
import { useTauriUpdate } from "@/hooks/use-tauri-update";
import { isTauri } from "@/lib/api-base";
import { useRouterState } from "@tanstack/react-router";
import { ThemeProvider } from "next-themes";
import { useEffect, useRef, useState, type ReactNode } from "react";

interface AppProviderProps {
  children: ReactNode;
}

type TauriWindowMode = "setup" | "app";
type WindowLayoutGuard = () => boolean;

async function showSetupWindow(isCurrent: WindowLayoutGuard): Promise<void> {
  const { getCurrentWindow } = await import("@tauri-apps/api/window");
  if (!isCurrent()) return;

  const win = getCurrentWindow();
  if (!isCurrent()) return;
  await win.center();
  if (!isCurrent()) return;
  await win.show();
}

async function applyAppWindowLayout(isCurrent: WindowLayoutGuard): Promise<void> {
  const { getCurrentWindow, currentMonitor, LogicalSize } = await import("@tauri-apps/api/window");
  if (!isCurrent()) return;

  const win = getCurrentWindow();
  const monitor = await currentMonitor();
  if (!isCurrent()) return;

  let finalW = 900;
  let finalH = 600;

  if (monitor) {
    const scale = monitor.scaleFactor;
    const screenW = monitor.size.width / scale;
    const screenH = monitor.size.height / scale;

    finalW = Math.max(900, Math.round(screenW * 0.75));
    const targetH = Math.max(600, Math.round(finalW / 1.618));
    finalH = Math.min(targetH, Math.round(screenH * 0.85));
  }

  if (!isCurrent()) return;
  await win.setSize(new LogicalSize(finalW, finalH));
  if (!isCurrent()) return;
  await win.setSizeConstraints({ minWidth: 900, minHeight: 600 });
  if (!isCurrent()) return;
  await win.setResizable(true);
  if (!isCurrent()) return;
  await win.center();
  if (!isCurrent()) return;
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
      updatePolicyMode={update.updatePolicyMode}
      manualReleaseUrl={update.manualReleaseUrl}
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

const WEB_UPDATE_HIDDEN_ROUTES = new Set([
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
  const windowLayoutGenerationRef = useRef(0);
  const [desktopAuthReady, setDesktopAuthReady] = useState(!isTauri);
  const [desktopAuthRetry, setDesktopAuthRetry] = useState(0);

  useEffect(() => {
    if (!isTauri) return;
    return () => {
      windowLayoutGenerationRef.current += 1;
      appliedWindowModeRef.current = null;
    };
  }, []);

  // Keep the Tauri window hidden until setup or app layout is ready.
  useEffect(() => {
    if (!isTauri) return;

    const nextMode = getTauriWindowMode(status, hasEnteredAppModeRef.current);
    if (!nextMode) {
      appliedWindowModeRef.current = null;
      windowLayoutGenerationRef.current += 1;
      return;
    }
    if (appliedWindowModeRef.current === nextMode) return;

    appliedWindowModeRef.current = nextMode;
    if (nextMode === "app") hasEnteredAppModeRef.current = true;

    const layoutGeneration = windowLayoutGenerationRef.current + 1;
    windowLayoutGenerationRef.current = layoutGeneration;
    const isCurrent = () => windowLayoutGenerationRef.current === layoutGeneration;
    const applyWindowMode = nextMode === "setup" ? showSetupWindow : applyAppWindowLayout;
    applyWindowMode(isCurrent).catch(async () => {
      if (!isCurrent()) return;
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

  if (!isTauri) {
    return (
      <>
        {children}
        <WebUpdateBanner enabled={!WEB_UPDATE_HIDDEN_ROUTES.has(pathname)} />
      </>
    );
  }

  const showApp = status === "running" && desktopAuthReady;
  const startupStatus = status === "running" ? "starting" : status;
  const startupProgressDetail =
    status === "running" && !desktopAuthReady
      ? "Signing in to desktop session..."
      : progressDetail;

  const content = showApp ? (
    <>
      <TauriUpdateLayer isExternalServer={isExternalServer} />
      <NativeIntentDrain />
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
