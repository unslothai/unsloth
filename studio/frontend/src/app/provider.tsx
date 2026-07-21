// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { LlamaUpdateBanner } from "@/components/llama-update-banner";
import { StartupScreen } from "@/components/tauri/startup-screen";
import { UpdateBanner } from "@/components/tauri/update-banner";
import { UpdateScreen } from "@/components/tauri/update-screen";
import {
  WindowTitlebar,
  shouldUseCustomWindowTitlebar,
  shouldUseNativeMacWindowTitlebar,
} from "@/components/tauri/window-titlebar";
import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { WebUpdateBanner } from "@/components/web/update-banner";
import { fetchDeviceType } from "@/config/env";
import { getTauriAuthFailure, tauriAutoAuth } from "@/features/auth";
import { DownloadManagerPanel } from "@/features/hub/download-manager";
import { NativeIntentDrain } from "@/features/native-intents/native-intent-drain";
import {
  applyCustomizationToDocument,
  useAppearanceCustomStore,
  useTheme,
} from "@/features/settings";
import { type BackendStatus, useTauriBackend } from "@/hooks/use-tauri-backend";
import { useTauriUpdate } from "@/hooks/use-tauri-update";
import { isTauri } from "@/lib/api-base";
import { useRouterState } from "@tanstack/react-router";
import { MotionConfig } from "motion/react";
import {
  type CSSProperties,
  type ReactNode,
  useEffect,
  useRef,
  useState,
} from "react";

interface AppProviderProps {
  children: ReactNode;
}

type TauriWindowMode = "setup" | "app";
type WindowLayoutGuard = () => boolean;

const MIN_WINDOW_WIDTH = 900;
const MIN_WINDOW_HEIGHT = 600;
const SETUP_WINDOW_WIDTH = 760;
const SETUP_WINDOW_HEIGHT = 560;

async function showSetupWindow(isCurrent: WindowLayoutGuard): Promise<void> {
  const { getCurrentWindow, LogicalSize } = await import(
    "@tauri-apps/api/window"
  );
  const { invoke } = await import("@tauri-apps/api/core");
  if (!isCurrent()) return;

  const win = getCurrentWindow();
  await invoke("reset_app_window_layout_initialized");
  if (!isCurrent()) return;
  await win.setResizable(false);
  if (!isCurrent()) return;
  await win.setSize(new LogicalSize(SETUP_WINDOW_WIDTH, SETUP_WINDOW_HEIGHT));
  if (!isCurrent()) return;
  await win.center();
  if (!isCurrent()) return;
  await win.show();
}

async function enforceMinimumWindowSize(
  win: Awaited<
    ReturnType<typeof import("@tauri-apps/api/window")["getCurrentWindow"]>
  >,
  LogicalSize: typeof import("@tauri-apps/api/window")["LogicalSize"],
  isCurrent: WindowLayoutGuard,
): Promise<void> {
  const [innerSize, scaleFactor] = await Promise.all([
    win.innerSize(),
    win.scaleFactor(),
  ]);
  if (!isCurrent()) return;

  const logicalWidth = Math.round(innerSize.width / scaleFactor);
  const logicalHeight = Math.round(innerSize.height / scaleFactor);
  const nextWidth = Math.max(logicalWidth, MIN_WINDOW_WIDTH);
  const nextHeight = Math.max(logicalHeight, MIN_WINDOW_HEIGHT);
  if (nextWidth !== logicalWidth || nextHeight !== logicalHeight) {
    await win.setSize(new LogicalSize(nextWidth, nextHeight));
  }
}

async function applyAppWindowLayout(
  isCurrent: WindowLayoutGuard,
): Promise<void> {
  const { getCurrentWindow, currentMonitor, LogicalSize } = await import(
    "@tauri-apps/api/window"
  );
  const { invoke } = await import("@tauri-apps/api/core");
  const { restoreStateCurrent, StateFlags } = await import(
    "@tauri-apps/plugin-window-state"
  );
  if (!isCurrent()) return;

  const win = getCurrentWindow();
  // Setup-window activity may create plugin state before the full app is ever
  // shown, so use a dedicated full-app marker to decide whether restoration is
  // appropriate. Keep checking plugin state so a missing/corrupt state file
  // falls back to a monitor-safe centered layout.
  const [hasInitializedAppLayout, hasSavedState] = await Promise.all([
    invoke<boolean>("has_initialized_app_window_layout"),
    invoke<boolean>("has_saved_window_state"),
  ]);
  if (!isCurrent()) return;

  await win.setResizable(true);
  if (!isCurrent()) return;

  if (hasInitializedAppLayout && hasSavedState) {
    // Subsequent launch: plugin restores size/position/maximized, with built-in
    // off-screen protection for positions saved on a now-disconnected display.
    await restoreStateCurrent(
      StateFlags.SIZE | StateFlags.POSITION | StateFlags.MAXIMIZED,
    );
  } else {
    // First launch: fit to the current monitor and center.
    const monitor = await currentMonitor();
    if (!isCurrent()) return;
    let finalW = MIN_WINDOW_WIDTH;
    let finalH = MIN_WINDOW_HEIGHT;
    if (monitor) {
      const scale = monitor.scaleFactor;
      const screenW = monitor.size.width / scale;
      const screenH = monitor.size.height / scale;
      finalW = Math.max(MIN_WINDOW_WIDTH, Math.round(screenW * 0.75));
      const targetH = Math.max(MIN_WINDOW_HEIGHT, Math.round(finalW / 1.618));
      finalH = Math.min(targetH, Math.round(screenH * 0.85));
    }
    await win.setSize(new LogicalSize(finalW, finalH));
    if (!isCurrent()) return;
    await win.center();
  }
  if (!isCurrent()) return;
  await win.show();
  if (!isCurrent()) return;
  // Apply constraints after restore/show: doing so before plugin restore can emit
  // a Resized event and overwrite the plugin's cached saved size.
  await win.setSizeConstraints({
    minWidth: MIN_WINDOW_WIDTH,
    minHeight: MIN_WINDOW_HEIGHT,
  });
  if (!isCurrent()) return;
  await enforceMinimumWindowSize(win, LogicalSize, isCurrent);

  if (!isCurrent()) return;
  await invoke("mark_app_window_layout_initialized");
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

function TauriUpdateLayer({
  isExternalServer,
  children,
}: {
  isExternalServer: boolean;
  children?: ReactNode;
}) {
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
    <div className="pointer-events-none fixed bottom-4 right-4 z-[9998] flex w-[calc(100vw-2rem)] max-w-[400px] flex-col items-stretch gap-2">
      <UpdateBanner
        status={update.status}
        info={update.info}
        dismissed={update.dismissed}
        lastFailure={update.lastFailure}
        isExternalServer={isExternalServer}
        updatePolicyMode={update.updatePolicyMode}
        manualReleaseUrl={update.manualReleaseUrl}
        positioned={false}
        onInstall={update.installUpdate}
        onDismiss={update.dismiss}
        onCopyDiagnostics={update.copyDiagnostics}
      />
      {children}
    </div>
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

const MAC_NATIVE_CHROME_STYLE = {
  "--studio-titlebar-height": "0px",
  "--studio-mac-titlebar-height": "34px",
  "--studio-mac-traffic-light-inset": "78px",
  "--studio-startup-top-inset": "58px",
  "--studio-content-top-inset": "0px",
  "--studio-non-chat-content-top-inset": "34px",
  "--studio-hidden-route-top-inset": "34px",
  "--studio-chat-header-height": "44px",
  "--studio-chat-header-padding-top": "8px",
  "--studio-chat-control-height": "33px",
  "--studio-chat-header-right-inset": "0px",
} as CSSProperties;

const CUSTOM_CHROME_STYLE = {
  "--studio-titlebar-height": "0px",
  "--studio-custom-titlebar-height": "34px",
  "--studio-sidebar-expanded-width": "17.5rem",
  "--studio-sidebar-collapsed-width": "3rem",
  "--studio-startup-top-inset": "42px",
  "--studio-content-top-inset": "34px",
  "--studio-hidden-route-top-inset": "34px",
  "--studio-chat-header-height": "48px",
  "--studio-chat-header-padding-top": "9px",
  "--studio-chat-control-height": "33px",
  "--studio-chat-header-right-inset": "0px",
  "--studio-window-control-inset": "112px",
} as CSSProperties;

function TauriWrapper({ children }: { children: ReactNode }) {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const {
    status,
    logs,
    error,
    isExternalServer,
    currentStepIndex,
    progressDetail,
    elevationPackages,
    startInstall,
    retry,
    retryInstall,
    approveElevation,
    copyDiagnostics,
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
    const isCurrent = () =>
      windowLayoutGenerationRef.current === layoutGeneration;
    const applyWindowMode =
      nextMode === "setup" ? showSetupWindow : applyAppWindowLayout;
    applyWindowMode(isCurrent).catch(async () => {
      if (!isCurrent()) return;
      // On failure, at minimum make the window visible and resizable so user can fix manually.
      try {
        await showWindowFallback();
      } catch {
        /* swallow; window may still be functional */
      }
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

    return () => {
      disposed = true;
    };
  }, [status, desktopAuthRetry]);

  useEffect(() => {
    if (!isTauri || status !== "running" || !desktopAuthReady) return;
    void fetchDeviceType({ force: true }).catch(() => undefined);
  }, [status, desktopAuthReady]);

  if (!isTauri) {
    return (
      <>
        {children}
        {/* One bottom-right stack so overlays never overlap; they stack with a
            gap, download panel anchored at the corner with banners above. */}
        <div className="pointer-events-none fixed bottom-4 right-4 z-[9998] flex w-[calc(100vw-2rem)] max-w-[400px] flex-col items-stretch gap-2">
          <WebUpdateBanner
            positioned={false}
            enabled={!WEB_UPDATE_HIDDEN_ROUTES.has(pathname)}
          />
          <LlamaUpdateBanner
            positioned={false}
            enabled={!WEB_UPDATE_HIDDEN_ROUTES.has(pathname)}
          />
          <DownloadManagerPanel positioned={false} />
        </div>
      </>
    );
  }

  const showApp = status === "running";
  const desktopBooting = status === "running" && !desktopAuthReady;
  const showInteractiveApp = showApp && desktopAuthReady;
  const startupStatus = status === "running" ? "starting" : status;
  const startupProgressDetail = progressDetail;
  const usesCustomTitlebar = shouldUseCustomWindowTitlebar();
  const usesNativeMacTitlebar = shouldUseNativeMacWindowTitlebar();
  const hidesTitlebarSidebar = HIDDEN_TITLEBAR_SIDEBAR_ROUTES.has(pathname);

  const content = showApp ? (
    <>
      <TauriUpdateLayer isExternalServer={isExternalServer}>
        <LlamaUpdateBanner
          positioned={false}
          enabled={showInteractiveApp && !hidesTitlebarSidebar}
        />
        {showInteractiveApp ? (
          <DownloadManagerPanel positioned={false} />
        ) : null}
      </TauriUpdateLayer>
      {showInteractiveApp ? <NativeIntentDrain /> : null}
      {showInteractiveApp ? children : null}
      {desktopBooting ? (
        <div className="pointer-events-none fixed inset-x-0 bottom-5 z-[9999] flex justify-center px-4">
          <div className="absolute inset-x-4 bottom-16 mx-auto flex max-w-[520px] flex-col items-center gap-2 rounded-2xl border border-border/70 bg-background/95 px-6 py-5 text-center shadow-xl">
            <div className="font-medium text-sm">Preparing Unsloth</div>
            <div className="text-muted-foreground text-xs">
              The local backend is ready. Signing in to your desktop session
              before loading chats.
            </div>
          </div>
          <div className="rounded-full border border-border/70 bg-background/95 px-4 py-2 text-xs text-muted-foreground shadow-lg">
            Signing in to desktop session...
          </div>
        </div>
      ) : null}
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

  if (!usesCustomTitlebar) {
    // macOS desktop uses the native titlebar and returns here before the
    // custom-titlebar branch, so mount the updater banner on this path too.
    if (usesNativeMacTitlebar) {
      return (
        <div
          className={
            hidesTitlebarSidebar
              ? "relative h-dvh min-h-0 overflow-x-hidden overflow-y-auto bg-background"
              : "relative h-dvh min-h-0 overflow-hidden bg-background"
          }
          style={MAC_NATIVE_CHROME_STYLE}
        >
          {!showApp || hidesTitlebarSidebar ? (
            <div
              data-tauri-drag-region={true}
              aria-hidden="true"
              className="pointer-events-auto fixed inset-x-0 top-0 z-50 h-[var(--studio-mac-titlebar-height,34px)] select-none"
            />
          ) : null}
          {content}
        </div>
      );
    }

    return <>{content}</>;
  }

  const showSidebarSurface = showApp && !hidesTitlebarSidebar;

  return (
    <div
      className="relative h-dvh min-h-0 overflow-hidden bg-background"
      style={CUSTOM_CHROME_STYLE}
    >
      <WindowTitlebar showSidebarSurface={showSidebarSurface} />
      <div className="h-full min-h-0 overflow-hidden">{content}</div>
    </div>
  );
}

/**
 * Mirrors the appearance customization store onto <html> (inline CSS vars,
 * classes, attributes). Colors are per resolved light/dark mode, so re-apply
 * whenever either the customization or the resolved theme changes.
 */
function AppearanceCustomizationEffect() {
  const { resolved } = useTheme();
  const customization = useAppearanceCustomStore((s) => s.customization);
  useEffect(() => {
    applyCustomizationToDocument(customization, resolved);
  }, [customization, resolved]);
  return null;
}

const REDUCED_MOTION_MAP = {
  system: "user",
  on: "always",
  off: "never",
} as const;

export function AppProvider({ children }: AppProviderProps) {
  const reduceMotion = useAppearanceCustomStore(
    (s) => s.customization.reduceMotion,
  );
  return (
    <MotionConfig reducedMotion={REDUCED_MOTION_MAP[reduceMotion]}>
      <TooltipProvider>
        <AppearanceCustomizationEffect />
        <TauriWrapper>{children}</TauriWrapper>
        <Toaster
          position="top-right"
          visibleToasts={2}
          expand={true}
          closeButton={true}
          // Clear the chat header buttons on the right.
          offset={{ top: 12, right: 64 }}
        />
      </TooltipProvider>
    </MotionConfig>
  );
}
