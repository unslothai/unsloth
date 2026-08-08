// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { LlamaUpdateBanner } from "@/components/llama-update-banner";
import {
  ClosingScreen,
  StartupScreen,
} from "@/components/tauri/startup-screen";
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
import { DeepLinkHandler } from "@/features/deep-links";
import { DownloadManagerPanel } from "@/features/hub/download-manager";
import { NativeIntentDrain } from "@/features/native-intents/native-intent-drain";
import {
  applyCustomizationToDocument,
  useAppearanceCustomStore,
  useTheme,
} from "@/features/settings";
import { SttDownloadPrompt } from "@/features/settings/components/stt-download-prompt";
import { TauriUpdateContext } from "@/hooks/tauri-update-context";
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
import {
  type LogicalWindowSize,
  PREFERRED_SETUP_WINDOW_SIZE,
  type WindowSizeBounds,
  calculateCenteredPosition,
  calculateFirstAppWindowSize,
  constrainWindowSize,
  fitWindowSize,
} from "./window-layout";
import {
  type MeasuredWindowLayout,
  type WindowLayoutGuard,
  finalizeAppWindowLayout,
  shouldFinishWindowLayoutWait,
  measureWindowLayout,
} from "./window-layout-lifecycle";

interface AppProviderProps {
  children: ReactNode;
}

type TauriWindowMode = "setup" | "app";
type WindowModule = typeof import("@tauri-apps/api/window");
type TauriWindow = Awaited<ReturnType<WindowModule["getCurrentWindow"]>>;
type TauriMonitor = NonNullable<
  Awaited<ReturnType<WindowModule["currentMonitor"]>>
>;

// Keep in step with MOBILE_BREAKPOINT in hooks/use-mobile.ts.
const MIN_DESKTOP_LAYOUT_WIDTH = 768;

// Logical px per CSS px: webview zoom above the display scale (Windows text
// scaling); 1 if none.
function logicalPerCssPx(monitorScale: number): number {
  if (typeof window === "undefined" || !(monitorScale > 0)) return 1;
  const ratio = window.devicePixelRatio / monitorScale;
  return Number.isFinite(ratio) && ratio > 1 ? ratio : 1;
}

// Autostart passes --hidden: layout still applies, but the window stays in the tray.
let launchedHidden: Promise<boolean> | null = null;
function wasLaunchedHidden(): Promise<boolean> {
  launchedHidden ??= import("@tauri-apps/api/core")
    .then(({ invoke }) => invoke<boolean>("was_launched_hidden"))
    .catch(() => false);
  return launchedHidden;
}

type WindowLayoutObserver = {
  waitForSettled: () => Promise<void>;
  dispose: () => void;
};

const WINDOW_LAYOUT_POLL_MS = 50;
const WINDOW_LAYOUT_TIMEOUT_MS = 1000;

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

/** Watches native geometry changes that may arrive after a restore call resolves. */
async function observeWindowLayout(
  win: TauriWindow,
  isCurrent: WindowLayoutGuard,
): Promise<WindowLayoutObserver | null> {
  let revision = 0;
  const noteChange = () => {
    revision += 1;
  };
  const unlistenResized = await win.onResized(noteChange);
  let unlistenMoved: (() => void) | undefined;
  try {
    unlistenMoved = await win.onMoved(noteChange);
  } catch (error) {
    unlistenResized();
    throw error;
  }
  if (!isCurrent()) {
    unlistenResized();
    unlistenMoved();
    return null;
  }

  return {
    waitForSettled: async () => {
      let observedRevision = revision;
      let sawPostShowChange = false;
      for (
        let elapsed = 0;
        elapsed < WINDOW_LAYOUT_TIMEOUT_MS && isCurrent();
        elapsed += WINDOW_LAYOUT_POLL_MS
      ) {
        await delay(WINDOW_LAYOUT_POLL_MS);
        if (revision !== observedRevision) {
          observedRevision = revision;
          sawPostShowChange = true;
          continue;
        }
        if (shouldFinishWindowLayoutWait(sawPostShowChange)) {
          return;
        }
      }
    },
    dispose: () => {
      unlistenResized();
      unlistenMoved();
    },
  };
}

function measureTauriWindowLayout(
  windowModule: WindowModule,
  win: TauriWindow,
  isCurrent: WindowLayoutGuard,
): Promise<MeasuredWindowLayout<TauriMonitor> | null> {
  return measureWindowLayout(
    {
      currentMonitor: () => windowModule.currentMonitor(),
      primaryMonitor: () => windowModule.primaryMonitor(),
      innerSize: () => win.innerSize(),
      outerSize: () => win.outerSize(),
    },
    isCurrent,
  );
}

/** Sizes the window and centers it inside the work area. */
async function placeWindow(
  win: TauriWindow,
  { LogicalSize, PhysicalPosition }: WindowModule,
  { monitor, frameSize }: MeasuredWindowLayout<TauriMonitor>,
  size: LogicalWindowSize,
  isCurrent: WindowLayoutGuard,
): Promise<void> {
  await win.setSize(new LogicalSize(size.width, size.height));
  if (!isCurrent()) return;
  if (!monitor) {
    await win.center();
    return;
  }
  // Center from the requested size because GTK may still report stale geometry.
  const scaleFactor = await win.scaleFactor();
  if (!isCurrent()) return;
  const position = calculateCenteredPosition(monitor.workArea, {
    width: Math.round(size.width * scaleFactor) + frameSize.width,
    height: Math.round(size.height * scaleFactor) + frameSize.height,
  });
  await win.setPosition(new PhysicalPosition(position.x, position.y));
}

async function showSetupWindow(isCurrent: WindowLayoutGuard): Promise<void> {
  const windowModule = await import("@tauri-apps/api/window");
  const { invoke } = await import("@tauri-apps/api/core");
  if (!isCurrent()) return;

  const win = windowModule.getCurrentWindow();
  await invoke("reset_app_window_layout_initialized");
  if (!isCurrent()) return;
  // Clear app-mode constraints before showing the smaller setup window.
  await win.setSizeConstraints(null);
  if (!isCurrent()) return;
  await win.setResizable(false);
  if (!isCurrent()) return;
  const measured = await measureTauriWindowLayout(windowModule, win, isCurrent);
  if (!measured) return;
  // Keep the non-resizable setup window fully visible.
  const setupSize = fitWindowSize(
    PREFERRED_SETUP_WINDOW_SIZE,
    measured.bounds.maximum,
  );
  await placeWindow(win, windowModule, measured, setupSize, isCurrent);
  if (!isCurrent()) return;
  if (await wasLaunchedHidden()) return;
  if (!isCurrent()) return;
  await win.show();
}

async function enforceWindowSizeBounds(
  win: TauriWindow,
  LogicalSize: WindowModule["LogicalSize"],
  isCurrent: WindowLayoutGuard,
  bounds: WindowSizeBounds,
  requestedSize: LogicalWindowSize = bounds.minimum,
): Promise<void> {
  const [innerSize, scaleFactor, isMaximized, isFullscreen] = await Promise.all(
    [win.innerSize(), win.scaleFactor(), win.isMaximized(), win.isFullscreen()],
  );
  if (!isCurrent()) return;
  // Let the window manager size maximized and fullscreen windows.
  if (isMaximized || isFullscreen) return;

  const currentSize = {
    width: Math.round(innerSize.width / scaleFactor),
    height: Math.round(innerSize.height / scaleFactor),
  };
  // Linux can briefly report the pre-resize size from its GTK cache.
  const nextSize = constrainWindowSize(currentSize, requestedSize, bounds);
  if (
    nextSize.width !== currentSize.width ||
    nextSize.height !== currentSize.height
  ) {
    await win.setSize(new LogicalSize(nextSize.width, nextSize.height));
  }
}

async function applyAppWindowLayout(
  isCurrent: WindowLayoutGuard,
): Promise<void> {
  const windowModule = await import("@tauri-apps/api/window");
  const { invoke } = await import("@tauri-apps/api/core");
  const { restoreStateCurrent, StateFlags } =
    await import("@tauri-apps/plugin-window-state");
  if (!isCurrent()) return;

  const win = windowModule.getCurrentWindow();
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
  const measured = await measureTauriWindowLayout(windowModule, win, isCurrent);
  if (!measured) return;

  let requestedSize: LogicalWindowSize | undefined;
  let restored = false;
  let layoutObserver: WindowLayoutObserver | null = null;
  try {
    if (hasInitializedAppLayout && hasSavedState) {
      restored = true;
      // Subscribe before restoring so native events cannot race listener setup.
      layoutObserver = await observeWindowLayout(win, isCurrent);
      if (!layoutObserver) return;
      // Subsequent launch: plugin restores size/position/maximized, with built-in
      // off-screen protection for positions saved on a now-disconnected display.
      await restoreStateCurrent(
        StateFlags.SIZE | StateFlags.POSITION | StateFlags.MAXIMIZED,
      );
      if (!isCurrent()) return;
    } else {
      // First launch: fit to the current work area and center.
      const cssSafeLogicalWidth = measured.monitor
        ? Math.round(
            MIN_DESKTOP_LAYOUT_WIDTH *
              logicalPerCssPx(measured.monitor.scaleFactor),
          )
        : undefined;
      requestedSize = calculateFirstAppWindowSize(
        measured.bounds,
        cssSafeLogicalWidth,
      );
      await placeWindow(win, windowModule, measured, requestedSize, isCurrent);
    }
    // Apply work-area constraints after restore to preserve the saved size.
    await finalizeAppWindowLayout({
      restored,
      measured,
      show: async () => {
        if (await wasLaunchedHidden()) return false;
        if (!isCurrent()) return false;
        await win.show();
        return true;
      },
      waitForSettled: layoutObserver?.waitForSettled,
      measure: () => measureTauriWindowLayout(windowModule, win, isCurrent),
      setMinimumConstraints: (minimum) =>
        win.setSizeConstraints({
          minWidth: minimum.width,
          minHeight: minimum.height,
        }),
      enforceBounds: (bounds) =>
        enforceWindowSizeBounds(
          win,
          windowModule.LogicalSize,
          isCurrent,
          bounds,
          requestedSize,
        ),
      isCurrent,
    });

    if (!isCurrent()) return;
    await invoke("mark_app_window_layout_initialized");
  } finally {
    layoutObserver?.dispose();
  }
}

async function showWindowFallback(): Promise<void> {
  const { getCurrentWindow } = await import("@tauri-apps/api/window");
  const win = getCurrentWindow();
  await win.setResizable(true);
  if (await wasLaunchedHidden()) return;
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
  appContent,
}: {
  isExternalServer: boolean;
  children?: ReactNode;
  appContent: ReactNode;
}) {
  const update = useTauriUpdate(isExternalServer);
  const isUpdating =
    update.status === "updating-backend" ||
    update.status === "downloading" ||
    update.status === "installing" ||
    (update.status === "error" && !update.dismissed);

  const content = isUpdating ? (
    <UpdateScreen
      status={update.status}
      logs={update.logs}
      progress={update.progress}
      error={update.error}
      onRetry={update.retryUpdate}
      onSkipRestart={update.skipAndRestart}
      onCopyDiagnostics={update.copyDiagnostics}
    />
  ) : (
    // Capped like the browser stack: the download panel shares it, so both must fit.
    <div className="pointer-events-none fixed bottom-4 right-4 z-[9998] flex max-h-[calc(100dvh_-_2rem)] flex-col items-end gap-2">
      <UpdateBanner
        status={update.status}
        info={update.info}
        dismissed={update.dismissed}
        lastFailure={update.lastFailure}
        isExternalServer={isExternalServer}
        updatePolicyMode={update.updatePolicyMode}
        manualReleaseUrl={update.manualReleaseUrl}
        releasePageUrl={update.releasePageUrl}
        positioned={false}
        onInstall={update.installUpdate}
        onDismiss={update.dismiss}
        onCopyDiagnostics={update.copyDiagnostics}
      />
      {children}
    </div>
  );

  return (
    <TauriUpdateContext.Provider value={update}>
      {content}
      {appContent}
    </TauriUpdateContext.Provider>
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
  "--studio-desktop-titlebar-height": "34px",
  "--studio-titlebar-navigation-offset-y": "4px",
  "--studio-mac-traffic-light-inset": "78px",
  "--studio-collapsed-chat-controls-inset": "188px",
  "--studio-startup-top-inset": "58px",
  "--studio-content-top-inset": "0px",
  "--studio-non-chat-content-top-inset": "34px",
  "--studio-hidden-route-top-inset": "34px",
  "--studio-chat-header-height": "44px",
  "--studio-chat-header-padding-top": "9px",
  "--studio-media-header-left-inset": "0.5rem",
  "--studio-chat-control-height": "33px",
  "--studio-chat-header-right-inset": "0px",
} as CSSProperties;

const CUSTOM_CHROME_STYLE = {
  "--studio-titlebar-height": "0px",
  "--studio-custom-titlebar-height": "34px",
  "--studio-desktop-titlebar-height": "34px",
  "--studio-sidebar-expanded-width": "17.5rem",
  "--studio-sidebar-collapsed-width": "3rem",
  "--studio-collapsed-chat-controls-inset": "12px",
  "--studio-startup-top-inset": "42px",
  "--studio-content-top-inset": "34px",
  "--studio-hidden-route-top-inset": "34px",
  "--studio-chat-header-height": "48px",
  "--studio-chat-header-padding-top": "9px",
  "--studio-media-header-left-inset": "0.5rem",
  "--studio-chat-control-height": "33px",
  "--studio-chat-header-right-inset": "0px",
  "--studio-window-control-inset": "112px",
} as CSSProperties;

// Mirror the titlebar heights onto <html>: the styles above sit on a wrapper
// div, so overlays portalled into document.body would read them as empty.
function DesktopChromeVarsEffect({
  usesCustomTitlebar,
  usesNativeMacTitlebar,
}: {
  usesCustomTitlebar: boolean;
  usesNativeMacTitlebar: boolean;
}) {
  useEffect(() => {
    const el = document.documentElement;
    const set = (name: string, value: string | null) =>
      value === null
        ? el.style.removeProperty(name)
        : el.style.setProperty(name, value);
    set("--studio-custom-titlebar-height", usesCustomTitlebar ? "34px" : null);
    set("--studio-mac-titlebar-height", usesNativeMacTitlebar ? "34px" : null);
    set("--studio-window-control-inset", usesCustomTitlebar ? "112px" : null);
    // How far body-portaled surfaces (dialogs, alert dialogs) must stay clear of the
    // top: either titlebar paints over them, and neither inherits the wrapper's style.
    set(
      "--studio-window-chrome-top",
      usesCustomTitlebar || usesNativeMacTitlebar ? "34px" : null,
    );
    return () => {
      set("--studio-custom-titlebar-height", null);
      set("--studio-mac-titlebar-height", null);
      set("--studio-window-control-inset", null);
      set("--studio-window-chrome-top", null);
    };
  }, [usesCustomTitlebar, usesNativeMacTitlebar]);
  return null;
}

function TauriWrapper({ children }: { children: ReactNode }) {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const {
    status,
    logs,
    error,
    isExternalServer,
    currentStepIndex,
    progressDetail,
    startupMessage,
    elevationPackages,
    closing,
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
  const [nativeMacControlsHidden, setNativeMacControlsHidden] = useState(false);

  const [windowRevealRevision, setWindowRevealRevision] = useState(0);
  const usesCustomTitlebar = shouldUseCustomWindowTitlebar();
  const usesNativeMacTitlebar = shouldUseNativeMacWindowTitlebar();
  const hidesTitlebarSidebar = HIDDEN_TITLEBAR_SIDEBAR_ROUTES.has(pathname);

  useEffect(() => {
    if (!isTauri) return;
    return () => {
      windowLayoutGenerationRef.current += 1;
      appliedWindowModeRef.current = null;
    };
  }, []);


  useEffect(() => {
    if (!isTauri) return;
    let disposed = false;
    let stopListening: (() => void) | undefined;

    void wasLaunchedHidden().then(async (hiddenAtLaunch) => {
      if (!hiddenAtLaunch || disposed) return;
      const { getCurrentWindow } = await import("@tauri-apps/api/window");
      const unlisten = await getCurrentWindow().onFocusChanged(({ payload }) => {
        if (!payload || disposed) return;
        stopListening?.();
        stopListening = undefined;
        // Native tray reveal focuses the window. Re-run the deferred layout now
        // that currentMonitor() can resolve the restored display.
        launchedHidden = Promise.resolve(false);
        appliedWindowModeRef.current = null;
        setWindowRevealRevision((revision) => revision + 1);
      });
      if (disposed) unlisten();
      else stopListening = unlisten;
    });

    return () => {
      disposed = true;
      stopListening?.();
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
  }, [status, windowRevealRevision]);

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

  useEffect(() => {
    if (!usesNativeMacTitlebar) return;
    let active = true;
    const refresh = () => {
      void import("@tauri-apps/api/window")
        .then(({ getCurrentWindow }) => getCurrentWindow().isFullscreen())
        .then((fullscreen) => {
          if (active) setNativeMacControlsHidden(fullscreen);
        })
        .catch(() => undefined);
    };
    const handleResize = () => {
      refresh();
    };
    refresh();
    window.addEventListener("resize", handleResize);

    return () => {
      active = false;
      window.removeEventListener("resize", handleResize);
    };
  }, [usesNativeMacTitlebar]);

  if (!isTauri) {
    return (
      <>
        {children}
        {/* One bottom-right stack so overlays never overlap: download panel at the
            corner, banners above, each owning its width. */}
        {/* Capped to the viewport, or a long download list plus expanded notes
            pushes the top of the stack off screen. */}
        <div className="pointer-events-none fixed bottom-4 right-4 z-[9998] flex max-h-[calc(100dvh_-_2rem)] flex-col items-end gap-2">
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

  const showApp = status === "running" && desktopAuthReady;
  const startupStatus = status === "running" ? "starting" : status;
  const startupProgressDetail = progressDetail;

  const shell = showApp ? (
    <TauriUpdateLayer
      isExternalServer={isExternalServer}
      appContent={
        <>
          <NativeIntentDrain />
          {children}
        </>
      }
    >
      <LlamaUpdateBanner positioned={false} enabled={!hidesTitlebarSidebar} />
      <DownloadManagerPanel positioned={false} />
    </TauriUpdateLayer>
  ) : (
    <StartupScreen
      status={startupStatus}
      logs={logs}
      error={error}
      currentStepIndex={currentStepIndex}
      progressDetail={startupProgressDetail}
      startupMessage={startupMessage}
      elevationPackages={elevationPackages}
      onInstall={startInstall}
      onRetry={retry}
      onRetryInstall={retryInstall}
      onApproveElevation={approveElevation}
      onStartServer={retry}
      onCopyDiagnostics={copyDiagnostics}
    />
  );

  // Over the shell, not instead of it: ClosingScreen covers the app and the update layer
  // alike, and a declined quit puts the user back where they were rather than remounting
  // the tree under them.
  const content = (
    <>
      {shell}
      {closing && <ClosingScreen />}
    </>
  );

  const chromeVars = (
    <DesktopChromeVarsEffect
      usesCustomTitlebar={usesCustomTitlebar}
      usesNativeMacTitlebar={usesNativeMacTitlebar}
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
          style={
            nativeMacControlsHidden
              ? ({
                  ...MAC_NATIVE_CHROME_STYLE,
                  "--studio-mac-traffic-light-inset": "6px",
                } as CSSProperties)
              : MAC_NATIVE_CHROME_STYLE
          }
        >
          {!showApp || hidesTitlebarSidebar ? (
            <div
              data-tauri-drag-region={true}
              aria-hidden="true"
              className="pointer-events-auto fixed inset-x-0 top-0 z-50 h-[var(--studio-mac-titlebar-height,34px)] select-none"
            />
          ) : null}
          {chromeVars}
          {content}
        </div>
      );
    }

    return (
      <>
        {chromeVars}
        {content}
      </>
    );
  }

  const showSidebarSurface = showApp && !hidesTitlebarSidebar;

  return (
    <div
      className="relative h-dvh min-h-0 overflow-hidden bg-background"
      style={CUSTOM_CHROME_STYLE}
    >
      {chromeVars}
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
  const { theme, resolved } = useTheme();
  const customization = useAppearanceCustomStore((s) => s.customization);
  useEffect(() => {
    applyCustomizationToDocument(customization, resolved);
  }, [customization, resolved]);
  useEffect(() => {
    if (!isTauri) return;
    void import("@tauri-apps/api/window")
      .then(({ getCurrentWindow }) =>
        getCurrentWindow().setTheme(theme === "system" ? null : theme),
      )
      .catch(() => undefined);
  }, [theme]);
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
        <DeepLinkHandler />
        <TauriWrapper>{children}</TauriWrapper>
        <SttDownloadPrompt />
        <Toaster
          position="top-right"
          visibleToasts={2}
          expand={true}
          closeButton={true}
          // Clear the chat header buttons on the right. On desktop, also drop
          // below the ~34px custom window titlebar so toasts don't cover the
          // minimize / maximize / close controls.
          offset={{ top: isTauri ? 46 : 12, right: 64 }}
        />
      </TooltipProvider>
    </MotionConfig>
  );
}
