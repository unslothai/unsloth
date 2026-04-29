// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { StartupScreen } from "@/components/tauri/startup-screen";
import { UpdateBanner } from "@/components/tauri/update-banner";
import { UpdateScreen } from "@/components/tauri/update-screen";
import { Toaster } from "@/components/ui/sonner";
import { useTauriBackend } from "@/hooks/use-tauri-backend";
import { useTauriUpdate } from "@/hooks/use-tauri-update";
import { isTauri } from "@/lib/api-base";
import { ThemeProvider } from "next-themes";
import { useEffect, useRef, type ReactNode } from "react";

interface AppProviderProps {
  children: ReactNode;
}

// ---------------------------------------------------------------------------
// Tauri window helpers (only imported in Tauri mode)
// ---------------------------------------------------------------------------

async function showWindow(): Promise<void> {
  const { getCurrentWindow } = await import("@tauri-apps/api/window");
  await getCurrentWindow().show();
}

function easeOutQuart(t: number): number {
  return 1 - (1 - t) ** 4;
}

async function animateToGoldenRatio(abortRef: { current: boolean }): Promise<void> {
  const { getCurrentWindow, currentMonitor, LogicalSize } = await import("@tauri-apps/api/window");
  const win = getCurrentWindow();

  // Ensure window is visible before resizing
  await win.show();

  const monitor = await currentMonitor();
  if (!monitor) return;

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

  // Check reduced motion preference
  const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  if (prefersReducedMotion) {
    await win.setSize(new LogicalSize(finalW, finalH));
  } else {
    // Read current size instead of hardcoding — stays correct if tauri.conf.json changes
    const inner = await win.innerSize();
    const factor = await win.scaleFactor();
    const startW = Math.round(inner.width / factor);
    const startH = Math.round(inner.height / factor);
    const steps = 15;
    const stepDuration = 23; // ~350ms total

    for (let i = 1; i <= steps; i++) {
      if (abortRef.current) return;
      const t = easeOutQuart(i / steps);
      const w = Math.round(startW + (finalW - startW) * t);
      const h = Math.round(startH + (finalH - startH) * t);
      await win.setSize(new LogicalSize(w, h));
      await new Promise((r) => setTimeout(r, stepDuration));
    }
  }

  if (abortRef.current) return;

  // Apply constraints and finalize
  await win.setResizable(true);
  await win.setSizeConstraints({ minWidth: 900, minHeight: 600 });
  await win.center();
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
      />
    );
  }

  return (
    <UpdateBanner
      status={update.status}
      info={update.info}
      dismissed={update.dismissed}
      isExternalServer={isExternalServer}
      onInstall={update.installUpdate}
      onDismiss={update.dismiss}
    />
  );
}

function TauriWrapper({ children }: { children: ReactNode }) {
  const {
    status, logs, error, isExternalServer,
    currentStepIndex, progressDetail, elevationPackages,
    startInstall, retry, retryInstall, approveElevation,
  } = useTauriBackend();

  const hasResized = useRef(false);
  const abortRef = useRef(false);

  // Show the window once the frontend mounts (for pre-running states)
  useEffect(() => {
    if (isTauri) void showWindow();
  }, []);

  // Animate resize when backend becomes ready
  useEffect(() => {
    if (status === "running" && !hasResized.current) {
      hasResized.current = true;
      abortRef.current = false;
      animateToGoldenRatio(abortRef).catch(async () => {
        // On failure, at minimum make the window resizable so user can fix manually
        try {
          const { getCurrentWindow } = await import("@tauri-apps/api/window");
          await getCurrentWindow().setResizable(true);
        } catch { /* swallow — window may still be functional */ }
      });
    }
    return () => { abortRef.current = true; };
  }, [status]);

  if (!isTauri) return <>{children}</>;
  if (status === "running") return <><TauriUpdateLayer isExternalServer={isExternalServer} />{children}</>;

  return (
    <StartupScreen
      status={status}
      logs={logs}
      error={error}
      currentStepIndex={currentStepIndex}
      progressDetail={progressDetail}
      elevationPackages={elevationPackages}
      onInstall={startInstall}
      onRetry={retry}
      onRetryInstall={retryInstall}
      onApproveElevation={approveElevation}
      onStartServer={retry}
    />
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
