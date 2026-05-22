// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useSidebarPin } from "@/hooks/use-sidebar-pin";
import { isTauri } from "@/lib/api-base";
import { cn } from "@/lib/utils";
import type { Window as TauriWindow } from "@tauri-apps/api/window";
import {
  type MouseEvent,
  type ReactElement,
  type ReactNode,
  useCallback,
  useEffect,
  useState,
} from "react";

const CUSTOM_TITLEBAR_PLATFORMS = ["win", "linux", "x11"] as const;

type WindowResizeDirection =
  | "East"
  | "North"
  | "NorthEast"
  | "NorthWest"
  | "South"
  | "SouthEast"
  | "SouthWest"
  | "West";

type NavigatorWithUserAgentData = Navigator & {
  userAgentData?: {
    platform?: string;
  };
};

function getClientPlatform(): string {
  if (typeof navigator === "undefined") {
    return "";
  }
  const nav = navigator as NavigatorWithUserAgentData;
  return (
    nav.userAgentData?.platform ??
    navigator.platform ??
    navigator.userAgent
  ).toLowerCase();
}

export function shouldUseCustomWindowTitlebar(): boolean {
  if (!isTauri) {
    return false;
  }
  const platform = getClientPlatform();
  if (!platform || platform.includes("mac")) {
    return false;
  }
  return CUSTOM_TITLEBAR_PLATFORMS.some((token) => platform.includes(token));
}

async function getAppWindow(): Promise<TauriWindow> {
  const { getCurrentWindow } = await import("@tauri-apps/api/window");
  return getCurrentWindow();
}

function WindowControlButton({
  label,
  className,
  onClick,
  children,
}: {
  label: string;
  className?: string;
  onClick: () => void;
  children: ReactNode;
}): ReactElement {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      onClick={onClick}
      className={cn(
        "relative z-[80] inline-flex size-8 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-muted/80 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
        className,
      )}
    >
      {children}
    </button>
  );
}

function MinimizeGlyph(): ReactElement {
  return (
    <span aria-hidden="true" className="h-px w-3.5 rounded-full bg-current" />
  );
}

function MaximizeGlyph(): ReactElement {
  return (
    <span
      aria-hidden="true"
      className="size-3 rounded-[2px] border border-current"
    />
  );
}

function RestoreGlyph(): ReactElement {
  return (
    <span aria-hidden="true" className="relative size-3.5">
      <span className="absolute left-0.5 top-0 size-2.5 rounded-[2px] border border-current" />
      <span className="absolute bottom-0 right-0 size-2.5 rounded-[2px] border border-current bg-muted" />
    </span>
  );
}

function CloseGlyph(): ReactElement {
  return (
    <span aria-hidden="true" className="relative size-3.5">
      <span className="absolute left-1/2 top-0 h-3.5 w-px -translate-x-1/2 rotate-45 rounded-full bg-current" />
      <span className="absolute left-1/2 top-0 h-3.5 w-px -translate-x-1/2 -rotate-45 rounded-full bg-current" />
    </span>
  );
}

export function WindowTitlebar({
  showSidebarSurface = false,
}: {
  showSidebarSurface?: boolean;
}): ReactElement | null {
  const [enabled] = useState(shouldUseCustomWindowTitlebar);
  const [maximized, setMaximized] = useState(false);
  const { pinned } = useSidebarPin();

  const refreshMaximized = useCallback(async () => {
    if (!enabled) {
      return;
    }
    try {
      const appWindow = await getAppWindow();
      setMaximized(await appWindow.isMaximized());
    } catch {
      // If a window permission is not ready yet, keep the previous visual state.
    }
  }, [enabled]);

  useEffect(() => {
    if (!enabled) {
      return;
    }
    let mounted = true;
    let unlistenResize: (() => void) | undefined;
    let unlistenFocus: (() => void) | undefined;

    const setupWindowListeners = async () => {
      try {
        const appWindow = await getAppWindow();
        if (!mounted) {
          return;
        }
        setMaximized(await appWindow.isMaximized());
        unlistenResize = await appWindow.onResized(() => {
          refreshMaximized().catch(() => undefined);
        });
        unlistenFocus = await appWindow.onFocusChanged(() => {
          refreshMaximized().catch(() => undefined);
        });
      } catch {
        // Missing capabilities should not break the rest of the app shell.
      }
    };

    setupWindowListeners().catch(() => undefined);

    return () => {
      mounted = false;
      unlistenResize?.();
      unlistenFocus?.();
    };
  }, [enabled, refreshMaximized]);

  const runWindowAction = useCallback(
    (action: (appWindow: TauriWindow) => Promise<void>) => {
      const runAction = async () => {
        try {
          const appWindow = await getAppWindow();
          await action(appWindow);
          await refreshMaximized();
        } catch {
          // Keep custom chrome inert rather than throwing into React on denied commands.
        }
      };

      runAction().catch(() => undefined);
    },
    [refreshMaximized],
  );

  const handleDragMouseDown = useCallback(
    (event: MouseEvent<HTMLDivElement>) => {
      if (event.button !== 0 || event.detail > 1) {
        return;
      }
      runWindowAction((appWindow) => appWindow.startDragging());
    },
    [runWindowAction],
  );

  const handleDragDoubleClick = useCallback(
    (event: MouseEvent<HTMLDivElement>) => {
      if (event.button !== 0) {
        return;
      }
      runWindowAction((appWindow) => appWindow.toggleMaximize());
    },
    [runWindowAction],
  );

  const handleResizeMouseDown = useCallback(
    (direction: WindowResizeDirection) =>
      (event: MouseEvent<HTMLDivElement>) => {
        if (event.button !== 0) {
          return;
        }
        event.preventDefault();
        event.stopPropagation();
        runWindowAction(async (appWindow) => {
          if (!(await appWindow.isResizable())) {
            return;
          }
          await appWindow.startResizeDragging(direction);
        });
      },
    [runWindowAction],
  );

  if (!enabled) {
    return null;
  }

  return (
    <>
      <header
        className="relative z-[60] flex h-[var(--studio-titlebar-height)] shrink-0 select-none items-center text-foreground"
        aria-label="Window titlebar"
      >
        {showSidebarSurface && (
          <div
            className="h-full shrink-0 border-r border-sidebar-border bg-sidebar"
            style={{ width: pinned ? "16rem" : "3rem" }}
            onMouseDown={handleDragMouseDown}
            onDoubleClick={handleDragDoubleClick}
            aria-hidden="true"
          />
        )}
        <div
          className="h-full min-w-0 flex-1 border-b border-border/35 bg-muted/35"
          onMouseDown={handleDragMouseDown}
          onDoubleClick={handleDragDoubleClick}
          aria-hidden="true"
        />
        <div
          className="flex h-full shrink-0 items-center gap-0.5 border-b border-border/35 bg-muted/35 px-1"
          role="toolbar"
          aria-label="Window controls"
        >
          <WindowControlButton
            label="Minimize window"
            onClick={() => runWindowAction((appWindow) => appWindow.minimize())}
          >
            <MinimizeGlyph />
          </WindowControlButton>
          <WindowControlButton
            label={maximized ? "Restore window" : "Maximize window"}
            onClick={() =>
              runWindowAction((appWindow) => appWindow.toggleMaximize())
            }
          >
            {maximized ? <RestoreGlyph /> : <MaximizeGlyph />}
          </WindowControlButton>
          <WindowControlButton
            label="Close window"
            onClick={() => runWindowAction((appWindow) => appWindow.close())}
            className="hover:bg-destructive hover:text-destructive-foreground focus-visible:ring-destructive/70"
          >
            <CloseGlyph />
          </WindowControlButton>
        </div>
      </header>
      <div
        aria-hidden="true"
        className="fixed inset-x-2 top-0 z-[70] h-1 cursor-n-resize"
        onMouseDown={handleResizeMouseDown("North")}
      />
      <div
        aria-hidden="true"
        className="fixed inset-x-2 bottom-0 z-[70] h-1 cursor-s-resize"
        onMouseDown={handleResizeMouseDown("South")}
      />
      <div
        aria-hidden="true"
        className="fixed inset-y-2 left-0 z-[70] w-1 cursor-w-resize"
        onMouseDown={handleResizeMouseDown("West")}
      />
      <div
        aria-hidden="true"
        className="fixed inset-y-2 right-0 z-[70] w-1 cursor-e-resize"
        onMouseDown={handleResizeMouseDown("East")}
      />
      <div
        aria-hidden="true"
        className="fixed left-0 top-0 z-[70] size-3 cursor-nw-resize"
        onMouseDown={handleResizeMouseDown("NorthWest")}
      />
      <div
        aria-hidden="true"
        className="fixed right-0 top-0 z-[70] size-3 cursor-ne-resize"
        onMouseDown={handleResizeMouseDown("NorthEast")}
      />
      <div
        aria-hidden="true"
        className="fixed bottom-0 left-0 z-[70] size-3 cursor-sw-resize"
        onMouseDown={handleResizeMouseDown("SouthWest")}
      />
      <div
        aria-hidden="true"
        className="fixed bottom-0 right-0 z-[70] size-3 cursor-se-resize"
        onMouseDown={handleResizeMouseDown("SouthEast")}
      />
    </>
  );
}
