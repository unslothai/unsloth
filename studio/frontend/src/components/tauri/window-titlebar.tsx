// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  clearAppClosing,
  markAppClosing,
} from "@/components/tauri/closing-signal";
import { useIsMobile } from "@/hooks/use-mobile";
import { useSidebarPin } from "@/hooks/use-sidebar-pin";
import { useSidebarWidth } from "@/hooks/use-sidebar-width";
import { isTauri } from "@/lib/api-base";
import { cn } from "@/lib/utils";
import { CopyIcon, LayoutAlignLeftIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { Window as TauriWindow } from "@tauri-apps/api/window";
import { ArrowLeft, ArrowRight, Minus, Square, X } from "lucide-react";
import {
  type MouseEvent,
  type PointerEvent,
  type ReactElement,
  type ReactNode,
  useCallback,
  useEffect,
  useRef,
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

export function getClientPlatform(): string {
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

export function shouldUseNativeMacWindowTitlebar(): boolean {
  if (!isTauri) {
    return false;
  }
  return getClientPlatform().includes("mac");
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
        "relative z-[80] inline-flex size-[30px] items-center justify-center rounded-[10px] text-muted-foreground/90 transition-colors hover:bg-nav-surface-hover hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
        className,
      )}
    >
      {children}
    </button>
  );
}

export function DesktopTitlebarNavigation({
  expanded,
  onToggleSidebar,
  className,
  showSidebarToggle = true,
}: {
  expanded: boolean;
  onToggleSidebar: () => void;
  className?: string;
  /** Off in mobile, where Navbar's SidebarTrigger owns the slot; a spacer holds it open. */
  showSidebarToggle?: boolean;
}): ReactElement {
  const stopTitlebarDrag = (event: MouseEvent<HTMLButtonElement>) => {
    event.stopPropagation();
  };
  const buttonClass =
    "inline-flex size-[30px] shrink-0 items-center justify-center rounded-[10px] text-nav-icon-idle dark:text-nav-fg-muted transition-colors hover:bg-nav-surface-hover hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring";

  return (
    <div
      className={cn(
        "flex mt-1 translate-y-[var(--studio-titlebar-navigation-offset-y,0px)] items-center gap-0.5",
        className,
      )}
      role="toolbar"
      aria-label="Sidebar and page navigation"
    >
      {showSidebarToggle ? (
        <button
          type="button"
          title={expanded ? "Collapse sidebar" : "Expand sidebar"}
          aria-label={expanded ? "Collapse sidebar" : "Expand sidebar"}
          onMouseDown={stopTitlebarDrag}
          onDoubleClick={stopTitlebarDrag}
          onClick={(event) => {
            event.stopPropagation();
            onToggleSidebar();
          }}
          className={buttonClass}
        >
          <HugeiconsIcon
            icon={LayoutAlignLeftIcon}
            strokeWidth={1.75}
            className="size-icon !size-[calc(var(--icon-size)+1px)]"
          />
        </button>
      ) : (
        <div aria-hidden="true" className="size-[30px] shrink-0" />
      )}
      <button
        type="button"
        title="Go back"
        aria-label="Go back"
        onMouseDown={stopTitlebarDrag}
        onDoubleClick={stopTitlebarDrag}
        onClick={(event) => {
          event.stopPropagation();
          window.history.back();
        }}
        className={buttonClass}
      >
        <ArrowLeft
          aria-hidden="true"
          strokeWidth={1.75}
          className="size-icon !size-[calc(var(--icon-size)+1px)]"
        />
      </button>
      <button
        type="button"
        title="Go forward"
        aria-label="Go forward"
        onMouseDown={stopTitlebarDrag}
        onDoubleClick={stopTitlebarDrag}
        onClick={(event) => {
          event.stopPropagation();
          window.history.forward();
        }}
        className={buttonClass}
      >
        <ArrowRight
          aria-hidden="true"
          strokeWidth={1.75}
          className="size-icon !size-[calc(var(--icon-size)+1px)]"
        />
      </button>
    </div>
  );
}

export function WindowTitlebar({
  showSidebarSurface = false,
}: {
  showSidebarSurface?: boolean;
}): ReactElement | null {
  const [enabled] = useState(shouldUseCustomWindowTitlebar);
  const [maximized, setMaximized] = useState(false);
  const { pinned, togglePinned } = useSidebarPin();
  // Outside SidebarProvider, so read the same media query the provider does.
  const isMobile = useIsMobile();

  const maximizeRefreshSequence = useRef(0);
  const maximizeRefreshTimer = useRef<number | null>(null);
  // The titlebar sits outside the sidebar wrapper, so it cannot inherit
  // --sidebar-width. Read the resized width from the same store instead.
  const { width } = useSidebarWidth();
  const sidebarWidth = showSidebarSurface
    ? pinned
      ? // The live value only exists mid-drag; otherwise the committed width.
        `var(--studio-sidebar-live-width, ${width}px)`
      : "var(--studio-sidebar-collapsed-width,3rem)"
    : "0px";

  const titlebarNavigationWidth =
    showSidebarSurface && !pinned ? "7rem" : sidebarWidth;
  const contentBorderLeft = pinned ? `calc(${sidebarWidth} + 12px)` : "0px";

  const refreshMaximized = useCallback(async () => {
    if (!enabled) {
      return;
    }
    const refreshSequence = ++maximizeRefreshSequence.current;
    try {
      const appWindow = await getAppWindow();
      const nextMaximized = await appWindow.isMaximized();
      if (refreshSequence === maximizeRefreshSequence.current) {
        setMaximized(nextMaximized);
      }
    } catch {
      // Window permission not ready yet: keep previous visual state.
    }
  }, [enabled]);

  const scheduleMaximizedRefresh = useCallback(() => {
    if (maximizeRefreshTimer.current !== null) {
      window.clearTimeout(maximizeRefreshTimer.current);
    }
    maximizeRefreshTimer.current = window.setTimeout(() => {
      maximizeRefreshTimer.current = null;
      refreshMaximized().catch(() => undefined);
    }, 80);
  }, [refreshMaximized]);

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
          scheduleMaximizedRefresh();
        });
        unlistenFocus = await appWindow.onFocusChanged(() => {
          scheduleMaximizedRefresh();
        });
      } catch {
        // Missing capabilities should not break the rest of the app shell.
      }
    };

    setupWindowListeners().catch(() => undefined);

    return () => {
      mounted = false;

      maximizeRefreshSequence.current += 1;
      if (maximizeRefreshTimer.current !== null) {
        window.clearTimeout(maximizeRefreshTimer.current);
        maximizeRefreshTimer.current = null;
      }
      unlistenResize?.();
      unlistenFocus?.();
    };
  }, [enabled, refreshMaximized, scheduleMaximizedRefresh]);

  const runWindowAction = useCallback(
    (action: (appWindow: TauriWindow) => Promise<void>) => {
      const runAction = async () => {
        try {
          const appWindow = await getAppWindow();
          await action(appWindow);
          scheduleMaximizedRefresh();
        } catch {
          // Keep custom chrome inert rather than throwing into React on denied commands.
        }
      };

      runAction().catch(() => undefined);
    },
    [scheduleMaximizedRefresh],
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

  // pointerdown, not mousedown: Radix dismisses modals on pointerdown, which fires first,
  // so a mousedown handler starts the resize but the dialog closes underneath it.
  const handleResizePointerDown = useCallback(
    (direction: WindowResizeDirection) =>
      (event: PointerEvent<HTMLDivElement>) => {
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
        className={cn(
          "pointer-events-none absolute inset-x-0 top-0 z-[70] h-[var(--studio-custom-titlebar-height)] select-none text-foreground",
          showSidebarSurface && "bg-sidebar text-sidebar-foreground",
        )}
        aria-label="Window titlebar"
      >
        {showSidebarSurface && pinned && (
          <div
            aria-hidden="true"
            className="pointer-events-none absolute top-full size-3 -translate-x-px bg-sidebar"
            style={{ left: sidebarWidth }}
          />
        )}
        {showSidebarSurface && (
          <div
            aria-hidden="true"
            className="pointer-events-none absolute top-full h-px bg-sidebar-border"
            style={{ left: contentBorderLeft, right: 0 }}
          />
        )}
        {showSidebarSurface && pinned && (
          <div
            aria-hidden="true"
            className="pointer-events-none absolute top-full size-3 -translate-x-px rounded-tl-[12px] border-l border-t border-sidebar-border bg-background"
            style={{ left: sidebarWidth }}
          />
        )}
        {showSidebarSurface && (
          <div
            className={cn(
              "pointer-events-auto absolute left-0 top-0 flex h-full min-w-0 items-center",
              "pl-3",
            )}
            style={{ width: titlebarNavigationWidth }}
            onMouseDown={handleDragMouseDown}
            onDoubleClick={handleDragDoubleClick}
          >
            <DesktopTitlebarNavigation
              expanded={pinned}
              onToggleSidebar={togglePinned}
              showSidebarToggle={!isMobile}
            />
          </div>
        )}
        <div
          className="pointer-events-auto absolute top-0 h-full"
          style={{
            left: titlebarNavigationWidth,
            right: "calc(var(--studio-window-control-inset,112px) + 0.5rem)",
          }}
          onMouseDown={handleDragMouseDown}
          onDoubleClick={handleDragDoubleClick}
          aria-hidden="true"
        />
        <div
          className="pointer-events-auto absolute right-1 top-0 flex h-full items-center gap-0.5 px-1"
          role="toolbar"
          aria-label="Window controls"
        >
          <WindowControlButton
            label="Minimize window"
            onClick={() => runWindowAction((appWindow) => appWindow.minimize())}
          >
            <Minus aria-hidden="true" strokeWidth={1.75} className="w-[18px]" />
          </WindowControlButton>
          <WindowControlButton
            label={maximized ? "Restore window" : "Maximize window"}
            onClick={() =>
              runWindowAction((appWindow) => appWindow.toggleMaximize())
            }
          >
            {maximized ? (
              <HugeiconsIcon
                icon={CopyIcon}
                strokeWidth={1.75}
                className="size-[17px] rotate-180"
              />
            ) : (
              <Square
                aria-hidden="true"
                strokeWidth={1.75}
                className="size-[16px]"
              />
            )}
          </WindowControlButton>
          <WindowControlButton
            label="Close window"
            onClick={() =>
              runWindowAction(async (appWindow) => {
                // This titlebar is Windows and Linux only, where close means quit, so
                // paint the overlay now instead of waiting for Rust's app-closing to
                // come back over IPC. Rust retracts it if a confirmation declines.
                markAppClosing();
                try {
                  await appWindow.close();
                } catch (error) {
                  // The quit never started, so nothing will take the overlay down.
                  clearAppClosing();
                  throw error;
                }
              })
            }
            className="hover:bg-destructive/10 hover:text-destructive focus-visible:ring-destructive/70 dark:hover:bg-destructive/20"
          >
            <X aria-hidden="true" strokeWidth={1.75} className="size-[18px]" />
          </WindowControlButton>
        </div>
      </header>
      <div
        aria-hidden="true"
        className="pointer-events-auto fixed inset-x-2 top-0 z-[70] h-1 cursor-n-resize"
        onPointerDown={handleResizePointerDown("North")}
      />
      <div
        aria-hidden="true"
        className="pointer-events-auto fixed inset-x-2 bottom-0 z-[70] h-1 cursor-s-resize"
        onPointerDown={handleResizePointerDown("South")}
      />
      <div
        aria-hidden="true"
        className="pointer-events-auto fixed inset-y-2 left-0 z-[70] w-1 cursor-w-resize"
        onPointerDown={handleResizePointerDown("West")}
      />
      <div
        aria-hidden="true"
        className="pointer-events-auto fixed inset-y-2 right-0 z-[70] w-1 cursor-e-resize"
        onPointerDown={handleResizePointerDown("East")}
      />
      <div
        aria-hidden="true"
        className="pointer-events-auto fixed left-0 top-0 z-[70] size-3 cursor-nw-resize"
        onPointerDown={handleResizePointerDown("NorthWest")}
      />
      <div
        aria-hidden="true"
        className="pointer-events-auto fixed right-0 top-0 z-[70] size-3 cursor-ne-resize"
        onPointerDown={handleResizePointerDown("NorthEast")}
      />
      <div
        aria-hidden="true"
        className="pointer-events-auto fixed bottom-0 left-0 z-[70] size-3 cursor-sw-resize"
        onPointerDown={handleResizePointerDown("SouthWest")}
      />
      <div
        aria-hidden="true"
        className="pointer-events-auto fixed bottom-0 right-0 z-[70] size-3 cursor-se-resize"
        onPointerDown={handleResizePointerDown("SouthEast")}
      />
    </>
  );
}
