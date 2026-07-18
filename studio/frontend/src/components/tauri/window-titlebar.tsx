// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useSidebarPin } from "@/hooks/use-sidebar-pin";
import { isTauri } from "@/lib/api-base";
import { cn } from "@/lib/utils";
import {
  Cancel01Icon,
  LayoutAlignLeftIcon,
  MinusSignIcon,
  SquareIcon,
  SquareSquareIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
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
        "relative z-[80] inline-flex size-8 items-center justify-center rounded-[10px] text-muted-foreground/90 transition-colors hover:bg-nav-surface-hover hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
        className,
      )}
    >
      {children}
    </button>
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
  const sidebarWidth = showSidebarSurface
    ? pinned
      ? "var(--studio-sidebar-expanded-width,17.5rem)"
      : "var(--studio-sidebar-collapsed-width,3rem)"
    : "0px";
  const contentBorderLeft = `calc(${sidebarWidth} + 12px)`;

  const refreshMaximized = useCallback(async () => {
    if (!enabled) {
      return;
    }
    try {
      const appWindow = await getAppWindow();
      setMaximized(await appWindow.isMaximized());
    } catch {
      // Window permission not ready yet: keep previous visual state.
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
        className={cn(
          "pointer-events-none absolute inset-x-0 top-0 z-[70] h-[var(--studio-custom-titlebar-height)] select-none text-foreground",
          showSidebarSurface && "bg-sidebar text-sidebar-foreground",
        )}
        aria-label="Window titlebar"
      >
        {showSidebarSurface && (
          <div
            aria-hidden="true"
            className="pointer-events-none absolute top-full h-3 w-px -translate-x-px bg-sidebar"
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
        {showSidebarSurface && (
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
              pinned ? "gap-2 px-3" : "justify-center",
            )}
            style={{ width: sidebarWidth }}
            onMouseDown={handleDragMouseDown}
            onDoubleClick={handleDragDoubleClick}
          >
            {pinned ? (
              <>
                <div className="flex min-w-0 flex-1 items-center gap-2">
                  <img
                    src="/rounded-512.png"
                    alt=""
                    aria-hidden="true"
                    draggable={false}
                    className="size-5 shrink-0 rounded-[6px] object-cover"
                  />
                  <span className="min-w-0 truncate text-[13px] font-semibold leading-none tracking-[0.01em] text-nav-fg">
                    Unsloth Studio
                  </span>
                </div>
                <button
                  type="button"
                  title="Collapse sidebar"
                  aria-label="Collapse sidebar"
                  onMouseDown={(event) => event.stopPropagation()}
                  onDoubleClick={(event) => event.stopPropagation()}
                  onClick={(event) => {
                    event.stopPropagation();
                    togglePinned();
                  }}
                  className="inline-flex size-8 shrink-0 items-center justify-center rounded-[10px] text-nav-icon-idle transition-colors hover:bg-nav-surface-hover hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                >
                  <HugeiconsIcon
                    icon={LayoutAlignLeftIcon}
                    strokeWidth={1.75}
                    className="size-icon"
                  />
                </button>
              </>
            ) : (
              <button
                type="button"
                title="Expand sidebar"
                aria-label="Expand sidebar"
                onMouseDown={(event) => event.stopPropagation()}
                onDoubleClick={(event) => event.stopPropagation()}
                onClick={(event) => {
                  event.stopPropagation();
                  togglePinned();
                }}
                className="inline-flex size-8 items-center justify-center rounded-[10px] transition-colors hover:bg-nav-surface-hover focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
              >
                <img
                  src="/rounded-512.png"
                  alt=""
                  aria-hidden="true"
                  draggable={false}
                  className="size-5 rounded-[6px] object-cover"
                />
                <span className="sr-only">Expand sidebar</span>
              </button>
            )}
          </div>
        )}
        <div
          className="pointer-events-auto absolute top-0 h-full"
          style={{
            left: sidebarWidth,
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
            <HugeiconsIcon
              icon={MinusSignIcon}
              strokeWidth={1.75}
              className="size-[15px]"
            />
          </WindowControlButton>
          <WindowControlButton
            label={maximized ? "Restore window" : "Maximize window"}
            onClick={() =>
              runWindowAction((appWindow) => appWindow.toggleMaximize())
            }
          >
            <HugeiconsIcon
              icon={maximized ? SquareSquareIcon : SquareIcon}
              strokeWidth={1.75}
              className="size-[14px]"
            />
          </WindowControlButton>
          <WindowControlButton
            label="Close window"
            onClick={() => runWindowAction((appWindow) => appWindow.close())}
            className="hover:bg-destructive/10 hover:text-destructive focus-visible:ring-destructive/70 dark:hover:bg-destructive/20"
          >
            <HugeiconsIcon
              icon={Cancel01Icon}
              strokeWidth={1.75}
              className="size-[15px]"
            />
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
