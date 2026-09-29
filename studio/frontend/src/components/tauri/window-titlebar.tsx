// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useIsMobileShell } from "@/hooks/use-mobile";
import { useSidebarPin } from "@/hooks/use-sidebar-pin";
import { useSidebarWidth } from "@/hooks/use-sidebar-width";
import { isTauri } from "@/lib/api-base";
import { cn } from "@/lib/utils";
import { Z_LAYER } from "@/lib/z-layers";
import { LayoutAlignLeftIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import type { Window as TauriWindow } from "@tauri-apps/api/window";
import { ArrowLeft, ArrowRight } from "lucide-react";
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

/**
 * The Windows caption symbols, on Windows and Linux alike: a 10-DIP glyph whose size and
 * stroke round to whole device pixels, so it stays crisp at fractional display scales.
 */
function CaptionGlyph({
  kind,
}: {
  kind: "minimize" | "maximize" | "restore" | "close";
}): ReactElement {
  const [scale, setScale] = useState(() => window.devicePixelRatio || 1);
  useEffect(() => {
    const update = () => setScale(window.devicePixelRatio || 1);
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, []);
  const pixels = Math.round(10 * scale);
  const stroke = (Math.max(1, Math.round(scale)) * 10) / pixels;
  const inset = stroke / 2;
  const edge = 10 - inset;
  return (
    <svg
      aria-hidden="true"
      width={pixels / scale}
      height={pixels / scale}
      viewBox="0 0 10 10"
      fill="none"
      stroke="currentColor"
      strokeWidth={stroke}
      className="shrink-0"
    >
      {kind === "minimize" && <path d="M0 5h10" />}
      {kind === "maximize" && (
        <rect
          x={inset}
          y={inset}
          width={10 - stroke}
          height={10 - stroke}
          rx="0.7"
        />
      )}
      {/* Windows puts the front window bottom-left. */}
      {kind === "restore" && (
        <>
          <path d={`M2.5 2.5V${inset}H${edge}V7.5H7.5`} />
          <rect
            x={inset}
            y="2.5"
            width={7.5 - inset}
            height={7.5 - inset}
            rx="0.7"
          />
        </>
      )}
      {kind === "close" && (
        <path
          d={`M${inset} ${inset}L${edge} ${edge}M${edge} ${inset}L${inset} ${edge}`}
        />
      )}
    </svg>
  );
}

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

/** Drags the window on press and maximizes on double click, like a native titlebar. */
export function WindowDragRegion({
  className,
}: {
  className?: string;
}): ReactElement {
  return (
    <div
      aria-hidden="true"
      className={className}
      onMouseDown={(event) => {
        if (event.button !== 0 || event.detail > 1) {
          return;
        }
        getAppWindow()
          .then((appWindow) => appWindow.startDragging())
          .catch(() => undefined);
      }}
      onDoubleClick={(event) => {
        if (event.button !== 0) {
          return;
        }
        getAppWindow()
          .then((appWindow) => appWindow.toggleMaximize())
          .catch(() => undefined);
      }}
    />
  );
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
        // The hit area runs up to the window edge, where a pointer thrown at the controls lands.
        "after:absolute after:inset-x-0 after:bottom-0 after:top-[-4px]",
        // Hovers like the navigation buttons it mirrors.
        "relative z-[80] inline-flex size-[34px] shrink-0 items-center justify-center rounded-[10px] text-foreground transition-colors hover:bg-nav-surface-hover focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-inset focus-visible:ring-ring",
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
  // Window chrome: the band around these is fixed, so they keep their size while the
  // page beside them scales.
  const buttonClass =
    "inline-flex size-[30px] shrink-0 items-center justify-center rounded-[10px] text-nav-icon-idle dark:text-nav-fg-muted transition-colors hover:bg-nav-surface-hover hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring";
  // The custom titlebar's are its window controls' 34px, with 18px glyphs.
  const customTitlebar = shouldUseCustomWindowTitlebar();
  const customSize = customTitlebar && "size-[34px]";
  const iconClass = customTitlebar
    ? "size-[18px]"
    : "size-icon !size-[calc(var(--icon-size)+1px)]";

  return (
    <div
      className={cn(
        "flex mt-[var(--studio-titlebar-navigation-margin-top,0px)] translate-y-[var(--studio-titlebar-navigation-offset-y,0px)] items-center",
        customTitlebar ? "gap-[4px]" : "gap-0.5",
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
          className={cn(buttonClass, customSize)}
        >
          <HugeiconsIcon
            icon={LayoutAlignLeftIcon}
            strokeWidth={1.75}
            className={iconClass}
          />
        </button>
      ) : (
        // Holds the slot the navbar's own trigger sits in, so it is the
        // button's fixed size, not a scaled one.
        <div
          aria-hidden="true"
          className={cn("size-[30px] shrink-0", customSize)}
        />
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
        className={cn(buttonClass, customSize)}
      >
        <ArrowLeft
          aria-hidden="true"
          strokeWidth={1.75}
          className={iconClass}
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
        className={cn(buttonClass, customSize)}
      >
        <ArrowRight
          aria-hidden="true"
          strokeWidth={1.75}
          className={iconClass}
        />
      </button>
    </div>
  );
}

export function WindowTitlebar({
  showSidebarSurface = false,
  pageHeaderInBand = false,
}: {
  showSidebarSurface?: boolean;
  /** A chat-like route, whose header shares the band and drags from under it. */
  pageHeaderInBand?: boolean;
}): ReactElement | null {
  const [enabled] = useState(shouldUseCustomWindowTitlebar);
  const [maximized, setMaximized] = useState(false);
  const { pinned, togglePinned } = useSidebarPin();
  // Outside SidebarProvider, so read the same media query the provider does.
  const isMobile = useIsMobileShell();

  const maximizeRefreshSequence = useRef(0);
  const maximizeRefreshTimer = useRef<number | null>(null);
  // The titlebar sits outside the sidebar wrapper, so it cannot inherit
  // --sidebar-width. Read the resized width from the same store instead.
  const { width, scale: widthScale } = useSidebarWidth();
  const sidebarWidth = showSidebarSurface
    ? pinned
      ? // The live value only exists mid-drag; otherwise the committed width.
        `var(--studio-sidebar-live-width, ${width * widthScale}px)`
      : "var(--studio-sidebar-collapsed-width,3rem)"
    : "0px";

  // Collapsed, the slot is exactly the 12px edge inset and three 34px buttons with their
  // two 4px gaps, so it never covers the page header that starts beside it. The drag
  // region starts where it ends.
  const titlebarNavigationWidth =
    showSidebarSurface && !pinned ? "122px" : sidebarWidth;
  const unifiedRow = showSidebarSurface && !isMobile && pageHeaderInBand;

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
        data-slot="window-titlebar"
        className="pointer-events-none absolute inset-x-0 top-0 z-[70] h-[var(--studio-custom-titlebar-height)] select-none text-foreground"
        // Marks a consumer of --studio-sidebar-live-width, so PANEL_RESIZE_SCOPED_VARS_ENABLED
        // writes the live width here instead of restyling the whole document per drag frame.
        data-titlebar-live-width-scope=""
        aria-label="Window titlebar"
      >
        {showSidebarSurface && (
          <div
            // 12px from the edge, as the window controls are from theirs. The toggle's glyph
            // then starts at x=20, in the sidebar's icon column below it.
            className="pointer-events-auto absolute left-0 top-0 flex h-full min-w-0 items-center pl-[12px]"
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
        {/* A chat-like page's header shares the band and Navbar drags from under it; every
            other screen leaves the band empty and drags from here. */}
        {!unifiedRow && (
          <div
            className="pointer-events-auto absolute top-0 h-full"
            style={{
              left: showSidebarSurface ? titlebarNavigationWidth : 0,
              right: "var(--studio-window-control-inset,122px)",
            }}
            onMouseDown={handleDragMouseDown}
            onDoubleClick={handleDragDoubleClick}
            aria-hidden="true"
          />
        )}
        <div
          // Mirrors the navigation: three 34px buttons, 4px apart, 12px from the edge.
          className="pointer-events-auto absolute right-[12px] top-0 flex h-full items-center gap-[4px]"
          role="toolbar"
          aria-label="Window controls"
        >
          <WindowControlButton
            label="Minimize window"
            onClick={() => runWindowAction((appWindow) => appWindow.minimize())}
          >
            <CaptionGlyph kind="minimize" />
          </WindowControlButton>
          <WindowControlButton
            label={maximized ? "Restore window" : "Maximize window"}
            onClick={() =>
              runWindowAction((appWindow) => appWindow.toggleMaximize())
            }
          >
            <CaptionGlyph kind={maximized ? "restore" : "maximize"} />
          </WindowControlButton>
          <WindowControlButton
            label="Close window"
            // No optimistic overlay here. Rust raises it only once the quit confirmations
            // have passed, and one of those can be a dialog asking whether to keep
            // training: painting "Closing Unsloth Desktop..." behind that question would
            // answer it before the user does. The wait this covers is the reap, and Rust's
            // app-closing arrives well ahead of that.
            onClick={() => runWindowAction((appWindow) => appWindow.close())}
            // Close also owns the corner, as on a native Windows titlebar.
            className="after:right-[-12px] hover:bg-[#e81123] hover:text-white active:bg-[#e81123]/60 dark:hover:text-white"
          >
            <CaptionGlyph kind="close" />
          </WindowControlButton>
        </div>
      </header>
      {/* A maximized window has no edges to resize, and the corner belongs to Close. */}
      {!maximized && (
        <>
          <div
            aria-hidden="true"
            className="pointer-events-auto fixed inset-x-2 top-0 h-1 cursor-n-resize"
            style={{ zIndex: Z_LAYER.WINDOW_RESIZE_EDGE }}
            onPointerDown={handleResizePointerDown("North")}
          />
          {/* resize grips stay above dialogs and notifications. */}
          <div
            aria-hidden="true"
            className="pointer-events-auto fixed inset-x-2 bottom-0 h-1 cursor-s-resize"
            style={{ zIndex: Z_LAYER.WINDOW_RESIZE_EDGE }}
            onPointerDown={handleResizePointerDown("South")}
          />
          <div
            aria-hidden="true"
            className="pointer-events-auto fixed inset-y-2 left-0 w-1 cursor-w-resize"
            style={{ zIndex: Z_LAYER.WINDOW_RESIZE_EDGE }}
            onPointerDown={handleResizePointerDown("West")}
          />
          <div
            aria-hidden="true"
            className="pointer-events-auto fixed inset-y-2 right-0 w-1 cursor-e-resize"
            style={{ zIndex: Z_LAYER.WINDOW_RESIZE_EDGE }}
            onPointerDown={handleResizePointerDown("East")}
          />
          <div
            aria-hidden="true"
            className="pointer-events-auto fixed left-0 top-0 size-3 cursor-nw-resize"
            style={{ zIndex: Z_LAYER.WINDOW_RESIZE_EDGE }}
            onPointerDown={handleResizePointerDown("NorthWest")}
          />
          <div
            aria-hidden="true"
            className="pointer-events-auto fixed right-0 top-0 size-3 cursor-ne-resize"
            style={{
              zIndex: Z_LAYER.WINDOW_RESIZE_EDGE,
              // keep the resize corner outside the close button.
              clipPath:
                "polygon(0 0, 100% 0, 100% 100%, calc(100% - 0.25rem) 100%, calc(100% - 0.25rem) 0.25rem, 0 0.25rem)",
            }}
            onPointerDown={handleResizePointerDown("NorthEast")}
          />
          <div
            aria-hidden="true"
            className="pointer-events-auto fixed bottom-0 left-0 size-3 cursor-sw-resize"
            style={{ zIndex: Z_LAYER.WINDOW_RESIZE_EDGE }}
            onPointerDown={handleResizePointerDown("SouthWest")}
          />
          <div
            aria-hidden="true"
            className="pointer-events-auto fixed bottom-0 right-0 size-3 cursor-se-resize"
            style={{ zIndex: Z_LAYER.WINDOW_RESIZE_EDGE }}
            onPointerDown={handleResizePointerDown("SouthEast")}
          />
        </>
      )}
    </>
  );
}
