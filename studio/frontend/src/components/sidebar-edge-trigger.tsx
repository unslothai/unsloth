// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { shouldUseCustomWindowTitlebar } from "@/components/tauri/window-titlebar";
import { PanelResizeHandle } from "@/components/ui/panel-resize-handle";
import { useSidebar } from "@/components/ui/sidebar";
import {
  SIDEBAR_WIDTH_MIN,
  clampSidebarWidth,
} from "@/hooks/use-sidebar-width";
import { useT } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { cn } from "@/lib/utils";
import { type ReactElement, useRef, useState } from "react";

/**
 * Draggable window edge for the zero-width desktop sidebar. On macOS a decorated window's resize
 * border takes presses in the first few pixels, so hover and keyboard are what work there.
 */
export function SidebarEdgeTrigger({
  className,
}: {
  className?: string;
}): ReactElement | null {
  const t = useT();
  const ref = useRef<HTMLDivElement>(null);
  const [usesCustomTitlebar] = useState(shouldUseCustomWindowTitlebar);
  const {
    isMobile,
    openMobile,
    peeking,
    pinned,
    setPeeking,
    toggleSidebar,
    width,
    widthScale,
    storedWidth,
    maxWidth,
    setWidth,
    resetWidth,
  } = useSidebar();
  // Pinning mid-drag would unmount this and lose pointer capture; hold until release.
  const [holding, setHolding] = useState(false);
  const release = () => setHolding(false);

  const sidebarShowing = isMobile ? openMobile : pinned;
  if (!isTauri || (sidebarShowing && !holding)) {
    return null;
  }

  return (
    <div
      ref={ref}
      className="contents"
      // Only a primary press takes pointer capture, so only it has a release to wait for.
      onPointerDownCapture={(event) => {
        if (event.button === 0) setHolding(true);
      }}
      onPointerUp={release}
      onPointerCancel={release}
      onLostPointerCapture={release}
      // Keyboard users have no pointer to hold the sidebar out, so focus does it.
      onFocus={() => setPeeking(true)}
      onBlur={() => setPeeking(false)}
    >
      <PanelResizeHandle
        edge="right"
        // Follows the pin, which a drag flips midway; then the handle resizes and commits on release.
        open={pinned}
        width={width}
        stored={storedWidth}
        min={SIDEBAR_WIDTH_MIN}
        max={maxWidth}
        scale={widthScale}
        clamp={clampSidebarWidth}
        setWidth={setWidth}
        resetWidth={resetWidth}
        onToggle={toggleSidebar}
        // Start from on-screen width so a drag from a held-out panel does not jump.
        measure={() => (peeking ? width : 0)}
        target={() =>
          ref.current?.closest<HTMLElement>('[data-slot="sidebar-wrapper"]') ??
          null
        }
        cssVar="--sidebar-width"
        rootVar="--studio-sidebar-live-width"
        // The sidebar declares --sidebar-width on itself; query it, since this strip is a sibling.
        scopedTarget={() =>
          ref.current
            ?.closest<HTMLElement>('[data-slot="sidebar-wrapper"]')
            ?.querySelector<HTMLElement>('[data-slot="sidebar"]') ?? null
        }
        rootVarTargets={() =>
          Array.from(
            document.querySelectorAll<HTMLElement>(
              "[data-titlebar-live-width-scope]",
            ),
          )
        }
        label={t("shell.aria.resizeSidebar")}
        toggleLabel={t("shell.aria.openSidebar")}
        hideTooltip={true}
        onHoverChange={setPeeking}
        dataSlot="sidebar-edge-trigger"
        className={cn(
          // Below the titlebar, above the held-out panel. `block` beats the handle's `hidden sm:block`.
          "fixed bottom-0 left-0 right-auto top-[var(--studio-desktop-titlebar-height,48px)] z-[55] block",
          // Undecorated windows (Windows, Linux) hit-test their resize border inside the window, so wider.
          usesCustomTitlebar ? "w-3" : "w-0.5",
          "after:hidden",
          // The shared handle's focus mark would sit off-screen on a 2px strip, so mark the strip.
          "focus-visible:bg-sidebar-ring/60",
          "cursor-col-resize!",
          className,
        )}
      />
    </div>
  );
}
