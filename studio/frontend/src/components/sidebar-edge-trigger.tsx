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
 * Draggable window edge for the desktop sidebar, which collapses to zero width
 * and so has no edge of its own to grab. The same handle the pinned sidebar
 * uses, anchored to the window instead of to an off-screen panel: hover holds
 * the sidebar out, click or Enter pins it, drag resizes it.
 *
 * Hover and the keyboard are what land on macOS. A decorated window keeps its
 * resize border over the first few CSS pixels of the content and takes the
 * press there, so at 2px the click and the drag go to the window instead. Both
 * reach the 12px strip on Windows and Linux.
 *
 * Desktop only. On the web a collapsed sidebar keeps its icon rail, and that
 * rail already carries the handle.
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
    storedWidth,
    maxWidth,
    setWidth,
    resetWidth,
  } = useSidebar();
  // A drag that reaches the minimum pins the sidebar part-way through, which
  // would unmount this: the handle would lose pointer capture and the width
  // would never commit. Hold on until the pointer is released.
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
      // Only a primary press takes pointer capture, so only a primary press
      // has a release to wait for. Holding for any other would never clear.
      onPointerDownCapture={(event) => {
        if (event.button === 0) setHolding(true);
      }}
      onPointerUp={release}
      onPointerCancel={release}
      onLostPointerCapture={release}
      // Tab reaches the strip with no pointer to hold the sidebar out, so
      // focus does it instead. focusin and focusout bubble, which leaves the
      // strip's own focus handling to the shared handle.
      onFocus={() => setPeeking(true)}
      onBlur={() => setPeeking(false)}
    >
      <PanelResizeHandle
        edge="right"
        // Follows the pin, which a drag flips part-way through: from there the
        // handle resizes the sidebar it just opened and commits on release.
        open={pinned}
        width={width}
        stored={storedWidth}
        min={SIDEBAR_WIDTH_MIN}
        max={maxWidth}
        clamp={clampSidebarWidth}
        setWidth={setWidth}
        resetWidth={resetWidth}
        onToggle={toggleSidebar}
        // Start from what is on screen: held out, the panel is already at its
        // full width, so a drag carries on from there instead of jumping.
        measure={() => (peeking ? width : 0)}
        target={() =>
          ref.current?.closest<HTMLElement>('[data-slot="sidebar-wrapper"]') ??
          null
        }
        cssVar="--sidebar-width"
        rootVar="--studio-sidebar-live-width"
        // The sidebar declares --sidebar-width on itself, so a live width
        // painted any higher up is shadowed. Queried, not `closest`: this
        // strip is the sidebar's sibling.
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
        // The sidebar coming out says it better than a label would.
        hideTooltip={true}
        onHoverChange={setPeeking}
        dataSlot="sidebar-edge-trigger"
        className={cn(
          // Below the titlebar so the window controls and drag region keep
          // their clicks, above the panel it holds out so the edge still
          // answers under the sidebar. `block` beats the handle's
          // `hidden sm:block`, which a narrowed window would trip.
          "fixed bottom-0 left-0 right-auto top-[var(--studio-desktop-titlebar-height,48px)] z-[55] block",
          // The window edge and nothing more: every pixel here takes a click
          // from the page. Wider where the window is undecorated (Windows,
          // Linux, see `setup_custom_titlebar`): there the toolkit hit-tests
          // its own resize border inside the window, so 2px would land
          // entirely within it. macOS keeps that border outside the content.
          usesCustomTitlebar ? "w-3" : "w-0.5",
          // The sidebar sliding out is the hover feedback, not a hairline
          // hanging down the middle of the page.
          "after:hidden",
          // That hairline is also where the shared handle marks focus, and on
          // a 2px strip it would sit off-screen, so mark it on the strip.
          "focus-visible:bg-sidebar-ring/60",
          // The two-headed resize cursor other chat apps put here, not the
          // one-way `e-resize` of a collapsed panel border.
          "cursor-col-resize!",
          className,
        )}
      />
    </div>
  );
}
