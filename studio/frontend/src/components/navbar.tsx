// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  DesktopTitlebarNavigation,
  shouldUseCustomWindowTitlebar,
  shouldUseNativeMacWindowTitlebar,
} from "@/components/tauri/window-titlebar";
import { SidebarTrigger, useSidebar } from "@/components/ui/sidebar";
import { cn } from "@/lib/utils";
import { useState } from "react";

export function Navbar() {
  const { isMobile, pinned, togglePinned } = useSidebar();
  const [usesNativeMacTitlebar] = useState(shouldUseNativeMacWindowTitlebar);
  const [usesCustomTitlebar] = useState(shouldUseCustomWindowTitlebar);

  if (!isMobile) {
    return (
      <>
        <header className="pointer-events-none absolute inset-x-0 top-0 z-40 h-[48px]">
          {usesNativeMacTitlebar && (
            <div
              data-tauri-drag-region
              aria-hidden="true"
              className="pointer-events-auto absolute inset-x-0 top-0 h-[var(--studio-mac-titlebar-height,34px)] select-none"
            />
          )}
        </header>

        {usesNativeMacTitlebar && !pinned && (
          <DesktopTitlebarNavigation
            expanded={false}
            onToggleSidebar={togglePinned}
            className="pointer-events-auto absolute left-[calc(var(--studio-mac-traffic-light-inset,78px)+6px)] top-px z-[60]"
          />
        )}
      </>
    );
  }
  // Windows text scaling shrinks the CSS viewport, so a desktop window can land
  // here; sit inside the custom titlebar band instead of buried under its z-[70].
  return (
    <header
      className={cn(
        "absolute top-0 inset-x-0 pointer-events-none",
        usesCustomTitlebar
          ? "z-[80] h-[var(--studio-custom-titlebar-height,34px)]"
          : "z-[45] h-[48px]",
      )}
    >
      <div
        className={cn(
          "flex h-full",
          usesCustomTitlebar
            ? "items-center pl-3"
            : "items-start pt-[11px] pl-2",
        )}
      >
        <SidebarTrigger className="pointer-events-auto !size-[34px]" />
      </div>
    </header>
  );
}
