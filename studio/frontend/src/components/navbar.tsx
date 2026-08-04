// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  DesktopTitlebarNavigation,
  shouldUseNativeMacWindowTitlebar,
} from "@/components/tauri/window-titlebar";
import { SidebarTrigger, useSidebar } from "@/components/ui/sidebar";
import { useEffect, useState } from "react";

export function Navbar() {
  const { isMobile, pinned, togglePinned } = useSidebar();
  const [usesNativeMacTitlebar] = useState(shouldUseNativeMacWindowTitlebar);

  const [windowFocused, setWindowFocused] = useState(() => document.hasFocus());

  useEffect(() => {
    if (!usesNativeMacTitlebar) return;
    const handleFocus = () => setWindowFocused(true);
    const handleBlur = () => setWindowFocused(false);
    window.addEventListener("focus", handleFocus);
    window.addEventListener("blur", handleBlur);
    return () => {
      window.removeEventListener("focus", handleFocus);
      window.removeEventListener("blur", handleBlur);
    };
  }, [usesNativeMacTitlebar]);
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

        {usesNativeMacTitlebar && !windowFocused && (
          <div
            aria-hidden="true"
            className="pointer-events-none absolute left-[14px] top-[11px] z-[55] flex gap-2"
          >
            <span className="size-3 rounded-full bg-[#b8b8b8] ring-1 ring-black/10" />
            <span className="size-3 rounded-full bg-[#b8b8b8] ring-1 ring-black/10" />
            <span className="size-3 rounded-full bg-[#b8b8b8] ring-1 ring-black/10" />
          </div>
        )}

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
  return (
    <header className="absolute top-0 inset-x-0 z-[45] h-[48px] pointer-events-none">
      <div className="flex h-full items-start pt-[11px] pl-2">
        <SidebarTrigger className="pointer-events-auto !size-[34px]" />
      </div>
    </header>
  );
}
