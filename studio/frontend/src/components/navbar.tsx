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
  const { isMobile, pinned, peeking, togglePinned } = useSidebar();
  const [usesNativeMacTitlebar] = useState(shouldUseNativeMacWindowTitlebar);
  const [usesCustomTitlebar] = useState(shouldUseCustomWindowTitlebar);

  if (!isMobile) {
    return (
      <>
        <header className="pointer-events-none absolute inset-x-0 top-0 z-40 h-[calc(48px*var(--ui-space-scale,1))]">
          {usesNativeMacTitlebar && (
            <div
              data-tauri-drag-region
              aria-hidden="true"
              className="pointer-events-auto absolute inset-x-0 top-0 h-[var(--studio-mac-titlebar-height,34px)] select-none"
            />
          )}
        </header>

        {/* A held-out sidebar brings its own copy of this cluster, in the same
            place. */}
        {usesNativeMacTitlebar && !pinned && !peeking && (
          <DesktopTitlebarNavigation
            expanded={false}
            onToggleSidebar={togglePinned}
            className="pointer-events-auto absolute left-[calc(var(--studio-mac-traffic-light-inset,78px)+calc(6px*var(--ui-space-scale,1)))] top-px z-[60]"
          />
        )}
      </>
    );
  }
  // Desktop windows can land here under Windows text scaling, so sit inside the
  // custom titlebar band instead of under its z-[70].
  return (
    <header
      className={cn(
        "absolute top-0 inset-x-0 pointer-events-none",
        usesCustomTitlebar
          ? "z-[80] h-[var(--studio-custom-titlebar-height,34px)]"
          : "z-[45] h-[calc(48px*var(--ui-space-scale,1))]",
      )}
    >
      <div
        className={cn(
          "flex h-full",
          usesCustomTitlebar
            ? "items-center pl-3"
            : usesNativeMacTitlebar
              ? "items-start pt-[calc(11px*var(--ui-space-scale,1))] pl-[calc(var(--studio-mac-traffic-light-inset,78px)+calc(6px*var(--ui-space-scale,1)))]"
              : "items-start pt-[calc(11px*var(--ui-space-scale,1))] pl-2",
        )}
      >
        {/* Scales with the header, except in the fixed titlebar band. */}
        <SidebarTrigger
          className={cn(
            "pointer-events-auto",
            usesCustomTitlebar
              ? "!size-[34px]"
              : "!size-[calc(34px*var(--ui-space-scale,1))]",
          )}
        />
      </div>
    </header>
  );
}
