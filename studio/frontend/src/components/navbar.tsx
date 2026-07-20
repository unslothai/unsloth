// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { shouldUseNativeMacWindowTitlebar } from "@/components/tauri/window-titlebar";
import { SidebarTrigger, useSidebar } from "@/components/ui/sidebar";
import { useState } from "react";

export function Navbar() {
  const { isMobile } = useSidebar();
  const [usesNativeMacTitlebar] = useState(shouldUseNativeMacWindowTitlebar);
  if (!isMobile) {
    return (
      <header className="absolute top-0 inset-x-0 z-40 h-[48px] pointer-events-none">
        {usesNativeMacTitlebar && (
          <div
            data-tauri-drag-region
            aria-hidden="true"
            className="pointer-events-auto absolute inset-x-0 top-0 h-[var(--studio-mac-titlebar-height,34px)] select-none"
          />
        )}
      </header>
    );
  }
  return (
    <header className="absolute top-0 inset-x-0 z-45 h-[48px] pointer-events-none">
      <div className="flex h-full items-start pt-[11px] pl-2">
        <SidebarTrigger className="pointer-events-auto !size-[34px]" />
      </div>
    </header>
  );
}
