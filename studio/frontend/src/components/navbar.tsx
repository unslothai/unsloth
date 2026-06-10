// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { SidebarTrigger, useSidebar } from "@/components/ui/sidebar";

export function Navbar() {
  const { isMobile } = useSidebar();
  if (!isMobile) {
    return (
      <header className="absolute top-0 inset-x-0 z-40 h-[48px] pointer-events-none" />
    );
  }
  return (
    <header className="absolute top-0 inset-x-0 z-40 h-[48px] pointer-events-none">
      <div className="flex h-full items-start pt-[11px] pl-2">
        <SidebarTrigger className="pointer-events-auto !size-[34px]" />
      </div>
    </header>
  );
}
