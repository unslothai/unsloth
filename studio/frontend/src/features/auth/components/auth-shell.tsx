// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { publicAssetUrl } from "@/components/mascot-img";
import type { ReactNode } from "react";

export function AuthShell({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-[calc(100dvh-var(--studio-titlebar-height,0px))] flex-col items-center justify-center gap-6 bg-background px-4 py-10 sm:px-6">
      <div className="w-full max-w-[calc(400px*var(--ui-space-scale,1))] rounded-[28px] bg-card px-7 py-9 shadow-[0_1px_2px_rgba(0,0,0,0.04),0_12px_40px_-12px_rgba(0,0,0,0.12)] ring-1 ring-[color-mix(in_oklab,var(--foreground)_7%,transparent)] sm:px-9 sm:py-10 dark:shadow-none">
        {children}
      </div>
      <div className="flex items-center gap-1.5 text-muted-foreground text-xs">
        <img
          src={publicAssetUrl("unsloth-gem.png")}
          alt=""
          className="size-4 object-contain"
        />
        <span className="font-medium">Unsloth</span>
      </div>
    </div>
  );
}
