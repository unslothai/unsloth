// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ReactNode } from "react";

export function HubTopBar({ children }: { children: ReactNode }) {
  return (
    <div className="hub-canvas shrink-0">
      <div className="mx-auto flex w-full max-w-[var(--hub-measure)] flex-col gap-2 px-5 pb-1 pt-1 sm:px-8 md:gap-4 md:pb-3 md:pt-8">
        {children}
      </div>
    </div>
  );
}
