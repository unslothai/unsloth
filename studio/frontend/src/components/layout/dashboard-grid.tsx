// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type * as React from "react";

import { cn } from "@/lib/utils";

const colsVariants = {
  3: "lg:grid-cols-3",
  4: "lg:grid-cols-4",
} as const;

function DashboardGrid({
  className,
  cols = 3,
  ...props
}: React.ComponentProps<"div"> & { cols?: 3 | 4 }) {
  return (
    <div
      data-slot="dashboard-grid"
      className={cn(
        "grid grid-cols-1 gap-6 md:grid-cols-2",
        colsVariants[cols],
        className,
      )}
      {...props}
    />
  );
}

export { DashboardGrid };
