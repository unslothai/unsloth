// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type * as React from "react";

import { cn } from "@/lib/utils";

function DashboardLayout({
  className,
  children,
  ...props
}: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="dashboard-layout"
      className={cn(
        "min-h-screen w-full bg-background",
        "flex justify-center",
        className,
      )}
      {...props}
    >
      <div className="w-full max-w-7xl px-6 py-8 lg:px-8">{children}</div>
    </div>
  );
}

export { DashboardLayout };
