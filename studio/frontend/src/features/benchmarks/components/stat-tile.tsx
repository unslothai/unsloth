// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import type { ReactElement, ReactNode } from "react";
import { BENCH_CARD } from "./bench-ui";

/** The headline number above a Benchmarks chart: label, value, one muted line of detail. */
export function StatTile({
  label,
  value,
  detail,
}: {
  label: string;
  value: ReactNode;
  detail?: ReactNode;
}): ReactElement {
  return (
    <div className={cn(BENCH_CARD, "flex min-w-0 flex-col gap-1.5 px-5 py-4")}>
      <span className="text-ui-11 font-medium tracking-nav text-muted-foreground">
        {label}
      </span>
      <span className="truncate font-heading text-ui-22 font-semibold leading-tight tracking-[-0.02em] tabular-nums text-foreground">
        {value}
      </span>
      {detail && (
        <span className="truncate text-ui-11 text-muted-foreground">
          {detail}
        </span>
      )}
    </div>
  );
}
