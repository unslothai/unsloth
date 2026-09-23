// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { Loader2Icon } from "lucide-react";
import { cn } from "@/lib/utils";

/** App-wide spinner inheriting the current text color. `label` overrides the announcement
 * where "loading" is not what it means (a sidebar chat is generating). `variant="ring"` turns
 * only an arc over a static track, so it reads as fixed in place rather than drifting. */
function Spinner({
  className,
  label = "Loading",
  variant = "arc",
  "data-testid": dataTestId,
}: {
  className?: string;
  label?: string;
  variant?: "arc" | "ring";
  "data-testid"?: string;
}) {
  if (variant === "ring") {
    return (
      <svg
        viewBox="0 0 24 24"
        fill="none"
        role="status"
        aria-label={label}
        data-testid={dataTestId}
        className={cn("size-4 shrink-0", className)}
      >
        <circle
          cx="12"
          cy="12"
          r="9"
          stroke="currentColor"
          strokeOpacity={0.2}
          strokeWidth={2.5}
        />
        <circle
          cx="12"
          cy="12"
          r="9"
          stroke="currentColor"
          strokeWidth={2.5}
          strokeLinecap="round"
          // About a quarter of the circumference.
          strokeDasharray="14 43"
          className="origin-center animate-spin [transform-box:view-box] will-change-transform"
        />
      </svg>
    );
  }
  return (
    <Loader2Icon
      role="status"
      aria-label={label}
      data-testid={dataTestId}
      className={cn("size-4 shrink-0 animate-spin", className)}
    />
  );
}

export { Spinner };
