// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { cn } from "@/lib/utils";

/** Spinner in current text color; `label` overrides the "loading" announcement. */
function Spinner({
  className,
  label = "Loading",
  "data-testid": dataTestId,
}: {
  className?: string;
  label?: string;
  "data-testid"?: string;
}) {
  // lucide's loader-circle, but the arc turns inside the svg: WebKitGTK's compositor wobbles a
  // rotating <svg> by up to 1.5px when it sits off the device pixel grid.
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      role="status"
      aria-label={label}
      data-testid={dataTestId}
      className={cn("lucide lucide-loader-circle size-4 shrink-0", className)}
    >
      <g className="animate-spin [transform-box:view-box] [transform-origin:12px_12px]">
        <path d="M21 12a9 9 0 1 1-6.219-8.56" />
      </g>
    </svg>
  );
}

export { Spinner };
