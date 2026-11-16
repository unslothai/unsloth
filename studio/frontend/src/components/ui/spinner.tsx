// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { Loader2Icon } from "lucide-react";
import { cn } from "@/lib/utils";

/** WebKitGTK's compositor wobbles a rotating <svg> by up to 1.5px when it sits off the device pixel
 *  grid, so the Linux desktop turns the arc inside it instead. SMIL, because WebKitGTK before 2.40
 *  does not run CSS animations on SVG children. */
function turnsInside(): boolean {
  if (typeof document === "undefined") return false;
  const html = document.documentElement.classList;
  return html.contains("tauri") && html.contains("render-linux");
}

/** index.css slows .animate-spin to 1.5s under reduced motion; SMIL does not see that rule. */
function turnSeconds(): number {
  const html = document.documentElement.classList;
  const reduced =
    html.contains("force-reduced-motion") ||
    (!html.contains("force-motion") &&
      window.matchMedia?.("(prefers-reduced-motion: reduce)").matches === true);
  return reduced ? 1.5 : 1;
}

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
  if (!turnsInside()) {
    return (
      <Loader2Icon
        role="status"
        aria-label={label}
        data-testid={dataTestId}
        className={cn("size-4 shrink-0 animate-spin", className)}
      />
    );
  }
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
      <g>
        <path d="M21 12a9 9 0 1 1-6.219-8.56" />
        <animateTransform
          attributeName="transform"
          type="rotate"
          from="0 12 12"
          to="360 12 12"
          dur={`${turnSeconds()}s`}
          repeatCount="indefinite"
        />
      </g>
    </svg>
  );
}

export { Spinner };
