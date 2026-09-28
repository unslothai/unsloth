// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { StripDropCue } from "@/hooks/use-strip-reorder";
import { cn } from "@/lib/utils";

/** Insertion line beside a tile, centred in the strip's gap. `axis="y"` draws it across a list row. */
export function StripDropLine({
  edge,
  axis = "x",
}: {
  edge: StripDropCue["edge"];
  axis?: "x" | "y";
}) {
  return (
    <span
      aria-hidden={true}
      className={cn(
        "pointer-events-none absolute z-40 rounded-full bg-primary",
        axis === "x"
          ? cn(
              "inset-y-1 w-0.5",
              edge === "before"
                ? "-left-[calc(5px*var(--ui-space-scale,1))]"
                : "-right-[calc(5px*var(--ui-space-scale,1))]",
            )
          : cn(
              "inset-x-1 h-0.5",
              edge === "before"
                ? "-top-[calc(3px*var(--ui-space-scale,1))]"
                : "-bottom-[calc(3px*var(--ui-space-scale,1))]",
            ),
      )}
    />
  );
}
