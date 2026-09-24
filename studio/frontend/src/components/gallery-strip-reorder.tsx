// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { StripDropCue } from "@/hooks/use-strip-reorder";
import { cn } from "@/lib/utils";

/** Insertion line beside a tile, centred in the strip's gap. */
export function StripDropLine({ edge }: { edge: StripDropCue["edge"] }) {
  return (
    <span
      aria-hidden={true}
      className={cn(
        "pointer-events-none absolute inset-y-1 z-40 w-0.5 rounded-full bg-primary",
        edge === "before"
          ? "-left-[calc(5px*var(--ui-space-scale,1))]"
          : "-right-[calc(5px*var(--ui-space-scale,1))]",
      )}
    />
  );
}
