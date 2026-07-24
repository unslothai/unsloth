// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import type { ReactElement } from "react";

type WireLabelProps = {
  label: string;
  /** Absolute flow coords of the source (output) handle the wire leaves from. */
  x: number;
  y: number;
  active?: boolean;
};

/**
 * A small net label rendered on a wire, just past the output pin and above the
 * lead — the recipe analogue of a schematic net name (D0, S0, Out). Must be
 * used inside React Flow's <EdgeLabelRenderer>.
 */
export function WireLabel({
  label,
  x,
  y,
  active = false,
}: WireLabelProps): ReactElement {
  return (
    <div
      className={cn(
        "nodrag nopan pointer-events-none absolute whitespace-nowrap rounded-md border px-1.5 py-0.5 text-[10px] font-medium leading-none shadow-sm",
        active
          ? "border-primary/50 bg-primary/10 text-primary"
          : "border-border/70 bg-background/90 text-muted-foreground",
      )}
      style={{
        transform: `translate(${x + 8}px, ${y - 5}px) translateY(-100%)`,
      }}
    >
      {label}
    </div>
  );
}
