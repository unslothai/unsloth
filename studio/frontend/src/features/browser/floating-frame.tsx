// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import { type FloatingEdge, startFloatingGesture } from "./floating-preview";

// Straddling the frame, mostly outside it: a native page covers whatever lies over it.
const HANDLES: { edge: FloatingEdge; className: string }[] = [
  { edge: "n", className: "inset-x-3 -top-1.5 h-2.5 cursor-ns-resize" },
  { edge: "s", className: "inset-x-3 -bottom-1.5 h-2.5 cursor-ns-resize" },
  { edge: "w", className: "inset-y-3 -left-1.5 w-2.5 cursor-ew-resize" },
  { edge: "e", className: "inset-y-3 -right-1.5 w-2.5 cursor-ew-resize" },
  { edge: "nw", className: "-left-1.5 -top-1.5 size-4 cursor-nwse-resize" },
  { edge: "ne", className: "-right-1.5 -top-1.5 size-4 cursor-nesw-resize" },
  { edge: "sw", className: "-bottom-1.5 -left-1.5 size-4 cursor-nesw-resize" },
  { edge: "se", className: "-bottom-1.5 -right-1.5 size-4 cursor-nwse-resize" },
];

export function FloatingResizeHandles() {
  return (
    <>
      {HANDLES.map(({ edge, className }) => (
        <div
          key={edge}
          aria-hidden={true}
          data-native-clickable=""
          data-floating-resize={edge}
          onPointerDown={(event) => startFloatingGesture(event, edge)}
          className={cn("absolute z-10 touch-none", className)}
        />
      ))}
    </>
  );
}
