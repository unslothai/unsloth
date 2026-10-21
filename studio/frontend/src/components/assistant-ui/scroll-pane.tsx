// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import type { ComponentProps, ReactNode } from "react";

/**
 * A capped, scrollable pane with rounded corners: tool results, thinking traces,
 * pasted text.
 *
 * The wrapper paints the background and the scroller sits inside its padding. That split
 * is the point. A space-reserving scrollbar (the app sets `scrollbar-width: thin`, so
 * ~11px) is painted outside the element's border-radius clip, so a rounded scroller
 * renders square on the scrollbar's side: right corners for a vertical bar, bottom for a
 * horizontal one.
 *
 * Clipping it does not help. `clip-path`, `contain: paint`, an opaque `mask-image` and an
 * `overflow: hidden` parent were all tried in the thread and all left the corners square,
 * since a composited scroller's scrollbar escapes an ancestor's rounded clip too. Only
 * insetting the scrollbar away from the corners works.
 *
 * So padding belongs on `className`, never on `scrollerClassName`, or the scrollbar is
 * flush again. The max-height caps the scroller, so outer height is that plus padding.
 */
export function ScrollPane({
  className,
  scrollerClassName,
  children,
  ...props
}: {
  /** Wrapper: background, radius, border, margins, and the padding that insets the scrollbar. */
  className?: string;
  /** Scroller: max-height, overflow, and typography. No padding and no background here. */
  scrollerClassName?: string;
  children: ReactNode;
} & Omit<ComponentProps<"div">, "className" | "children">) {
  return (
    <div className={cn("min-w-0", className)} {...props}>
      <pre className={cn("min-w-0", scrollerClassName)}>{children}</pre>
    </div>
  );
}
