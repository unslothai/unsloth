// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import type { ComponentProps, ReactNode } from "react";

/**
 * A scrollbar paints outside the element's border-radius clip, so the wrapper's padding insets
 * the scroller instead. Padding belongs on `className`, never on `scrollerClassName`.
 */
export function ScrollPane({
  className,
  scrollerClassName,
  children,
  ...props
}: {
  className?: string;
  scrollerClassName?: string;
  children: ReactNode;
} & Omit<ComponentProps<"div">, "className" | "children">) {
  return (
    <div className={cn("min-w-0", className)} {...props}>
      <pre className={cn("min-w-0", scrollerClassName)}>{children}</pre>
    </div>
  );
}
