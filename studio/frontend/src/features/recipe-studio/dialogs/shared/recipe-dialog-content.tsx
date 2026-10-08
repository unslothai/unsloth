// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ComponentProps, ReactElement } from "react";

import { DialogContent } from "@/components/ui/dialog";
import { cn } from "@/lib/utils";

/** A recipe dialog over the canvas. The overlay is clear, so the shadow is its edge: Firefox's
 *  .scroll-rounded clip would cut it, so an inner viewport scrolls instead of the shadowed box. */
export function RecipeDialogContent({
  className,
  viewportClassName,
  children,
  ...props
}: ComponentProps<typeof DialogContent> & {
  viewportClassName?: string;
}): ReactElement {
  return (
    <DialogContent
      position="absolute"
      overlayPosition="absolute"
      overlayClassName="bg-transparent"
      className={cn(
        "flex max-h-[min(calc(650px*var(--ui-space-scale,1)),calc(100dvh-var(--studio-window-chrome-top,0px)-2rem))] flex-col overflow-hidden p-0 sm:max-w-2xl shadow-border",
        className,
      )}
      {...props}
    >
      <div
        data-slot="recipe-dialog-viewport"
        className={cn(
          "grid min-h-0 gap-6 overflow-y-auto overflow-x-hidden scroll-rounded rounded-4xl px-7 pt-8 pb-7",
          viewportClassName,
        )}
      >
        {children}
      </div>
    </DialogContent>
  );
}
