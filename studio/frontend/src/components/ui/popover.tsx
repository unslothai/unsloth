// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { Popover as PopoverPrimitive } from "radix-ui";
import type * as React from "react";

import { useDialogPortalContainer } from "@/components/ui/dialog";
import { useSnappedPaddingRef } from "@/lib/snap-padding";
import { useWindowChromeCollisionPadding } from "@/lib/window-chrome";
import { cn } from "@/lib/utils";

function Popover({
  ...props
}: React.ComponentProps<typeof PopoverPrimitive.Root>) {
  return <PopoverPrimitive.Root data-slot="popover" {...props} />;
}

function PopoverTrigger({
  ...props
}: React.ComponentProps<typeof PopoverPrimitive.Trigger>) {
  return <PopoverPrimitive.Trigger data-slot="popover-trigger" {...props} />;
}

function PopoverContent({
  className,
  align = "center",
  sideOffset = 0,
  collisionPadding,
  container,
  ref,
  ...props
}: React.ComponentProps<typeof PopoverPrimitive.Content> & {
  container?: HTMLElement | null;
}) {
  const snappedRef = useSnappedPaddingRef(ref);
  // Inside a modal dialog the body scroll lock swallows wheel events on
  // body-portaled content; portal into the dialog instead (like Select).
  const dialogContainer = useDialogPortalContainer();
  return (
    <PopoverPrimitive.Portal
      container={container ?? dialogContainer ?? undefined}
    >
      <PopoverPrimitive.Content
        ref={snappedRef}
        data-slot="popover-content"
        collisionPadding={useWindowChromeCollisionPadding(collisionPadding)}
        align={align}
        sideOffset={sideOffset}
        // Padding alone doesn't shrink it, so cap it and scroll.
        className={cn(
          "bg-popover text-popover-foreground data-open:animate-in data-closed:animate-out data-closed:fade-out-0 data-open:fade-in-0 data-closed:zoom-out-95 data-open:zoom-in-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2 flex flex-col gap-4 rounded-2xl p-4 text-sm duration-100 z-50 w-72 max-w-[calc(100vw-32px)] max-h-(--radix-popover-content-available-height) overflow-y-auto origin-(--radix-popover-content-transform-origin) outline-hidden",
          className,
        )}
        {...props}
      />
    </PopoverPrimitive.Portal>
  );
}

function PopoverAnchor({
  ...props
}: React.ComponentProps<typeof PopoverPrimitive.Anchor>) {
  return <PopoverPrimitive.Anchor data-slot="popover-anchor" {...props} />;
}

function PopoverHeader({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="popover-header"
      className={cn("flex flex-col gap-1 text-sm", className)}
      {...props}
    />
  );
}

function PopoverTitle({ className, ...props }: React.ComponentProps<"h2">) {
  return (
    <div
      data-slot="popover-title"
      className={cn("text-base font-medium", className)}
      {...props}
    />
  );
}

function PopoverDescription({
  className,
  ...props
}: React.ComponentProps<"p">) {
  return (
    <p
      data-slot="popover-description"
      className={cn("text-muted-foreground", className)}
      {...props}
    />
  );
}

export {
  Popover,
  PopoverAnchor,
  PopoverContent,
  PopoverDescription,
  PopoverHeader,
  PopoverTitle,
  PopoverTrigger,
};
