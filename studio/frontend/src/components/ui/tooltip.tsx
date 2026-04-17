// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Tooltip as TooltipPrimitive } from "radix-ui";
import { createContext, useCallback, useContext, useState } from "react";
import type * as React from "react";

import { cn } from "@/lib/utils";

type ToggleFn = () => void;
const TooltipToggleCtx = createContext<ToggleFn | null>(null);

function TooltipProvider({
  delayDuration = 400,
  ...props
}: React.ComponentProps<typeof TooltipPrimitive.Provider>) {
  return (
    <TooltipPrimitive.Provider
      data-slot="tooltip-provider"
      delayDuration={delayDuration}
      {...props}
    />
  );
}

function Tooltip({
  open: controlledOpen,
  onOpenChange: controlledOnOpenChange,
  ...props
}: React.ComponentProps<typeof TooltipPrimitive.Root>) {
  const isControlled = controlledOpen !== undefined;
  const [clickOpen, setClickOpen] = useState(false);

  const onOpenChange = useCallback(
    (nextOpen: boolean) => {
      if (!nextOpen) setClickOpen(false);
      controlledOnOpenChange?.(nextOpen);
    },
    [controlledOnOpenChange],
  );

  const toggle = useCallback(() => {
    setClickOpen((prev) => !prev);
  }, []);

  return (
    <TooltipProvider>
      <TooltipToggleCtx.Provider value={toggle}>
        <TooltipPrimitive.Root
          data-slot="tooltip"
          open={isControlled ? controlledOpen : clickOpen || undefined}
          onOpenChange={onOpenChange}
          {...props}
        />
      </TooltipToggleCtx.Provider>
    </TooltipProvider>
  );
}

function TooltipTrigger({
  onClick,
  ...props
}: React.ComponentProps<typeof TooltipPrimitive.Trigger>) {
  const toggle = useContext(TooltipToggleCtx);

  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLButtonElement>) => {
      e.preventDefault();
      toggle?.();
      onClick?.(e);
    },
    [toggle, onClick],
  );

  return (
    <TooltipPrimitive.Trigger
      data-slot="tooltip-trigger"
      onClick={handleClick}
      {...props}
    />
  );
}

function TooltipContent({
  className,
  sideOffset = 0,
  children,
  ...props
}: React.ComponentProps<typeof TooltipPrimitive.Content>) {
  return (
    <TooltipPrimitive.Portal>
      <TooltipPrimitive.Content
        data-slot="tooltip-content"
        sideOffset={sideOffset}
        className={cn(
          "data-open:animate-in data-open:fade-in-0 data-open:zoom-in-95 data-[state=delayed-open]:animate-in data-[state=delayed-open]:fade-in-0 data-[state=delayed-open]:zoom-in-95 data-closed:animate-out data-closed:fade-out-0 data-closed:zoom-out-95 data-[side=bottom]:slide-in-from-top-2 data-[side=left]:slide-in-from-right-2 data-[side=right]:slide-in-from-left-2 data-[side=top]:slide-in-from-bottom-2 rounded-2xl corner-squircle px-3 py-1.5 text-xs **:data-[slot=kbd]:rounded-4xl bg-foreground text-background border border-foreground/40 shadow-lg z-[999999] w-fit max-w-xs origin-(--radix-tooltip-content-transform-origin)",
          className,
        )}
        {...props}
      >
        {children}
        <TooltipPrimitive.Arrow className="size-2.5 translate-y-[calc(-50%_-_2px)] rotate-45 rounded-[2px] data-[side=left]:translate-x-[-1.5px] data-[side=right]:translate-x-[1.5px] bg-foreground fill-foreground z-[999999] translate-y-[calc(-50%_-_2px)]" />
      </TooltipPrimitive.Content>
    </TooltipPrimitive.Portal>
  );
}

export { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger };
