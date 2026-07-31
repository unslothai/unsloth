// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Tooltip as TooltipPrimitive } from "radix-ui";
import { createContext, useCallback, useContext, useState } from "react";
import type * as React from "react";

import { cn } from "@/lib/utils";

type ToggleFn = () => void;
const TooltipToggleCtx = createContext<ToggleFn | null>(null);

// Default to instant open (no hover delay). Most tooltips in the app —
// chat-area icon labels, sidebar nav labels, the context/token
// calculators — should feel snappy. Consumers that want a delay still
// pass an explicit `delayDuration` prop.
function TooltipProvider({
  delayDuration = 0,
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
    <TooltipToggleCtx.Provider value={toggle}>
      <TooltipPrimitive.Root
        data-slot="tooltip"
        open={isControlled ? controlledOpen : clickOpen || undefined}
        onOpenChange={onOpenChange}
        {...props}
      />
    </TooltipToggleCtx.Provider>
  );
}

function TooltipTrigger({
  onClick,
  ...props
}: React.ComponentProps<typeof TooltipPrimitive.Trigger>) {
  const toggle = useContext(TooltipToggleCtx);

  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLButtonElement>) => {
      // Run the composed handler first: when this trigger wraps another Radix
      // trigger (e.g. DialogTrigger around an attachment tile), that trigger's
      // action is skipped if the event is already default-prevented.
      onClick?.(e);
      // preventDefault keeps Radix Tooltip's internal close-on-click from
      // undoing the tap-toggle below (its composed handler checks it).
      e.preventDefault();
      toggle?.();
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

type TooltipVariant = "default" | "rich" | "none";

// `default` applies the compact black-pill styling shared with the
// sidebar/chat icon labels. `rich` opts into the larger multi-row
// popover surface used for timing/context breakdowns. `none` is an
// escape hatch for tooltips that need to bring their own surface.
function TooltipContent({
  variant = "default",
  className,
  sideOffset = 0,
  children,
  ...props
}: React.ComponentProps<typeof TooltipPrimitive.Content> & {
  variant?: TooltipVariant;
}) {
  // Single-line compact tooltips render as a full pill; wrapped ones keep
  // the squarer corners so tall pills do not look like capsules. A ref
  // callback measures on mount: Radix mounts the portal content without
  // re-rendering this wrapper, so an effect here would never see the node.
  const measureRef = useCallback(
    (el: HTMLDivElement | null) => {
      if (!el || variant !== "default") return;
      const cs = getComputedStyle(el);
      const lineHeight = Number.parseFloat(cs.lineHeight) || 16;
      const innerHeight =
        el.clientHeight -
        Number.parseFloat(cs.paddingTop) -
        Number.parseFloat(cs.paddingBottom);
      el.classList.toggle("rounded-full!", innerHeight < lineHeight * 1.5);
    },
    [variant],
  );
  return (
    <TooltipPrimitive.Portal>
      <TooltipPrimitive.Content
        ref={measureRef}
        data-slot="tooltip-content"
        sideOffset={sideOffset}
        className={cn(
          "z-[999999] w-fit max-w-xs",
          variant === "default" && "tooltip-compact",
          variant === "rich" && "tooltip-rich",
          className,
        )}
        {...props}
      >
        {children}
      </TooltipPrimitive.Content>
    </TooltipPrimitive.Portal>
  );
}

export { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger };
