// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Tooltip as TooltipPrimitive } from "radix-ui";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  useSyncExternalStore,
} from "react";
import type * as React from "react";

import { cn } from "@/lib/utils";

type ToggleFn = () => void;
const TooltipToggleCtx = createContext<ToggleFn | null>(null);

// Radix sets body pointer-events to none while a modal layer is up. That is also
// when a hovered trigger stops receiving pointerleave, so a tooltip already on
// screen hangs over the dialog with nothing able to close it. One observer
// serves every tooltip.
let modalLayerUp = false;
const modalLayerListeners = new Set<() => void>();
let modalLayerObserver: MutationObserver | null = null;

function readModalLayer(): void {
  const next = document.body.style.pointerEvents === "none";
  if (next === modalLayerUp) return;
  modalLayerUp = next;
  for (const listener of modalLayerListeners) listener();
}

function subscribeModalLayer(listener: () => void): () => void {
  modalLayerListeners.add(listener);
  if (!modalLayerObserver && typeof MutationObserver !== "undefined") {
    modalLayerObserver = new MutationObserver(readModalLayer);
    modalLayerObserver.observe(document.body, {
      attributes: true,
      attributeFilter: ["style"],
    });
    readModalLayer();
  }
  return () => {
    modalLayerListeners.delete(listener);
  };
}

function getModalLayer(): boolean {
  return modalLayerUp;
}

/** Tap-to-pin is for touch, which has no hover. The click's own pointerType is
 * the only thing that answers for the pointer actually used: the media query
 * reports the primary device, so on a hybrid it mislabels every event from the
 * other one. Keyboard activation reports "", which correctly does not pin. */
function isTouchClick(event: React.MouseEvent): boolean {
  const pointerType = (event.nativeEvent as Partial<PointerEvent>).pointerType;
  if (typeof pointerType === "string") return pointerType === "touch";
  // No PointerEvent (older WebViews): fall back to the device class.
  return (
    typeof window !== "undefined" &&
    typeof window.matchMedia === "function" &&
    window.matchMedia("(pointer: coarse)").matches
  );
}

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
  const modalUp = useSyncExternalStore(
    subscribeModalLayer,
    getModalLayer,
    () => false,
  );

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

  // Drop the pin too, so it does not reappear when the dialog closes.
  useEffect(() => {
    if (modalUp) setClickOpen(false);
  }, [modalUp]);

  // A pin must not outlive its interaction: once a dialog covers the trigger,
  // pointerleave never fires and Radix can no longer close it. Presses on a
  // trigger are skipped so tapping the same one still toggles it shut.
  useEffect(() => {
    if (!clickOpen) return;
    const release = (event: Event) => {
      const target = event.target as Element | null;
      if (
        target?.closest?.(
          '[data-slot="tooltip-trigger"],[data-slot="tooltip-content"]',
        )
      ) {
        return;
      }
      setClickOpen(false);
    };
    const releaseOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setClickOpen(false);
    };
    const releaseNow = () => setClickOpen(false);
    document.addEventListener("pointerdown", release, true);
    document.addEventListener("keydown", releaseOnEscape, true);
    window.addEventListener("blur", releaseNow);
    return () => {
      document.removeEventListener("pointerdown", release, true);
      document.removeEventListener("keydown", releaseOnEscape, true);
      window.removeEventListener("blur", releaseNow);
    };
  }, [clickOpen]);

  return (
    // Controlled tooltips own their open state; pinning would never render.
    <TooltipToggleCtx.Provider value={isControlled ? null : toggle}>
      <TooltipPrimitive.Root
        data-slot="tooltip"
        // false, not undefined: a hovered tooltip has to be forced shut when a
        // dialog opens, and stay shut until it closes.
        open={
          isControlled ? controlledOpen : modalUp ? false : clickOpen || undefined
        }
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
      // With a mouse, hover already shows it and pinning only strands it. Let
      // Radix's own close-on-click run instead. `toggle` is absent when the
      // consumer controls `open`, where a pin would be dead state anyway.
      if (!toggle || !isTouchClick(e)) return;
      // preventDefault keeps Radix Tooltip's internal close-on-click from
      // undoing the tap-toggle below (its composed handler checks it).
      e.preventDefault();
      toggle();
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
