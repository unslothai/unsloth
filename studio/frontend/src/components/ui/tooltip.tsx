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

import {
  getModalLayer,
  isBlockedByActiveModal,
  subscribeModalLayer,
} from "@/components/ui/tooltip-modal-layer";
import { resolveTooltipOpen } from "@/components/ui/tooltip-open-state";
import { isTouchClick } from "@/components/ui/touch-click";
import { cn } from "@/lib/utils";

type ToggleFn = () => void;
const TooltipToggleCtx = createContext<ToggleFn | null>(null);
const TooltipTriggerElementCtx = createContext<(element: HTMLElement | null) => void>(
  () => undefined,
);

function assignRef<T>(ref: React.Ref<T> | undefined, value: T | null): void {
  if (typeof ref === "function") {
    ref(value);
  } else if (ref) {
    ref.current = value;
  }
}

type ModalBlockStore = {
  getSnapshot: () => boolean;
  getTriggerElement: () => HTMLElement | null;
  setTriggerElement: (element: HTMLElement | null) => void;
  subscribe: (listener: () => void) => () => void;
};

function createModalBlockStore(): ModalBlockStore {
  let triggerElement: HTMLElement | null = null;
  let blocked = false;
  let unsubscribeModalLayer: (() => void) | null = null;
  const listeners = new Set<() => void>();

  const update = () => {
    // No trigger to ask (a child that drops the ref) falls back to the whole modal.
    const next =
      getModalLayer() &&
      (triggerElement === null || isBlockedByActiveModal(triggerElement));
    if (next === blocked) return;
    blocked = next;
    for (const listener of listeners) listener();
  };

  return {
    getSnapshot: () => blocked,
    getTriggerElement: () => triggerElement,
    setTriggerElement: (element) => {
      triggerElement = element;
      update();
    },
    subscribe: (listener) => {
      listeners.add(listener);
      if (listeners.size === 1) {
        unsubscribeModalLayer = subscribeModalLayer(update);
      }
      update();
      return () => {
        listeners.delete(listener);
        if (listeners.size === 0) {
          unsubscribeModalLayer?.();
          unsubscribeModalLayer = null;
        }
      };
    },
  };
}

function getServerModalBlock(): boolean {
  return false;
}

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
  // Hover is tracked here and `open` always supplied: `undefined` would re-expose a stale `true`.
  const [hoverOpen, setHoverOpen] = useState(false);
  const [clickOpen, setClickOpen] = useState(false);
  // A controlled tooltip missed the swallowed pointerleave; stay shut until owner says false.
  const [dismissedUntilOwnerResets, setDismissedUntilOwnerResets] =
    useState(false);
  const [modalBlockStore] = useState(createModalBlockStore);
  const blocked = useSyncExternalStore(
    modalBlockStore.subscribe,
    modalBlockStore.getSnapshot,
    getServerModalBlock,
  );

  const onOpenChange = useCallback(
    (nextOpen: boolean) => {
      setHoverOpen(nextOpen);
      if (!nextOpen) setClickOpen(false);
      controlledOnOpenChange?.(nextOpen);
    },
    [controlledOnOpenChange],
  );

  const toggle = useCallback(() => {
    setClickOpen((prev) => !prev);
  }, []);

  // Drop what is open when a modal takes over; controlled owners are told.
  useEffect(() => {
    if (!blocked) return;
    setHoverOpen(false);
    setClickOpen(false);
    setDismissedUntilOwnerResets(true);
    controlledOnOpenChange?.(false);
  }, [blocked, controlledOnOpenChange]);

  // `panel-resize-handle.tsx` passes `open` with no onOpenChange, so its say-so alone is not enough.
  useEffect(() => {
    if (controlledOpen === false) setDismissedUntilOwnerResets(false);
  }, [controlledOpen]);

  // A pin must not outlive its interaction: under a dialog Radix can no longer close it.
  useEffect(() => {
    if (!clickOpen) return;
    const release = (event: Event) => {
      const target = event.target as Node | null;
      // Matched by element, not data-slot: an `asChild` child can drop the attribute.
      if (target && modalBlockStore.getTriggerElement()?.contains(target)) {
        return;
      }
      if (
        (target as Element | null)?.closest?.('[data-slot="tooltip-content"]')
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
  }, [clickOpen, modalBlockStore]);

  return (
    <TooltipTriggerElementCtx.Provider value={modalBlockStore.setTriggerElement}>
      <TooltipToggleCtx.Provider value={isControlled ? null : toggle}>
        <TooltipPrimitive.Root
          data-slot="tooltip"
          open={resolveTooltipOpen({
            blocked,
            controlledOpen,
            dismissedUntilOwnerResets,
            hoverOpen,
            clickOpen,
          })}
          onOpenChange={onOpenChange}
          {...props}
        />
      </TooltipToggleCtx.Provider>
    </TooltipTriggerElementCtx.Provider>
  );
}

function TooltipTrigger({
  onClick,
  disableClickToggle = false,
  ref,
  ...props
}: React.ComponentProps<typeof TooltipPrimitive.Trigger> & {
  disableClickToggle?: boolean;
}) {
  const toggle = useContext(TooltipToggleCtx);
  const setTriggerElement = useContext(TooltipTriggerElementCtx);

  // The trigger stays mounted, unlike the content, so it always says which layer owns this.
  const triggerRef = useCallback(
    (el: HTMLElement | null) => {
      assignRef(ref, el);
      setTriggerElement(el);
    },
    [ref, setTriggerElement],
  );

  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLButtonElement>) => {
      // Composed handler first: a wrapped Radix trigger skips its action if already default-prevented.
      onClick?.(e);
      // With a mouse, hover already shows it and a pin only strands it.
      if (disableClickToggle || !toggle || !isTouchClick(e)) return;
      // preventDefault stops Radix's close-on-click undoing the tap-toggle below.
      e.preventDefault();
      toggle();
    },
    [disableClickToggle, toggle, onClick],
  );

  return (
    <TooltipPrimitive.Trigger
      ref={triggerRef}
      data-slot="tooltip-trigger"
      onClick={handleClick}
      {...props}
    />
  );
}

type TooltipVariant = "default" | "rich" | "none";

function TooltipContent({
  variant = "default",
  className,
  sideOffset = 0,
  collisionPadding = 8,
  children,
  ref,
  ...props
}: React.ComponentProps<typeof TooltipPrimitive.Content> & {
  variant?: TooltipVariant;
}) {
  // A ref callback, not an effect: Radix mounts the portal content without re-rendering this.
  const contentRef = useCallback(
    (el: React.ComponentRef<typeof TooltipPrimitive.Content> | null) => {
      assignRef(ref, el);
      if (!el || variant !== "default") return;
      const cs = getComputedStyle(el);
      const lineHeight = Number.parseFloat(cs.lineHeight) || 16;
      const innerHeight =
        el.clientHeight -
        Number.parseFloat(cs.paddingTop) -
        Number.parseFloat(cs.paddingBottom);
      el.classList.toggle("rounded-full!", innerHeight < lineHeight * 1.5);
    },
    [ref, variant],
  );
  return (
    <TooltipPrimitive.Portal>
      <TooltipPrimitive.Content
        ref={contentRef}
        data-slot="tooltip-content"
        sideOffset={sideOffset}
        collisionPadding={collisionPadding}
        className={cn(
          "z-[999999] w-fit max-w-xs origin-(--radix-tooltip-content-transform-origin) data-[state=delayed-open]:animate-in data-[state=delayed-open]:fade-in-0 data-[state=delayed-open]:zoom-in-95 data-[state=delayed-open]:ease-out",
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
